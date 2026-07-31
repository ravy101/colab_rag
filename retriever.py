import bm25s
import os
import json
import pickle

import faiss
import time
import numpy as np
from Stemmer import Stemmer
from langchain_core.documents import Document

try:
    from google.colab import drive
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False


def _doc_record_to_document(rec):
    if isinstance(rec, str):
        return Document(page_content=rec, metadata={})
    if isinstance(rec, dict):
        text = rec.get("text", "")
        meta = {k: v for k, v in rec.items() if k != "text"}
        return Document(page_content=text, metadata=meta)
    return Document(page_content=str(rec), metadata={})


def _content_key(doc_or_rec):
    if isinstance(doc_or_rec, Document):
        meta = doc_or_rec.metadata or {}
        if "article_idx" in meta and "chunk_idx" in meta:
            return ("idx", meta["article_idx"], meta["chunk_idx"])
        return ("txt", doc_or_rec.page_content)
    if isinstance(doc_or_rec, dict):
        if "article_idx" in doc_or_rec and "chunk_idx" in doc_or_rec:
            return ("idx", doc_or_rec["article_idx"], doc_or_rec["chunk_idx"])
        return ("txt", doc_or_rec.get("text", ""))
    return ("txt", str(doc_or_rec))


class _LeanFaissStore:
    def __init__(self, faiss_root, corpus_path, mmap=True):
        self.faiss_root = faiss_root
        self.corpus_path = corpus_path
        self.mmap = mmap

        t0 = time.time()
        off_path = os.path.join(faiss_root, "corpus_offsets.pkl")
        sz = os.path.getsize(off_path) / 1e6
        print(f"[faiss] loading offset index ({sz:,.1f} MB) ...", flush=True)
        with open(off_path, "rb") as f:
            self.offsets = pickle.load(f)
        print(f"[faiss] offsets: {len(self.offsets):,} entries in "
              f"{time.time()-t0:,.1f}s", flush=True)

        self.shard_dirs = sorted(
            os.path.join(faiss_root, d)
            for d in os.listdir(faiss_root)
            if d.startswith("shard_")
            and os.path.exists(os.path.join(faiss_root, d, "index.faiss"))
        )
        if not self.shard_dirs:
            raise FileNotFoundError(f"No shards under {faiss_root}")
        print(f"[faiss] {len(self.shard_dirs)} shards found", flush=True)

        self.indexes = []
        self.locators = []
        total = 0
        for i, d in enumerate(self.shard_dirs, 1):
            ts = time.time()
            flags = faiss.IO_FLAG_MMAP if mmap else 0
            idx = faiss.read_index(os.path.join(d, "index.faiss"), flags)
            t_idx = time.time() - ts
            tl = time.time()
            with open(os.path.join(d, "locators.pkl"), "rb") as f:
                loc = pickle.load(f)
            t_loc = time.time() - tl
            self.indexes.append(idx)
            self.locators.append(loc)
            total += idx.ntotal
            print(f"[faiss]   shard {i}/{len(self.shard_dirs)} "
                  f"{os.path.basename(d)}: {idx.ntotal:,} vecs "
                  f"(index {t_idx:,.1f}s, locators {t_loc:,.1f}s)",
                  flush=True)

        self._corpus_f = open(corpus_path, "rb")
        print(f"[faiss] READY: {len(self.shard_dirs)} shards, "
              f"{total:,} items.", flush=True)

    def _fetch_text(self, article_idx, chunk_idx):
        pos = self.offsets.get((article_idx, chunk_idx))
        if pos is None:
            return ""
        self._corpus_f.seek(pos)
        line = self._corpus_f.readline()
        try:
            return json.loads(line)["text"]
        except Exception:
            return ""

    def search_all(self, query_vec, k, verbose=False):
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        pooled = []
        for i, (idx, loc) in enumerate(zip(self.indexes, self.locators), 1):
            kk = min(k, idx.ntotal) if idx.ntotal else 0
            if kk == 0:
                continue
            ts = time.time()
            distances, ids = idx.search(q, kk)
            for dist, vid in zip(distances[0], ids[0]):
                if vid == -1:
                    continue
                entry = loc.get(int(vid))
                if entry is None:
                    continue
                aidx, cidx, title = entry
                text = self._fetch_text(aidx, cidx)
                pooled.append((Document(
                    page_content=text,
                    metadata={"title": title,
                              "article_idx": aidx, "chunk_idx": cidx},
                ), float(dist)))
            if verbose:
                print(f"[search] shard {i}/{len(self.indexes)} "
                      f"searched in {time.time()-ts:,.2f}s", flush=True)
        return pooled


class SimpleHybridRetriever:
    """
    BM25 + FAISS retriever with Reciprocal Rank Fusion.

    FAISS uses a lean store: vectors mmapped, text streamed from
    corpus.jsonl, docstore never loaded. Requires the migration script
    (corpus_offsets.pkl + per-shard locators.pkl) to have been run.
    """

    def __init__(
        self,
        embedding_model,
        faiss_path=None,
        bm25s_path=None,
        corpus_path=None,
        device="cpu",
        mount_drive=True,
        bm25_mmap=False,
        faiss_mmap=True,
    ):
        if mount_drive and _IN_COLAB:
            drive.mount("/content/drive")

        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"device": device},
        )

        self.faiss_store = None
        if faiss_path:
            if corpus_path is None:
                raise ValueError(
                    "corpus_path is required for the lean FAISS store."
                )
            self.faiss_store = _LeanFaissStore(
                faiss_path, corpus_path, mmap=faiss_mmap
            )

        if bm25s_path:
            shard_dirs = sorted(
                os.path.join(bm25s_path, d)
                for d in os.listdir(bm25s_path)
                if d.startswith("shard_")
            )
            self.retrievers_bm25 = [
                bm25s.BM25.load(d, load_corpus=True, mmap=bm25_mmap)
                for d in shard_dirs
            ]
            total = sum(
                len(r.corpus) if r.corpus is not None else 0
                for r in self.retrievers_bm25
            )
            print(
                f"BM25 loaded: {len(self.retrievers_bm25)} shards, "
                f"{total} items."
            )
        else:
            self.retrievers_bm25 = None

        self.stemmer = Stemmer("english")

    def hybrid_retrieve(self, query, k=5, rrf_k=60, fetch_multiplier=2):
        fetch_k = k * fetch_multiplier
        rrf_scores = {}
        doc_map = {}

        # --- Dense (FAISS) --- distances lower is better -> sort ascending
        if self.faiss_store is not None:
            query_vec = self.embeddings.embed_query(query)
            dense_pooled = self.faiss_store.search_all(query_vec, fetch_k)
            dense_pooled.sort(key=lambda x: x[1])
            for rank, (doc, _dist) in enumerate(dense_pooled[:fetch_k]):
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                if key not in doc_map:
                    doc_map[key] = doc

        # --- Sparse (BM25, sharded) --- scores higher is better
        if self.retrievers_bm25 is not None:
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
            pooled = []
            for r in self.retrievers_bm25:
                docs, scores = r.retrieve(query_tokens, k=fetch_k)
                for rec, sc in zip(docs[0], scores[0]):
                    pooled.append((sc, rec))
            pooled.sort(key=lambda x: x[0], reverse=True)
            for rank, (_sc, rec) in enumerate(pooled[:fetch_k]):
                doc = _doc_record_to_document(rec)
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                if key not in doc_map:
                    doc_map[key] = doc

        sorted_keys = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {"doc": doc_map[key], "score": score}
            for key, score in sorted_keys[:k]
        ]

    def mcqa_hybrid_retrieve(self, question, choices, k=5, k_per_choice=3):
        all_results = {}
        for choice in choices:
            query = f"{question} {choice}"
            for r in self.hybrid_retrieve(query, k=k_per_choice):
                meta = r["doc"].metadata
                key = (meta.get("article_idx", -1), meta.get("chunk_idx", -1))
                if key == (-1, -1):
                    key = ("txt", r["doc"].page_content[:200])
                if key not in all_results or r["score"] > all_results[key]["score"]:
                    all_results[key] = r
        return sorted(
            all_results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:k]


def consolidate_context(retrieval_results, threshold=0.015):
    """
    Format retrieval results into a single context string for the LLM.

    Filters out results below an RRF score threshold. The default threshold
    of 0.015 corresponds to roughly the 60th rank under standard RRF (k=60).
    """
    context_blocks = []
    for i, entry in enumerate(retrieval_results):
        doc = entry["doc"]
        score = entry["score"]
        if score < threshold:
            continue
        title = (doc.metadata or {}).get("title", "unknown subject")
        header = (
            f"[Source {i + 1} | RRF Score: {score:.4f} | Subject: {title}]"
        )
        block = f"{header}\n{doc.page_content.strip()}"
        context_blocks.append(block)

    prefix = (
        "Supporting context sources may or may not contain the required "
        "information.\n"
        "Supporting sources may refer to other entities if not specified.\n"
        "If sources are irrelevant, give your best guess as a short answer.\n"
        "Do not cite sources.\n"
        "Do not mention sources at all.\n"
        "Do not mention context.\n"
        "Only output the direct answer to the question.\n"
        "If you use the word 'source' or 'context' I will punish you.\n"
    )
    return prefix + "\n\n".join(context_blocks)


