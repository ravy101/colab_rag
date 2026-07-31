import bm25s
import os
import pickle
from collections import OrderedDict

import faiss
import numpy as np
from Stemmer import Stemmer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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


class _LazyFaissShardStore:
    """
    Lazily searches FAISS shards one at a time to keep RAM bounded.

    Nothing is loaded at construction. For each query, a shard's mmapped
    index and its docstore pickle are opened, searched, matched hits are
    materialised as Documents, then the shard is released. An optional
    small LRU cache keeps the most-recently-used shards resident so
    repeated queries don't re-read the same pickle every time.

    Set cache_size to the number of shards you can afford in RAM at once.
    cache_size=1 gives the smallest footprint (one shard's docstore).
    """

    def __init__(self, shard_dirs, mmap=True, cache_size=1):
        self.shard_dirs = shard_dirs
        self.mmap = mmap
        self.cache_size = max(1, cache_size)
        self._cache = OrderedDict()  # dir -> (index, docstore, id_map)

        # Cheap up-front pass: read ntotal only, then release, so we can
        # report totals without holding anything resident.
        total = 0
        for d in shard_dirs:
            idx = self._read_index(d)
            total += idx.ntotal
            del idx
        self.total = total

    def _read_index(self, d):
        flags = faiss.IO_FLAG_MMAP if self.mmap else 0
        return faiss.read_index(os.path.join(d, "index.faiss"), flags)

    def _load_shard(self, d):
        if d in self._cache:
            self._cache.move_to_end(d)
            return self._cache[d]
        index = self._read_index(d)
        with open(os.path.join(d, "index.pkl"), "rb") as f:
            docstore, id_map = pickle.load(f)
        self._cache[d] = (index, docstore, id_map)
        self._cache.move_to_end(d)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # evict least-recently-used
        return self._cache[d]

    def search_all(self, query_vec, k):
        """Pool (Document, distance) hits across every shard."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        pooled = []
        for d in self.shard_dirs:
            index, docstore, id_map = self._load_shard(d)
            kk = min(k, index.ntotal) if index.ntotal else 0
            if kk == 0:
                continue
            distances, ids = index.search(q, kk)
            for dist, vid in zip(distances[0], ids[0]):
                if vid == -1:
                    continue
                doc_id = id_map.get(int(vid))
                if doc_id is None:
                    continue
                doc = docstore.search(doc_id)
                if isinstance(doc, Document):
                    pooled.append((doc, float(dist)))
        return pooled


class SimpleHybridRetriever:
    """
    BM25 + FAISS retriever with Reciprocal Rank Fusion.

    faiss_mmap=True memory-maps each shard's vectors AND loads shards
    lazily one at a time, so neither the full vector set nor all
    docstores are resident at once. Use faiss_cache_size to trade RAM
    for speed (number of shards kept warm; default 1 = smallest RAM).
    """

    def __init__(
        self,
        embedding_model,
        faiss_path=None,
        bm25s_path=None,
        device="cpu",
        mount_drive=True,
        bm25_mmap=False,
        faiss_mmap=False,
        faiss_cache_size=1,
    ):
        if mount_drive and _IN_COLAB:
            drive.mount("/content/drive")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"device": device},
        )

        self.faiss_mmap = faiss_mmap
        self.vector_dbs = None       # eager langchain FAISS list
        self.faiss_store = None      # lazy shard store

        if faiss_path:
            shard_dirs = sorted(
                os.path.join(faiss_path, d)
                for d in os.listdir(faiss_path)
                if d.startswith("shard_")
                and os.path.exists(
                    os.path.join(faiss_path, d, "index.faiss")
                )
            )
            if not shard_dirs and os.path.exists(
                os.path.join(faiss_path, "index.faiss")
            ):
                shard_dirs = [faiss_path]
            if not shard_dirs:
                raise FileNotFoundError(
                    f"No FAISS index found under {faiss_path}."
                )

            if faiss_mmap:
                self.faiss_store = _LazyFaissShardStore(
                    shard_dirs, mmap=True, cache_size=faiss_cache_size
                )
                print(
                    f"FAISS ready (lazy mmap): {len(shard_dirs)} shard(s), "
                    f"{self.faiss_store.total} items, "
                    f"cache_size={faiss_cache_size}."
                )
            else:
                self.vector_dbs = [
                    FAISS.load_local(
                        d, self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    for d in shard_dirs
                ]
                total = sum(
                    len(v.index_to_docstore_id) for v in self.vector_dbs
                )
                print(
                    f"FAISS loaded: {len(self.vector_dbs)} shard(s), "
                    f"{total} items."
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

        # --- Dense (FAISS) ---  distances: lower is better -> sort ascending
        dense_pooled = []
        if self.faiss_store is not None:
            query_vec = self.embeddings.embed_query(query)
            dense_pooled = self.faiss_store.search_all(query_vec, fetch_k)
        elif self.vector_dbs is not None:
            for v in self.vector_dbs:
                for doc, dist in v.similarity_search_with_score(
                    query, k=fetch_k
                ):
                    dense_pooled.append((doc, dist))

        if dense_pooled:
            dense_pooled.sort(key=lambda x: x[1])  # ascending distance
            for rank, (doc, _dist) in enumerate(dense_pooled[:fetch_k]):
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                if key not in doc_map:
                    doc_map[key] = doc

        # --- Sparse (BM25, sharded) ---  scores: higher is better
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


