"""
Hybrid retriever for the multiaxial cascade retrieval axis.

Combines BM25 (sparse) and FAISS (dense) retrieval via Reciprocal Rank
Fusion. Designed to consume indexes built by retrieval_builder.py.

Supports both sharded FAISS (a root containing shard_XXXX/ dirs, as written
by build_faiss_sharded) and a single monolithic FAISS index directory.
When faiss_mmap=True, each shard's vectors are memory-mapped via
faiss.read_index(IO_FLAG_MMAP) and queried lazily one shard at a time,
so the full float32 vector set is never resident at once.
"""

import bm25s
import os
import pickle

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
    """Convert a BM25 corpus record (dict or str) into a langchain Document."""
    if isinstance(rec, str):
        return Document(page_content=rec, metadata={})
    if isinstance(rec, dict):
        text = rec.get("text", "")
        meta = {k: v for k, v in rec.items() if k != "text"}
        return Document(page_content=text, metadata=meta)
    return Document(page_content=str(rec), metadata={})


def _content_key(doc_or_rec):
    """Stable key for deduplication across BM25 and FAISS results."""
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


class _MmapFaissShard:
    """
    A single FAISS shard whose vectors are memory-mapped, not loaded.

    Opens index.faiss with IO_FLAG_MMAP so the float32 vectors stay on
    disk and are paged in on demand. The docstore (index.pkl) is read
    only to resolve matched ids -> Documents; we do not hold a langchain
    FAISS wrapper (which would keep the whole docstore resident via its
    in-memory dict). page_content/metadata are pulled per-hit.
    """

    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        self.index = faiss.read_index(
            os.path.join(shard_dir, "index.faiss"),
            faiss.IO_FLAG_MMAP,
        )
        # langchain saves (docstore, index_to_docstore_id) as a tuple.
        with open(os.path.join(shard_dir, "index.pkl"), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        self._docstore = docstore
        self._index_to_id = index_to_docstore_id

    @property
    def ntotal(self):
        return self.index.ntotal

    def search(self, query_vec, k):
        """Return list of (Document, distance) for the top-k in this shard."""
        # query_vec: shape (dim,) float32
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(k, self.index.ntotal) if self.index.ntotal else 0
        if k == 0:
            return []
        distances, ids = self.index.search(q, k)
        out = []
        for dist, vid in zip(distances[0], ids[0]):
            if vid == -1:
                continue
            doc_id = self._index_to_id.get(int(vid))
            if doc_id is None:
                continue
            doc = self._docstore.search(doc_id)
            if isinstance(doc, Document):
                out.append((doc, float(dist)))
        return out


class SimpleHybridRetriever:
    """
    BM25 + FAISS retriever with Reciprocal Rank Fusion.

    Args:
        embedding_model: HF model id for the FAISS embedder. Must match the
                         model used to build the FAISS index.
        faiss_path: Directory containing FAISS index (monolithic dir, or a
                    root of shard_XXXX/ dirs). None to disable dense.
        bm25s_path: Directory containing BM25 index. None to disable sparse.
        device: "cpu" or "cuda" for the embedding model.
        mount_drive: Mount Google Drive (Colab only).
        bm25_mmap: Memory-map BM25 shards.
        faiss_mmap: Memory-map FAISS shard vectors and query lazily
                    (keeps the full vector set off the heap).
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
    ):
        if mount_drive and _IN_COLAB:
            drive.mount("/content/drive")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"device": device},
        )

        self.faiss_mmap = faiss_mmap
        self.vector_dbs = None      # list of langchain FAISS (non-mmap path)
        self.faiss_shards = None    # list of _MmapFaissShard (mmap path)

        if faiss_path:
            shard_dirs = sorted(
                os.path.join(faiss_path, d)
                for d in os.listdir(faiss_path)
                if d.startswith("shard_")
                and os.path.exists(
                    os.path.join(faiss_path, d, "index.faiss")
                )
            )
            # Monolithic fallback: treat the path itself as one "shard".
            if not shard_dirs and os.path.exists(
                os.path.join(faiss_path, "index.faiss")
            ):
                shard_dirs = [faiss_path]

            if not shard_dirs:
                raise FileNotFoundError(
                    f"No FAISS index found under {faiss_path} "
                    f"(no shard_XXXX/index.faiss and no index.faiss)."
                )

            if faiss_mmap:
                self.faiss_shards = [
                    _MmapFaissShard(d) for d in shard_dirs
                ]
                total = sum(s.ntotal for s in self.faiss_shards)
                print(
                    f"FAISS loaded (mmap): {len(self.faiss_shards)} "
                    f"shard(s), {total} items."
                )
            else:
                self.vector_dbs = [
                    FAISS.load_local(
                        d,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    for d in shard_dirs
                ]
                total = sum(
                    len(v.index_to_docstore_id) for v in self.vector_dbs
                )
                print(
                    f"FAISS loaded: {len(self.vector_dbs)} "
                    f"shard(s), {total} items."
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
        """
        Hybrid search via Reciprocal Rank Fusion.

        Returns:
            List of dicts: {"doc": Document, "score": float}
        """
        fetch_k = k * fetch_multiplier
        rrf_scores = {}
        doc_map = {}

        # --- Dense (FAISS, sharded) ---
        # Distances are "lower is better", so pool across shards and sort
        # ASCENDING before assigning a single global rank for RRF.
        dense_pooled = []  # (distance, doc)

        if self.faiss_shards is not None:
            # mmap path: embed once, query each shard, release as we go.
            query_vec = self.embeddings.embed_query(query)
            for shard in self.faiss_shards:
                for doc, dist in shard.search(query_vec, fetch_k):
                    dense_pooled.append((dist, doc))
        elif self.vector_dbs is not None:
            for v in self.vector_dbs:
                results = v.similarity_search_with_score(query, k=fetch_k)
                for doc, dist in results:
                    dense_pooled.append((dist, doc))

        if dense_pooled:
            dense_pooled.sort(key=lambda x: x[0])  # ascending distance
            for rank, (_dist, doc) in enumerate(dense_pooled[:fetch_k]):
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                if key not in doc_map:
                    doc_map[key] = doc

        # --- Sparse (BM25, sharded) ---
        if self.retrievers_bm25 is not None:
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
            pooled = []  # (raw_score, record)
            for r in self.retrievers_bm25:
                docs, scores = r.retrieve(query_tokens, k=fetch_k)
                for rec, sc in zip(docs[0], scores[0]):
                    pooled.append((sc, rec))
            # BM25 scores are "higher is better" -> sort DESCENDING.
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
        """
        Per-choice MCQA retrieval with cross-choice deduplication.
        Returns top-k unique chunks across all choice-conditioned queries.
        """
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


