"""
Hybrid retriever for the multiaxial cascade retrieval axis.

Combines BM25 (sparse) and FAISS (dense) retrieval via Reciprocal Rank
Fusion. Designed to consume indexes built by retrieval_builder.py, which
stores the BM25 corpus as a list of dicts with `text`, `title`, and
chunk-locator fields. Falls back gracefully to plain-string corpora for
backwards compatibility with older indexes.
"""

import bm25s
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
    # Fallback: stringify
    return Document(page_content=str(rec), metadata={})


def _content_key(doc_or_rec):
    """Stable key for deduplication across BM25 and FAISS results."""
    if isinstance(doc_or_rec, Document):
        # Prefer (article_idx, chunk_idx) when available; fall back to text.
        meta = doc_or_rec.metadata or {}
        if "article_idx" in meta and "chunk_idx" in meta:
            return ("idx", meta["article_idx"], meta["chunk_idx"])
        return ("txt", doc_or_rec.page_content)
    if isinstance(doc_or_rec, dict):
        if "article_idx" in doc_or_rec and "chunk_idx" in doc_or_rec:
            return ("idx", doc_or_rec["article_idx"], doc_or_rec["chunk_idx"])
        return ("txt", doc_or_rec.get("text", ""))
    return ("txt", str(doc_or_rec))


class SimpleHybridRetriever:
    """
    BM25 + FAISS retriever with Reciprocal Rank Fusion.

    Args:
        embedding_model: HF model id for the FAISS embedder. Must match the
                         model used to build the FAISS index.
        faiss_path: Directory containing FAISS index. None to disable dense.
        bm25s_path: Directory containing BM25 index. None to disable sparse.
        device: "cpu" or "cuda" for the embedding model.
        mount_drive: Mount Google Drive (Colab only).
    """

    def __init__(
        self,
        embedding_model,
        faiss_path=None,
        bm25s_path=None,
        device="cpu",
        mount_drive=True,
    ):
        if mount_drive and _IN_COLAB:
            drive.mount("/content/drive")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"device": device},
        )

        if faiss_path:
            self.vector_db = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(
                f"FAISS loaded: "
                f"{len(self.vector_db.index_to_docstore_id)} items."
            )
        else:
            self.vector_db = None

        if bm25s_path:
            self.retriever_bm25 = bm25s.BM25.load(
                bm25s_path, load_corpus=True
            )
            corpus_n = (
                len(self.retriever_bm25.corpus)
                if self.retriever_bm25.corpus is not None
                else 0
            )
            print(f"BM25 loaded: {corpus_n} items.")
        else:
            self.retriever_bm25 = None

        self.stemmer = Stemmer("english")

    def hybrid_retrieve(self, query, k=5, rrf_k=60, fetch_multiplier=2):
        """
        Hybrid search via Reciprocal Rank Fusion.

        Args:
            query: Query string.
            k: Number of fused results to return.
            rrf_k: RRF damping constant (60 is the standard default).
            fetch_multiplier: Per-retriever fetch depth as a multiple of k.
                              k * fetch_multiplier candidates are pulled
                              from each retriever before fusion.

        Returns:
            List of dicts: {"doc": Document, "score": float}
        """
        fetch_k = k * fetch_multiplier
        rrf_scores = {}
        doc_map = {}

        # --- Dense (FAISS) ---
        if self.vector_db is not None:
            dense_results = self.vector_db.similarity_search_with_score(
                query, k=fetch_k
            )
            for rank, (doc, _faiss_score) in enumerate(dense_results):
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                if key not in doc_map:
                    doc_map[key] = doc

        # --- Sparse (BM25) ---
        if self.retriever_bm25 is not None:
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
            sparse_docs, _sparse_scores = self.retriever_bm25.retrieve(
                query_tokens, k=fetch_k
            )
            # bm25s returns shape (n_queries, k); we have one query.
            for rank, rec in enumerate(sparse_docs[0]):
                doc = _doc_record_to_document(rec)
                key = _content_key(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (
                    rrf_k + (rank + 1)
                )
                # Prefer existing entry (likely from FAISS, may have richer
                # metadata via langchain docstore).
                if key not in doc_map:
                    doc_map[key] = doc

        # Sort and return top-k
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
            meta = r['doc'].metadata
            key = (meta.get('article_idx', -1), meta.get('chunk_idx', -1))
            if key == (-1, -1):
                key = ('txt', r['doc'].page_content[:200])
            if key not in all_results or r['score'] > all_results[key]['score']:
                all_results[key] = r
    return sorted(all_results.values(),
                  key=lambda x: x['score'],
                  reverse=True)[:k]

def consolidate_context(retrieval_results, threshold=0.015):
    """
    Format retrieval results into a single context string for the LLM.

    Filters out results below an RRF score threshold. The default threshold
    of 0.015 corresponds to roughly the 60th rank under standard RRF (k=60),
    i.e. results that appeared only once at low rank in either retriever.
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
