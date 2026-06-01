"""
Retrieval index builders for the multiaxial cascade retrieval axis.

Builds BM25 (sparse) and FAISS (dense) indexes over a streaming HuggingFace
dataset. Default corpus is wikimedia/wikipedia 20231101.en (full English
Wikipedia, ~6.7M articles).

Key design choices:
  - Append-only JSONL corpus on disk during streaming (O(1) per chunk).
  - BM25 indexed once at the end over the full corpus (avoids the O(N^2)
    re-indexing pattern that is intractable for full Wikipedia).
  - FAISS supports incremental add_documents and is checkpointed per batch.
  - Resume state tracks *articles* processed (not chunks), so resuming after
    a crash skips exactly the right amount on the streaming iterator.
  - Paragraph-level chunking via RecursiveCharacterTextSplitter; whole-article
    indexing hurts retrieval precision and inflates context windows.
  - Field names default to wikimedia/wikipedia (text, title) and are
    parameterised for other corpora.
"""

import os
import json
import shutil
import bm25s
from datasets import load_dataset
from Stemmer import Stemmer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from google.colab import drive
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False


# --- STATE / RESUME UTILITIES ---

def get_state(db_dir, tag=""):
    """Load resume state. Tracks last *article* index and chunk count."""
    state_file = os.path.join(db_dir, tag + "progress.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"last_article_index": 0, "n_chunks_written": 0}


def save_state(db_dir, article_idx, n_chunks, tag=""):
    state_file = os.path.join(db_dir, tag + "progress.json")
    os.makedirs(db_dir, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(
            {"last_article_index": article_idx,
             "n_chunks_written": n_chunks},
            f,
        )


def _maybe_mount_drive(mount_drive):
    if mount_drive and _IN_COLAB:
        drive.mount("/content/drive")


def _default_dataset():
    """Default corpus: full English Wikipedia."""
    return load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )


def _make_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _iter_chunks(entry, splitter, text_field, title_field, min_chunk_chars):
    """Yield (chunk_text, title, chunk_idx) for one dataset entry."""
    text = entry.get(text_field, "") or ""
    title = entry.get(title_field, "") or ""
    if len(text) < min_chunk_chars:
        return
    for i, chunk in enumerate(splitter.split_text(text)):
        if len(chunk) < min_chunk_chars:
            continue
        yield chunk, title, i



# --- BM25 BUILDER ---

def build_bm25_database(
    db_dir,
    total_target=None,
    batch_size=10000,
    hf_dataset=None,
    text_field="text",
    title_field="title",
    chunk_size=1000,
    chunk_overlap=150,
    min_chunk_chars=100,
    mount_drive=True,
    corpus_filename="corpus.jsonl",
):
    """
    Build a BM25 index over a HuggingFace dataset.

    Strategy:
      1. Stream and chunk articles, appending each chunk to a JSONL file.
         This is O(1) per chunk, resumable, and survives crashes.
      2. Once streaming completes (or total_target is reached), tokenise
         and index the full corpus in one pass.

    Args:
        db_dir: Output directory for index, corpus, and state.
        total_target: Max number of *articles* to process (None = all).
        batch_size: Articles per state-flush. Bigger is fine for BM25
                    (work per chunk is just append).
        hf_dataset: Pre-loaded dataset, or None to load wikimedia/wikipedia.
        text_field: Article body field. Default "text" (wikimedia/wikipedia).
        title_field: Article title field. Default "title".
        chunk_size: Target chunk size in characters (~4 chars/token).
        chunk_overlap: Char overlap between adjacent chunks.
        min_chunk_chars: Drop chunks shorter than this (filters fragments).
        mount_drive: Whether to mount Google Drive (Colab only).
        corpus_filename: Name of the append-only JSONL corpus file.

    Returns:
        Total number of chunks indexed.
    """
    _maybe_mount_drive(mount_drive)

    bm25_path = os.path.join(db_dir, "bm25s_index")
    corpus_path = os.path.join(db_dir, corpus_filename)
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(bm25_path, exist_ok=True)

    stemmer = Stemmer("english")
    splitter = _make_splitter(chunk_size, chunk_overlap)

    # Resume
    state = get_state(db_dir, tag="bm25_")
    current_article_idx = state["last_article_index"]
    n_chunks_written = state["n_chunks_written"]
    print(
        f"Resuming from article index {current_article_idx} "
        f"({n_chunks_written} chunks already written)."
    )

    # Dataset
    ds = hf_dataset if hf_dataset is not None else _default_dataset()
    it = iter(ds)
    for _ in range(current_article_idx):
        try:
            next(it)
        except StopIteration:
            print("Resume index exceeds dataset length; nothing to stream.")
            break

    if total_target is None:
        total_target = float("inf")

    # Append-only chunk write loop
    with open(corpus_path, "a", encoding="utf-8") as corpus_f:
        while current_article_idx < total_target:
            batch_articles = 0
            batch_chunks = 0

            for _ in range(batch_size):
                try:
                    entry = next(it)
                except StopIteration:
                    break

                article_idx_for_record = current_article_idx + batch_articles
                for chunk, title, chunk_i in _iter_chunks(
                    entry, splitter, text_field, title_field, min_chunk_chars
                ):
                    rec = {
                        "text": chunk,
                        "title": title,
                        "article_idx": article_idx_for_record,
                        "chunk_idx": chunk_i,
                    }
                    corpus_f.write(
                        json.dumps(rec, ensure_ascii=False) + "\n"
                    )
                    batch_chunks += 1
                batch_articles += 1

            if batch_articles == 0:
                print("Stream exhausted.")
                break

            current_article_idx += batch_articles
            n_chunks_written += batch_chunks
            corpus_f.flush()
            save_state(
                db_dir, current_article_idx, n_chunks_written, tag="bm25_"
            )
            print(
                f"Articles: {current_article_idx} | "
                f"Chunks: {n_chunks_written} | "
                f"+{batch_articles} articles, +{batch_chunks} chunks"
            )

    # Final one-shot BM25 index build
    print("\nBuilding BM25 index over full corpus...")
    n_total = _build_bm25_index_from_corpus(corpus_path, bm25_path, stemmer)
    print(f"Done. {n_total} chunks indexed at {bm25_path}")
    return n_total


def _build_bm25_index_from_corpus(corpus_path, bm25_path, stemmer):
    """Read JSONL corpus and build a fresh BM25 index in one pass."""
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    corpus_records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            corpus_records.append(rec)

    if not corpus_records:
        raise RuntimeError("Empty corpus; nothing to index.")

    texts = [r["text"] for r in corpus_records]
    print(f"Tokenising {len(texts)} chunks...")
    tokens = bm25s.tokenize(texts, stemmer=stemmer)

    print("Indexing...")
    # bm25s preserves whatever you pass as `corpus` and returns it on retrieve.
    # We pass the full record dict so retrieval has metadata available.
    retriever = bm25s.BM25(corpus=corpus_records)
    retriever.index(tokens)

    # Clean overwrite of the index dir
    if os.path.exists(bm25_path):
        shutil.rmtree(bm25_path)
    os.makedirs(bm25_path, exist_ok=True)
    retriever.save(bm25_path, corpus=corpus_records)

    return len(corpus_records)



# --- FAISS BUILDER ---

def build_faiss_database(
    db_dir,
    total_target=None,
    batch_size=5000,
    hf_dataset=None,
    text_field="text",
    title_field="title",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    chunk_size=1000,
    chunk_overlap=150,
    min_chunk_chars=100,
    mount_drive=True,
    save_every_n_batches=1,
    reuse_corpus_jsonl=True,
    corpus_filename="corpus.jsonl",
):
    """
    Build a FAISS dense index over a HuggingFace dataset.

    If a chunk corpus JSONL already exists in db_dir (e.g. from the BM25
    builder), it is reused so chunking is not redone. Otherwise the corpus
    is generated alongside indexing.

    Resume semantics match the BM25 builder: state tracks article index
    when streaming from HF, or chunk index when consuming a JSONL corpus.

    Args:
        db_dir: Output directory.
        total_target: Max articles (when streaming) or chunks (when reusing
                      JSONL). None = all.
        batch_size: Items per FAISS add + checkpoint.
        hf_dataset: Pre-loaded HF dataset, or None for wikimedia/wikipedia.
        text_field, title_field: Field names in the source dataset.
        embedding_model: HF model id for embeddings.
        device: "cpu" or "cuda".
        chunk_size, chunk_overlap, min_chunk_chars: chunker config (only
                      used if not reusing an existing JSONL corpus).
        mount_drive: Whether to mount Drive (Colab).
        save_every_n_batches: Persist FAISS to disk every N batches. Saving
                      every batch is safest but slower for large indexes.
        reuse_corpus_jsonl: If True and corpus_filename exists in db_dir,
                      embed from it directly (skips re-chunking). Strongly
                      recommended when BM25 index was built first.
        corpus_filename: Name of the JSONL corpus.

    Returns:
        Total number of chunks indexed.
    """
    _maybe_mount_drive(mount_drive)

    faiss_path = os.path.join(db_dir, "faiss_index")
    corpus_path = os.path.join(db_dir, corpus_filename)
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(faiss_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"device": device},
    )

    use_jsonl = reuse_corpus_jsonl and os.path.exists(corpus_path)

    if use_jsonl:
        print(f"Reusing existing chunk corpus at {corpus_path}")
        return _build_faiss_from_jsonl(
            corpus_path=corpus_path,
            faiss_path=faiss_path,
            db_dir=db_dir,
            embeddings=embeddings,
            batch_size=batch_size,
            total_target=total_target,
            save_every_n_batches=save_every_n_batches,
        )
    else:
        print("No chunk corpus found; streaming + chunking from HF dataset.")
        return _build_faiss_from_stream(
            db_dir=db_dir,
            faiss_path=faiss_path,
            corpus_path=corpus_path,
            embeddings=embeddings,
            hf_dataset=hf_dataset,
            text_field=text_field,
            title_field=title_field,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            batch_size=batch_size,
            total_target=total_target,
            save_every_n_batches=save_every_n_batches,
        )


def _load_or_init_faiss(faiss_path, embeddings, initial_docs):
    """Load FAISS if present, else initialise from initial_docs."""
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        return FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
    if not initial_docs:
        return None
    return FAISS.from_documents(initial_docs, embeddings)


def _build_faiss_from_jsonl(
    corpus_path,
    faiss_path,
    db_dir,
    embeddings,
    batch_size,
    total_target,
    save_every_n_batches,
):
    state = get_state(db_dir, tag="faiss_")
    start_chunk_idx = state.get("last_article_index", 0)
    n_indexed = state.get("n_chunks_written", 0)
    print(
        f"Resuming FAISS from chunk {start_chunk_idx} "
        f"({n_indexed} previously indexed)."
    )

    if total_target is None:
        total_target = float("inf")

    vector_db = None
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        vector_db = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )

    batch_docs = []
    batches_since_save = 0
    current_idx = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if current_idx < start_chunk_idx:
                current_idx += 1
                continue
            if current_idx >= total_target:
                break

            rec = json.loads(line)
            batch_docs.append(
                Document(
                    page_content=rec["text"],
                    metadata={
                        "title": rec.get("title", ""),
                        "article_idx": rec.get("article_idx", -1),
                        "chunk_idx": rec.get("chunk_idx", -1),
                    },
                )
            )
            current_idx += 1

            if len(batch_docs) >= batch_size:
                vector_db = _flush_faiss_batch(
                    vector_db, batch_docs, embeddings, faiss_path,
                    save=(batches_since_save + 1 >= save_every_n_batches),
                )
                n_indexed += len(batch_docs)
                save_state(db_dir, current_idx, n_indexed, tag="faiss_")
                print(
                    f"FAISS: {n_indexed} chunks indexed "
                    f"(read up to chunk {current_idx})."
                )
                batch_docs = []
                batches_since_save = (
                    0 if (batches_since_save + 1 >= save_every_n_batches)
                    else batches_since_save + 1
                )

    # Flush trailing batch
    if batch_docs:
        vector_db = _flush_faiss_batch(
            vector_db, batch_docs, embeddings, faiss_path, save=True
        )
        n_indexed += len(batch_docs)
        save_state(db_dir, current_idx, n_indexed, tag="faiss_")

    if vector_db is not None:
        vector_db.save_local(faiss_path)
    print(f"\nFAISS done. {n_indexed} chunks indexed at {faiss_path}")
    return n_indexed


def _flush_faiss_batch(vector_db, batch_docs, embeddings, faiss_path, save):
    if vector_db is None:
        vector_db = FAISS.from_documents(batch_docs, embeddings)
    else:
        vector_db.add_documents(batch_docs)
    if save:
        vector_db.save_local(faiss_path)
    return vector_db



def _build_faiss_from_stream(
    db_dir,
    faiss_path,
    corpus_path,
    embeddings,
    hf_dataset,
    text_field,
    title_field,
    chunk_size,
    chunk_overlap,
    min_chunk_chars,
    batch_size,
    total_target,
    save_every_n_batches,
):
    splitter = _make_splitter(chunk_size, chunk_overlap)

    state = get_state(db_dir, tag="faiss_")
    current_article_idx = state["last_article_index"]
    n_chunks_indexed = state["n_chunks_written"]
    print(
        f"Resuming FAISS stream-build from article {current_article_idx} "
        f"({n_chunks_indexed} chunks indexed)."
    )

    ds = hf_dataset if hf_dataset is not None else _default_dataset()
    it = iter(ds)
    for _ in range(current_article_idx):
        try:
            next(it)
        except StopIteration:
            print("Resume index exceeds dataset length.")
            break

    if total_target is None:
        total_target = float("inf")

    vector_db = None
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        vector_db = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )

    batches_since_save = 0
    with open(corpus_path, "a", encoding="utf-8") as corpus_f:
        while current_article_idx < total_target:
            batch_docs = []
            articles_in_batch = 0

            for _ in range(batch_size):
                try:
                    entry = next(it)
                except StopIteration:
                    break

                article_idx_for_record = (
                    current_article_idx + articles_in_batch
                )
                for chunk, title, chunk_i in _iter_chunks(
                    entry, splitter, text_field, title_field, min_chunk_chars
                ):
                    rec = {
                        "text": chunk,
                        "title": title,
                        "article_idx": article_idx_for_record,
                        "chunk_idx": chunk_i,
                    }
                    corpus_f.write(
                        json.dumps(rec, ensure_ascii=False) + "\n"
                    )
                    batch_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "title": title,
                                "article_idx": article_idx_for_record,
                                "chunk_idx": chunk_i,
                            },
                        )
                    )
                articles_in_batch += 1

            if articles_in_batch == 0:
                break

            if batch_docs:
                save_now = (batches_since_save + 1 >= save_every_n_batches)
                vector_db = _flush_faiss_batch(
                    vector_db, batch_docs, embeddings, faiss_path,
                    save=save_now,
                )
                n_chunks_indexed += len(batch_docs)
                batches_since_save = (
                    0 if save_now else batches_since_save + 1
                )

            current_article_idx += articles_in_batch
            corpus_f.flush()
            save_state(
                db_dir, current_article_idx, n_chunks_indexed, tag="faiss_"
            )
            print(
                f"FAISS: articles {current_article_idx} | "
                f"chunks indexed {n_chunks_indexed} "
                f"(+{len(batch_docs)} this batch)"
            )

    if vector_db is not None:
        vector_db.save_local(faiss_path)
    print(f"\nFAISS done. {n_chunks_indexed} chunks indexed at {faiss_path}")
    return n_chunks_indexed


# --- CONVENIENCE: build full pipeline ---

def build_hybrid_database(
    db_dir,
    total_target=None,
    bm25_batch_size=10000,
    faiss_batch_size=5000,
    hf_dataset=None,
    text_field="text",
    title_field="title",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    chunk_size=1000,
    chunk_overlap=150,
    min_chunk_chars=100,
    mount_drive=True,
):
    """
    Convenience wrapper: build BM25 first (cheap, fast, validates corpus),
    then FAISS reusing the BM25 chunk corpus (no re-chunking).
    """
    print("=== Phase 1: BM25 ===")
    n_bm25 = build_bm25_database(
        db_dir=db_dir,
        total_target=total_target,
        batch_size=bm25_batch_size,
        hf_dataset=hf_dataset,
        text_field=text_field,
        title_field=title_field,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
        mount_drive=mount_drive,
    )

    print("\n=== Phase 2: FAISS ===")
    n_faiss = build_faiss_database(
        db_dir=db_dir,
        total_target=total_target,
        batch_size=faiss_batch_size,
        hf_dataset=hf_dataset,
        text_field=text_field,
        title_field=title_field,
        embedding_model=embedding_model,
        device=device,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
        mount_drive=mount_drive,
        reuse_corpus_jsonl=True,
    )
    print(f"\nHybrid build complete: {n_bm25} BM25 chunks, "
          f"{n_faiss} FAISS chunks.")
    return n_bm25, n_faiss
