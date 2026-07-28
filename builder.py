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
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # fallback for older langchain installs
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


def _count_corpus_chunks(corpus_path):
    n = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def _restore_faiss_from_backup(faiss_path, db_dir):
    """Replace the primary index + progress with the verified backup.

    Returns True if a usable backup was restored, False otherwise. Used
    when resuming after a crash that may have truncated the primary
    (e.g. an interrupted save_local over a Drive mount).
    """
    backup_dir = os.path.join(db_dir, "faiss_index_backup")
    backup_progress = os.path.join(backup_dir, "faiss_progress.json")
    primary_progress = os.path.join(db_dir, "faiss_progress.json")

    if not os.path.exists(os.path.join(backup_dir, "index.faiss")):
        print("  no backup found; cannot restore. Starting from primary/scratch.")
        return False

    # Wipe possibly-truncated primary, replace with backup.
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
    shutil.copytree(backup_dir, faiss_path)

    # Restore the snapshotted progress so index + counter agree.
    src = backup_progress if os.path.exists(backup_progress) else None
    if src:
        shutil.copy2(src, primary_progress)
        # Remove the nested copy so it isn't mistaken for index data.
        nested = os.path.join(faiss_path, "faiss_progress.json")
        if os.path.exists(nested):
            os.remove(nested)
    print(f"  restored primary index + progress from backup: {backup_dir}")
    return True


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


import gc

def _verify_faiss(faiss_path, embeddings):
    """True only if the primary index loads and its docstore is readable."""
    try:
        vdb = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
        _ = len(vdb.index_to_docstore_id)
        return True
    except Exception as e:
        print(f"  BACKUP SKIPPED — primary failed verify: {e}")
        return False


def _refresh_faiss_backup(faiss_path, db_dir):
    """Atomically replace the single backup slot with the current index.

    Copies primary -> tmp, snapshots progress into tmp, then swaps tmp into
    the backup slot. A crash mid-copy leaves the previous backup intact.
    """
    backup_dir = os.path.join(db_dir, "faiss_index_backup")
    backup_tmp = os.path.join(db_dir, "faiss_index_backup_tmp")
    progress_src = os.path.join(db_dir, "faiss_progress.json")

    if os.path.exists(backup_tmp):
        shutil.rmtree(backup_tmp)
    shutil.copytree(faiss_path, backup_tmp)
    if os.path.exists(progress_src):
        shutil.copy2(
            progress_src, os.path.join(backup_tmp, "faiss_progress.json")
        )
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    os.rename(backup_tmp, backup_dir)
    print(f"  backup refreshed -> {backup_dir}")

# def _build_bm25_index_from_corpus(
#     corpus_path, bm25_path, stemmer, shard_size=2_000_000
# ):
#     """Build a BM25 index with sharded tokenisation (bm25s 0.3.10).

#     bm25s.tokenize has no shared-vocab argument, so each shard is
#     tokenised independently and its local token ids are remapped into a
#     single global vocab before the id lists are concatenated. Indexes
#     once at the end (no O(N^2) re-indexing). Output is a single index
#     dir, load-compatible with the retriever's BM25.load(load_corpus=True).
#     """
#     if not os.path.exists(corpus_path):
#         raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

#     corpus_records = []
#     with open(corpus_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             corpus_records.append(json.loads(line))

#     if not corpus_records:
#         raise RuntimeError("Empty corpus; nothing to index.")

#     n = len(corpus_records)
#     print(f"Tokenising {n} chunks in shards of {shard_size}...")

#     global_vocab = {}          # token string -> global id
#     all_ids = []               # list of per-doc lists of global ids

#     for start in range(0, n, shard_size):
#         end = min(start + shard_size, n)
#         shard_texts = [r["text"] for r in corpus_records[start:end]]

#         tok = bm25s.tokenize(
#             shard_texts,
#             stemmer=stemmer,
#             return_ids=True,
#             show_progress=False,
#         )
#         # tok.vocab: token -> local_id ; tok.ids: list of lists of local ids
#         local_id_to_token = {v: k for k, v in tok.vocab.items()}

#         # Build a local->global id translation for this shard's vocab.
#         local_to_global = {}
#         for token, local_id in tok.vocab.items():
#             g = global_vocab.get(token)
#             if g is None:
#                 g = len(global_vocab)
#                 global_vocab[token] = g
#             local_to_global[local_id] = g

#         # Remap this shard's doc-id-lists into global ids.
#         for doc_ids in tok.ids:
#             all_ids.append([local_to_global[i] for i in doc_ids])

#         del shard_texts, tok, local_id_to_token, local_to_global
#         gc.collect()
#         print(f"  tokenised {end}/{n}  (vocab size {len(global_vocab)})")

#     print("Indexing (single pass)...")
#     tokens = bm25s.tokenization.Tokenized(ids=all_ids, vocab=global_vocab)
#     retriever = bm25s.BM25(corpus=corpus_records)
#     retriever.index(tokens)

#     del all_ids, tokens, global_vocab
#     gc.collect()

#     if os.path.exists(bm25_path):
#         shutil.rmtree(bm25_path)
#     os.makedirs(bm25_path, exist_ok=True)
#     retriever.save(bm25_path, corpus=corpus_records)

#     del corpus_records, retriever
#     gc.collect()
#     return n
class _TokenizedCorpus:
    """Re-iterable stream of token-string lists, one per corpus line.

    Re-opens the JSONL and re-tokenises on every __iter__, so bm25s can
    iterate it more than once (vocab pass + postings pass) without us
    holding the whole tokenised corpus in RAM. This is what a bare
    generator could not do — a generator is exhausted after one pass,
    which produced the empty score matrix.
    """
    def __init__(self, corpus_path, stemmer):
        self.corpus_path = corpus_path
        self.stemmer = stemmer

    def __iter__(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"]
                tok = bm25s.tokenize(
                    text, stemmer=self.stemmer,
                    return_ids=False, show_progress=False,
                )
                yield tok[0]   # one document's token-string list


class _CorpusRecords:
    """Re-iterable stream of record dicts for BM25.save(corpus=...)."""
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _build_bm25_index_from_corpus(corpus_path, bm25_path, stemmer):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print("Indexing BM25 from re-iterable token stream...")
    retriever = bm25s.BM25()
    retriever.index(_TokenizedCorpus(corpus_path, stemmer))

    if os.path.exists(bm25_path):
        shutil.rmtree(bm25_path)
    os.makedirs(bm25_path, exist_ok=True)
    retriever.save(bm25_path, corpus=_CorpusRecords(corpus_path))

    n = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n



def build_bm25_sharded(corpus_path, bm25_root, shard_chunks=2000000):
    """Build BM25 as independent, resumable shards over an existing corpus.

    Each shard is a complete bm25s index over `shard_chunks` consecutive
    corpus lines, saved immediately. Existing shards are skipped, so a
    crash/disconnect only costs the in-progress shard. No monolithic
    in-memory matrix, so peak RAM is bounded by one shard.
    """
    import os, json, shutil, bm25s
    from Stemmer import Stemmer

    stemmer = Stemmer("english")
    os.makedirs(bm25_root, exist_ok=True)

    def shard_dir(i):
        return os.path.join(bm25_root, f"shard_{i:04d}")

    def _done(i):
        # a shard is complete only if its data array was actually written
        d = shard_dir(i)
        p = os.path.join(d, "data.csc.index.npy")
        return os.path.exists(p) and os.path.getsize(p) > 1024

    shard_i = 0
    buf = []

    def flush(idx, records):
        d = shard_dir(idx)
        if _done(idx):
            print(f"  shard {idx} already complete, skipping")
            return
        if os.path.exists(d):
            shutil.rmtree(d)  # remove any partial/truncated shard
        os.makedirs(d, exist_ok=True)
        texts = [r["text"] for r in records]
        tokens = bm25s.tokenize(texts, stemmer=stemmer, show_progress=False)
        r = bm25s.BM25(corpus=records)
        r.index(tokens)
        r.save(d, corpus=records)
        print(f"  shard {idx} saved: {len(records)} chunks -> {d}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.append(json.loads(line))
            if len(buf) >= shard_chunks:
                if not _done(shard_i):
                    flush(shard_i, buf)
                else:
                    print(f"  shard {shard_i} already complete, skipping")
                buf = []
                shard_i += 1

    if buf:  # final partial shard
        if not _done(shard_i):
            flush(shard_i, buf)
        shard_i += 1

    print(f"Done. {shard_i} shards at {bm25_root}")
    return shard_i

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
    backup_every=2_000_000,
    resume_from_backup=False,
):
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
            backup_every=backup_every,
            resume_from_backup=resume_from_backup,
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
            backup_every=backup_every,
            resume_from_backup=resume_from_backup,
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
    backup_every=2_000_000,
    resume_from_backup=False,
):
    if resume_from_backup:
        _restore_faiss_from_backup(faiss_path, db_dir)

    state = get_state(db_dir, tag="faiss_")
    start_chunk_idx = state.get("last_article_index", 0)  # chunk offset on disk
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
    last_backup_at = n_indexed
    current_idx = 0  # chunk counter (line number), used for resume position

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if current_idx < start_chunk_idx:
                current_idx += 1
                continue

            rec = json.loads(line)

            # Gate on ARTICLE index, so total_target means articles here too.
            if rec.get("article_idx", -1) >= total_target:
                break

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

                # Periodic verified single-slot backup.
                if n_indexed - last_backup_at >= backup_every:
                    if vector_db is not None:
                        vector_db.save_local(faiss_path)
                        if _verify_faiss(faiss_path, embeddings):
                            _refresh_faiss_backup(faiss_path, db_dir)
                            last_backup_at = n_indexed

    # Flush trailing batch
    if batch_docs:
        vector_db = _flush_faiss_batch(
            vector_db, batch_docs, embeddings, faiss_path, save=True
        )
        n_indexed += len(batch_docs)
        save_state(db_dir, current_idx, n_indexed, tag="faiss_")

    if vector_db is not None:
        vector_db.save_local(faiss_path)
        if _verify_faiss(faiss_path, embeddings):
            _refresh_faiss_backup(faiss_path, db_dir)

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
    backup_every=2_000_000,
    resume_from_backup=False,
):
    splitter = _make_splitter(chunk_size, chunk_overlap)

    if resume_from_backup:
        _restore_faiss_from_backup(faiss_path, db_dir)

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
    last_backup_at = n_chunks_indexed
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

            # Periodic verified single-slot backup.
            if n_chunks_indexed - last_backup_at >= backup_every:
                if vector_db is not None:
                    vector_db.save_local(faiss_path)
                    if _verify_faiss(faiss_path, embeddings):
                        _refresh_faiss_backup(faiss_path, db_dir)
                        last_backup_at = n_chunks_indexed

    if vector_db is not None:
        vector_db.save_local(faiss_path)
        if _verify_faiss(faiss_path, embeddings):
            _refresh_faiss_backup(faiss_path, db_dir)
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
