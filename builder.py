import bm25s
import os
import json
import bm25s
import shutil
from google.colab import drive
from datasets import load_dataset
from Stemmer import Stemmer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- UTILITIES ---
def get_state(db_dir, tag = ""):   
    STATE_FILE = os.path.join(db_dir, tag + "progress.json")
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: 
            return json.load(f)
    return {"last_index": 0, "doc_ids": []}

def save_state(db_dir, idx, doc_ids, tag = ""):
    STATE_FILE = os.path.join(db_dir, tag + "progress.json")
    with open(STATE_FILE, 'w') as f:
        json.dump({"last_index": idx, "doc_ids": doc_ids}, f)

def build_database(db_dir, total_target = None, build_faiss=True, build_bm25=True, batch_size=5000, embedding_model= "sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    drive.mount('/content/drive')
    #DB_DIR = "/content/drive/My Drive/hybrid_wiki_index"

   
    FAISS_PATH = os.path.join(db_dir, "faiss_index")
    BM25_PATH = os.path.join(db_dir, "bm25s_index")


    os.makedirs(FAISS_PATH, exist_ok=True)
    os.makedirs(BM25_PATH, exist_ok=True)

    if device == "cpu":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': "cpu"})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': device})
    stemmer = Stemmer("english")

    # --- MAIN LOOP ---
    state = get_state(db_dir)
    current_idx = state["last_index"]
    all_doc_texts = [] # We'll need the full corpus for BM25 re-indexing

    print(f"Resuming from index: {current_idx}")

    # Load Dataset
    ds = load_dataset("CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary", 
                    "simple", split="train", streaming=True)
    it = iter(ds)
    for _ in range(current_idx): next(it)

    if total_target is None:
        total_target = 1000000000000

    while current_idx < total_target:
        batch_docs = []
        batch_texts = []
        
        for _ in range(batch_size):
            try:
                entry = next(it)
                doc = Document(page_content=entry['text'], metadata={"title": entry['title']})
                batch_docs.append(doc)
                batch_texts.append(entry['text'])
            except StopIteration: break
        
        if not batch_docs: 
            break

        print(f"--- Processing Batch: {current_idx} to {current_idx + len(batch_docs)} ---")
        
        # 1. Update FAISS (Dense)
        new_faiss = FAISS.from_documents(batch_docs, embeddings)
        if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
            vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_db.merge_from(new_faiss)
            vector_db.save_local(FAISS_PATH)
        else:
            new_faiss.save_local(FAISS_PATH)

        # 2. Update BM25S (Sparse)
        current_corpus = []
        
        manifest_path = os.path.join(BM25_PATH, "params.index.json")
        
        if os.path.exists(manifest_path):
            try:
                # load_corpus=True is mandatory for your incremental append strategy
                old_retriever = bm25s.BM25.load(BM25_PATH, load_corpus=True)
                
                if old_retriever.corpus is not None:
                    current_corpus = [c['text'] for c in list(old_retriever.corpus)]
                    print(f"Resumed: Loaded {len(current_corpus)} documents from BM25.")
                else:
                    print("Index found but corpus is empty. Starting fresh.")
            except Exception as e:
                print(f"BM25 Resume Failed: {e}")
            else:
                print("No index, starting.")
        
        current_corpus.extend(batch_texts)
        
        # Re-index
        tokens = bm25s.tokenize(current_corpus, stemmer=stemmer)
        retriever = bm25s.BM25(corpus=current_corpus)
        retriever.index(tokens)
        
        # MANUAL OVERWRITE: Clear the folder first to avoid conflict
        if os.path.exists(BM25_PATH):
            shutil.rmtree(BM25_PATH)
        os.makedirs(BM25_PATH, exist_ok=True)
        
        # Save fresh
        retriever.save(BM25_PATH, corpus=current_corpus)

        current_idx += len(batch_docs)
        save_state(db_dir, current_idx, []) 
        print(f"Success. Total docs in Hybrid Store: {current_idx}")

    print("\nAll indices built and saved to Drive.")


def build_faiss_database(db_dir, total_target = None, batch_size=5000, embedding_model= "sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    drive.mount('/content/drive')
    #DB_DIR = "/content/drive/My Drive/hybrid_wiki_index"

    FAISS_PATH = os.path.join(db_dir, "faiss_index")


    os.makedirs(FAISS_PATH, exist_ok=True)

    if device == "cpu":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': "cpu"})
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': device})

    # --- MAIN LOOP ---
    state = get_state(db_dir)
    current_idx = state["last_index"]

    print(f"Resuming from index: {current_idx}")

    # Load Dataset
    ds = load_dataset("CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary", 
                    "simple", split="train", streaming=True)
    it = iter(ds)
    for _ in range(current_idx): next(it)

    if total_target is None:
        total_target = 1000000000000

    while current_idx < total_target:
        batch_docs = []
        batch_texts = []
        
        for _ in range(batch_size):
            try:
                entry = next(it)
                doc = Document(page_content=entry['text'], metadata={"title": entry['title']})
                batch_docs.append(doc)
                batch_texts.append(entry['text'])
            except StopIteration: break
        
        if not batch_docs: 
            break

        print(f"--- Processing Batch: {current_idx} to {current_idx + len(batch_docs)} ---")
        
        # 1. Update FAISS (Dense)
        new_faiss = FAISS.from_documents(batch_docs, embeddings)
        if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
            vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_db.merge_from(new_faiss)
            vector_db.save_local(FAISS_PATH)
        else:
            new_faiss.save_local(FAISS_PATH)


        current_idx += len(batch_docs)
        save_state(db_dir, current_idx, []) 
        print(f"Success. Total docs in FAISS Store: {current_idx}")

    print("\nAll indices built and saved to Drive.")

    
def build_bm25_database(db_dir, total_target = None, build_bm25=True, batch_size=5000):
    drive.mount('/content/drive')
   
    BM25_PATH = os.path.join(db_dir, "bm25s_index")

    os.makedirs(BM25_PATH, exist_ok=True)

    stemmer = Stemmer("english")

    # --- MAIN LOOP ---
    state = get_state(db_dir, tag="bm_25")
    current_idx = state["last_index"]
    all_doc_texts = [] # We'll need the full corpus for BM25 re-indexing

    print(f"Resuming from index: {current_idx}")

    # Load Dataset
    ds = load_dataset("CohereLabs/wikipedia-2023-11-embed-multilingual-v3-int8-binary", 
                    "simple", split="train", streaming=True)
    it = iter(ds)
    for _ in range(current_idx): next(it)

    if total_target is None:
        total_target = 1000000000000

    while current_idx < total_target:
        batch_docs = []
        batch_texts = []
        
        for _ in range(batch_size):
            try:
                entry = next(it)
                doc = Document(page_content=entry['text'], metadata={"title": entry['title']})
                batch_docs.append(doc)
                batch_texts.append(entry['text'])
            except StopIteration: 
                break
        
        if not batch_docs: 
            break

        print(f"--- Processing Batch: {current_idx} to {current_idx + len(batch_docs)} ---")
        

        # 2. Update BM25S (Sparse)
        current_corpus = []
        
        manifest_path = os.path.join(BM25_PATH, "params.index.json")
        
        if os.path.exists(manifest_path):
            try:
                # load_corpus=True is mandatory for your incremental append strategy
                old_retriever = bm25s.BM25.load(BM25_PATH, load_corpus=True)
                
                if old_retriever.corpus is not None:
                    current_corpus = [c['text'] for c in list(old_retriever.corpus)]
                    print(f"Resumed: Loaded {len(current_corpus)} documents from BM25.")
                else:
                    print("Index found but corpus is empty. Starting fresh.")
            except Exception as e:
                print(f"BM25 Resume Failed: {e}")
            else:
                print("No index, starting.")
        
        current_corpus.extend(batch_texts)
        
        # Re-index
        tokens = bm25s.tokenize(current_corpus, stemmer=stemmer)
        retriever = bm25s.BM25(corpus=current_corpus)
        retriever.index(tokens)
        
        # MANUAL OVERWRITE: Clear the folder first to avoid conflict
        if os.path.exists(BM25_PATH):
            shutil.rmtree(BM25_PATH)
        os.makedirs(BM25_PATH, exist_ok=True)
        
        # Save fresh
        retriever.save(BM25_PATH, corpus=current_corpus)

        current_idx += len(batch_docs)
        save_state(db_dir, current_idx, [], tag="bm_25") 
        print(f"Success. Total docs in Hybrid Store: {current_idx}")

    print("\nAll indices built and saved to Drive.")