import bm25s
import os
import json
import bm25s
from google.colab import drive
from datasets import load_dataset
from Stemmer import Stemmer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- UTILITIES ---
def get_state(db_dir, target_):   
    STATE_FILE = os.path.join(db_dir, "progress.json")
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: 
            return json.load(f)
    return {"last_index": 0, "doc_ids": []}

def save_state(db_dir, idx, doc_ids):
    STATE_FILE = os.path.join(db_dir, "progress.json")
    with open(STATE_FILE, 'w') as f:
        json.dump({"last_index": idx, "doc_ids": doc_ids}, f)

def build_database(db_dir, total_target = None, batch_size=5000, embedding_model= "sentence-transformers/all-MiniLM-L6-v2"):
    drive.mount('/content/drive')
    #DB_DIR = "/content/drive/My Drive/hybrid_wiki_index"

    FAISS_PATH = os.path.join(db_dir, "faiss_index")
    BM25_PATH = os.path.join(db_dir, "bm25s_index")


    os.makedirs(FAISS_PATH, exist_ok=True)
    os.makedirs(BM25_PATH, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
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
        
        if not batch_docs: break

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
        # Note: BM25S is fast but prefers full-corpus indexing. 
        # For incremental, we append texts and re-index the sparse matrix.
        # We retrieve the previous texts if they exist to keep the index whole.
        current_corpus = []
        if os.path.exists(os.path.join(BM25_PATH, "data.csc.npy")):
            # Load existing corpus to append
            old_retriever = bm25s.BM25.load(BM25_PATH, load_corpus=True)
            current_corpus = list(old_retriever.corpus)
        
        current_corpus.extend(batch_texts)
        
        # Re-index BM25 (Very fast with bm25s)
        tokens = bm25s.tokenize(current_corpus, stemmer=stemmer)
        retriever = bm25s.BM25(corpus=current_corpus)
        retriever.index(tokens)
        retriever.save(BM25_PATH, corpus=current_corpus)

        current_idx += len(batch_docs)
        save_state(db_dir, current_idx, []) 
        print(f"Success. Total docs in Hybrid Store: {current_idx}")

    print("\nAll indices built and saved to Drive.")

class SimpleHybridRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", faiss_path = None, bm25s_path = None):
        drive.mount('/content/drive')
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        if faiss_path:
            self.vector_db = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_db = None
        
        if bm25s_path:
            self.retriever_bm25 = bm25s.BM25.load(bm25s_path, load_corpus=True)
        else:
            self.retriever_bm25 = None
        self.stemmer = Stemmer("english")

    def hybrid_retrieve(self, query, k=5, rrf_k=60):
        """
        Performs hybrid search and merges results using Reciprocal Rank Fusion.
        k: Number of documents to return
        rrf_k: Constant for RRF (standard is 60)
        """
        # --- 1. Dense Retrieval (FAISS) ---
        # similarity_search_with_score returns (Document, score)
        # Note: FAISS scores here are usually L2 distances (lower is better)
        dense_results = self.vector_db.similarity_search_with_score(query, k=k*2)
        
        # --- 2. Sparse Retrieval (BM25S) ---
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        # BM25S returns (documents, scores)
        sparse_docs, sparse_scores = self.retriever_bm25.retrieve(query_tokens, k=k*2)
        
        # --- 3. Reciprocal Rank Fusion Logic ---
        rrf_scores = {} # Map of doc_content -> fused_score
        doc_map = {}    # Map of doc_content -> Document object for reconstruction

        # Process Dense Ranks
        for rank, (doc, _) in enumerate(dense_results):
            content = doc.page_content
            rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + (rank + 1))
            doc_map[content] = doc

        # Process Sparse Ranks
        # bm25s returns a 2D array (queries x k), we take the first query row [0]
        for rank, doc_text in enumerate(sparse_docs[0]):
            content = doc_text if isinstance(doc_text, str) else doc_text['text']
            rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (rrf_k + (rank + 1))
            if content not in doc_map:
                # Create a Document object if it only appeared in BM25
                doc_map[content] = Document(page_content=content)

        # Sort by fused score descending
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k results as (Document, RRF_Score)
        final_output = []
        for content, score in sorted_results[:k]:
            final_output.append({
                "doc": doc_map[content],
                "score": score
            })
            
        return final_output

def consolidate_context(retrieval_results):
    """
    Consolidates RRF results into a single string for LLM prompting.
    """
    context_blocks = []

    for i, entry in enumerate(retrieval_results):
        doc = entry['doc']
        score = entry['score']
        # We include the index and a header to help the LLM cite its sources
        header = f"[Source {i+1} | RRF Score: {score:.4f}]"
        block = f"{header}\n{doc.page_content.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

