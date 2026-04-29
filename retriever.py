import bm25s
from google.colab import drive
from Stemmer import Stemmer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class SimpleHybridRetriever:
    def __init__(self, embedding_model, faiss_path = None, bm25s_path = None, device='cpu'):
        drive.mount('/content/drive')

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, 
                                                model_kwargs={'device': 'cpu'},
                                                encode_kwargs={'device': 'cpu'})
                                                    
        if faiss_path:
            self.vector_db = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Vector DB loaded with {len(self.vector_db.index_to_docstore_id)} items.")
        else:
            self.vector_db = None
        
        if bm25s_path:
            self.retriever_bm25 = bm25s.BM25.load(bm25s_path, load_corpus=True)
            print(f"BM25s store loaded with {len(self.retriever_bm25.corpus)} items.")
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

def consolidate_context(retrieval_results, threshold = 0.15):
    """
    Consolidates RRF results into a single string for LLM prompting.
    """
    context_blocks = []

    for i, entry in enumerate(retrieval_results):
        doc = entry['doc']
        score = entry['score']
        if score < threshold:
            continue
        # We include the index and a header to help the LLM cite its sources
        header = f"[Source {i+1} | RRF Score: {score:.4f}]"
        block = f"{header}\n{doc.page_content.strip()}"
        context_blocks.append(block)
    prefix = "Supporting context sources may or may not contain the required information.\n"
    return prefix + "\n\n".join(context_blocks)

