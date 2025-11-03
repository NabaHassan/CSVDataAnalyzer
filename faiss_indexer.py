import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSIndex:
    def __init__(self, index_path: str, meta_path: str):
        """Load FAISS index and corresponding metadata"""
        self.index = faiss.read_index(index_path)

        # ✅ Load metadata (the original texts used for embeddings)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        # ✅ Load same embedding model used during indexing
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, query: str, k: int = 5):
        """Search for top-k most similar entries to the query"""
        # Convert query into an embedding
        query_vec = self.model.encode([query], convert_to_numpy=True)

        # Perform FAISS similarity search
        D, I = self.index.search(np.array(query_vec, dtype="float32"), k)

        # Map top-k indices back to metadata (text rows)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results
