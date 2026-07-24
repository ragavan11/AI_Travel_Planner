"""
vector_store.py
-----------------
A small, dependency-light wrapper around FAISS that:
  1) stores embeddings in an IndexFlatIP (inner product) index,
  2) keeps the original text + metadata for each vector alongside it,
  3) can persist itself to disk and reload later, so the index only
     has to be built once (see build_index.py).

Because embeddings are normalized (see embeddings.py), inner product
search here is equivalent to cosine-similarity search.
"""

import os
import pickle
from typing import List, Dict, Any

import numpy as np
import faiss


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Add a batch of embeddings + their corresponding documents."""
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top_k most similar documents to a single query
        embedding. Returns a list of {"score": float, "document": dict}.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), "document": self.documents[idx]})
        return results

    def save(self, folder: str) -> None:
        """Persist the FAISS index and the document store to disk."""
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "documents.pkl"), "wb") as f:
            pickle.dump(
                {"dimension": self.dimension, "documents": self.documents}, f
            )

    @classmethod
    def load(cls, folder: str) -> "FaissVectorStore":
        """Reload a previously saved index + document store from disk."""
        with open(os.path.join(folder, "documents.pkl"), "rb") as f:
            payload = pickle.load(f)

        store = cls(payload["dimension"])
        store.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        store.documents = payload["documents"]
        return store

    @staticmethod
    def exists(folder: str) -> bool:
        return os.path.exists(os.path.join(folder, "index.faiss")) and os.path.exists(
            os.path.join(folder, "documents.pkl")
        )
