"""
embeddings.py
--------------
Wraps a sentence-transformers model so the rest of the app never has to
think about the underlying transformer model directly. Embeddings are
L2-normalized so that a FAISS inner-product index behaves like cosine
similarity (this is a standard trick: cosine_sim(a, b) == dot(a, b) when
||a|| == ||b|| == 1).
"""

from typing import List, Union
import numpy as np


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily so the rest of the app (e.g. just inspecting the
        # FAISS index) doesn't need torch/sentence-transformers installed.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode one string or a list of strings into a (n, dim) float32
        numpy array of L2-normalized embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
