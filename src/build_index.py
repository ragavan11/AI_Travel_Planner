"""
build_index.py
----------------
One-time (or whenever data/travel_knowledge_base.json changes) script
that:
  1) loads + chunks the travel knowledge base,
  2) embeds every chunk with sentence-transformers,
  3) builds a FAISS index, and
  4) saves it to disk under index/.

Run this once before starting the Streamlit app:
    python -m src.build_index
"""

from . import config
from .data_loader import load_all_documents
from .embeddings import EmbeddingModel
from .vector_store import FaissVectorStore


def main():
    print(f"Loading knowledge base from {config.DATA_PATH} ...")
    documents = load_all_documents(config.DATA_PATH)
    print(f"Loaded {len(documents)} text chunks.")

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' ...")
    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)

    print("Embedding chunks (this may take a moment on first run)...")
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(texts)

    print(f"Building FAISS index (dimension={embedder.dimension}) ...")
    store = FaissVectorStore(embedder.dimension)
    store.add(embeddings, documents)

    print(f"Saving index to {config.INDEX_DIR} ...")
    store.save(str(config.INDEX_DIR))
    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
