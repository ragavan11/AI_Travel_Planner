"""
app.py
-------
Streamlit front-end for the AI Travel Planner.

Run with:
    streamlit run app.py

On first run, if no FAISS index exists yet under index/, the app builds
one automatically from data/travel_knowledge_base.json.
"""

import streamlit as st

from src import config
from src.data_loader import load_all_documents
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore
from src.rag_pipeline import TravelPlannerRAG

st.set_page_config(page_title="AI Travel Planner", layout="wide")

def load_all_documents():
    # your code here
    pass

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel(config.EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner="Loading / building knowledge base index...")
def get_vector_store(_embedder: EmbeddingModel) -> FaissVectorStore:
    if FaissVectorStore.exists(str(config.INDEX_DIR)):
        return FaissVectorStore.load(str(config.INDEX_DIR))

    # Build the index on the fly if it doesn't exist yet.
    documents = load_all_documents(config.DATA_PATH)
    texts = [doc["text"] for doc in documents]
    embeddings = _embedder.encode(texts)

    store = FaissVectorStore(_embedder.dimension)
    store.add(embeddings, documents)
    store.save(str(config.INDEX_DIR))
    return store


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(_embedder: EmbeddingModel, _store: FaissVectorStore) -> TravelPlannerRAG:
    return TravelPlannerRAG(_embedder, _store)

st.title("AI Travel Planner")
st.caption("Generative AI + RAG (FAISS + sentence-transformers + LLM) travel itinerary generator")

with st.sidebar:
    st.header("Trip preferences")

    destination = st.text_input(
        "Destination",
        placeholder="e.g. Goa, Paris, Tokyo, Bali...",
        help="Works best with destinations in the knowledge base (see README), "
        "but the LLM will do its best for any destination.",
    )
    duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=4)
    num_travelers = st.number_input("Number of travelers", min_value=1, max_value=20, value=2)
    budget_level = st.selectbox("Budget level", ["budget", "mid-range", "luxury"], index=1)
    interests = st.multiselect(
        "Interests",
        [
            "nature",
            "adventure",
            "history & heritage",
            "food",
            "beaches",
            "mountains",
            "city life",
            "arts & music",
            "sports",
            "nightlife",
            "religion & spirituality",
            "restorants & cafes",
            "shopping",
            "relaxation",
            "culture",
            "wildlife",
            "photography",
        ],
        default=["food", "culture"],
    )

    generate_clicked = st.button("Generate itinerary", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"LLM provider: **{config.LLM_PROVIDER}**")
    st.caption("Change LLM_PROVIDER / API keys in your .env file.")

if generate_clicked:
    if not destination.strip():
        st.warning("Please enter a destination first.")
        st.stop()

    try:
        embedder = get_embedding_model()
        store = get_vector_store(embedder)
        pipeline = get_rag_pipeline(embedder, store)
    except Exception as e:
        st.error(f"Failed to initialize the RAG pipeline: {e}")
        st.stop()

    with st.spinner("Retrieving relevant travel info and generating your itinerary..."):
        try:
            result = pipeline.generate_itinerary(
                destination=destination,
                duration_days=int(duration_days),
                budget_level=budget_level,
                interests=interests,
                num_travelers=int(num_travelers),
            )
        except Exception as e:
            st.error(
                f"Failed to generate itinerary: {e}\n\n"
                "Check that your LLM_PROVIDER and the matching API key "
                "(GOOGLE_API_KEY or OPENAI_API_KEY) are set correctly in .env."
            )
            st.stop()

    st.markdown(result["itinerary"])

    with st.expander("🔎 Retrieved context used for this itinerary (RAG transparency)"):
        for r in result["retrieved_context"]:
            meta = r["document"]["metadata"]
            st.markdown(
                f"**[{meta['category']}] {meta['destination']}** "
                f"(similarity score: {r['score']:.3f})\n\n{r['document']['text']}"
            )
            st.divider()
else:
    st.info(
        "👈 Fill in your trip preferences in the sidebar and click "
        "**Generate itinerary** to get a personalized, day-wise travel plan."
    )
    st.markdown(
        """
### How this works (RAG architecture)
1. Your destination + interests + budget are turned into a search query.
2. That query is embedded and matched against a **FAISS** vector index built
   from a travel knowledge base (attractions, hotels, food, transport, tips).
3. The most relevant chunks are retrieved and passed to an **LLM** as context.
4. The LLM generates a grounded, day-wise itinerary using that retrieved
   context instead of relying purely on its own (possibly outdated) memory.
"""
    )
