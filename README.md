# 🧭 AI Travel Planner using Generative AI and RAG

An AI-powered travel itinerary generator. The user enters a destination,
trip duration, budget level, and interests; the app retrieves relevant
travel knowledge (attractions, accommodation, food, transport, tips,
budgets) using **semantic search over a FAISS vector database**, then
asks an **LLM** to turn that retrieved context into a personalized,
day-wise itinerary — a complete Retrieval-Augmented Generation (RAG)
pipeline.

## Architecture

```
                ┌────────────────────────────┐
                │ travel_knowledge_base.json │  (10 destinations: attractions,
                └──────────┬─────────────────┘    hotels, food, transport, tips, budget)
                           │  chunked into ~80 small documents
                           ▼
                ┌──────────────────────┐
                │ sentence-transformers│  (all-MiniLM-L6-v2, 384-dim)
                │   EmbeddingModel     │
                └──────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │   FAISS IndexFlatIP │  <-- vector database (index/)
                └──────────┬──────────┘
                           ▲  cosine-similarity search
                           │
  User input (destination, duration, budget, interests)
                           │
                           ▼
                ┌─────────────────────┐
                │   Retrieved chunks  │  (top-k relevant context)
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │   LLM (Gemini /     │  --> Day-wise itinerary,
                │   OpenAI GPT)       │      budget breakdown, tips
                └─────────────────────┘
                           ▼
                ┌─────────────────────┐
                │   Streamlit UI      │
                └─────────────────────┘
```

## Project structure

```
ai-travel-planner/
├── app.py                       # Streamlit UI (entry point)
├── requirements.txt
├── .env.example                 # copy to .env and fill in API keys
├── data/
│   └── travel_knowledge_base.json   # sample dataset (10 destinations)
├── index/                       # FAISS index gets saved here (auto-generated)
└── src/
    ├── config.py                # env-var driven settings
    ├── data_loader.py           # JSON -> chunked text documents
    ├── embeddings.py            # sentence-transformers wrapper
    ├── vector_store.py          # FAISS index build/search/save/load
    ├── llm_client.py            # Gemini / OpenAI abstraction
    ├── rag_pipeline.py          # retrieval + prompt + generation
    └── build_index.py           # standalone script to (re)build the index
```

## How the RAG pipeline works

1. **Chunking** (`data_loader.py`): each destination record is split into
   small, focused chunks — one per attraction, one per accommodation
   tier, one per food item, etc. — instead of embedding the whole record
   as a single blob. This makes retrieval far more precise.
2. **Embedding** (`embeddings.py`): every chunk is converted into a
   384-dimensional vector using the `all-MiniLM-L6-v2` transformer model
   (via `sentence-transformers`), running locally — no API key or cost.
3. **Indexing** (`vector_store.py`): all vectors are stored in a FAISS
   `IndexFlatIP` index (inner product on normalized vectors = cosine
   similarity), alongside their original text + metadata.
4. **Retrieval** (`rag_pipeline.py`): the user's destination/interests/
   budget are turned into a natural-language query, embedded, and used
   to search FAISS for the top-k most relevant chunks.
5. **Generation** (`llm_client.py` + `rag_pipeline.py`): the retrieved
   chunks are inserted into a prompt as grounding context, and the LLM
   (Gemini or OpenAI, your choice) generates the final itinerary.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
# then edit .env and paste in your GOOGLE_API_KEY or OPENAI_API_KEY

# 4. (Optional) Pre-build the FAISS index - the app will also build it
#    automatically on first run if you skip this step.
python -m src.build_index

# 5. Run the app
streamlit run app.py
```

The first run will download the `all-MiniLM-L6-v2` embedding model
(~90 MB) from Hugging Face — this needs an internet connection once;
after that it's cached locally.

### Getting a free LLM API key

- **Gemini (recommended for students, generous free tier):**
  https://aistudio.google.com/apikey
- **OpenAI:** https://platform.openai.com/api-keys (requires billing setup)

Set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` in `.env` to switch.

## Extending the project

- **Add destinations**: append more entries to
  `data/travel_knowledge_base.json` following the existing schema, then
  re-run `python -m src.build_index`.
- **Add real-time data**: swap the static JSON for live calls to flight/
  hotel APIs (e.g. Skyscanner, Amadeus, Google Places) inside
  `data_loader.py`.
- **Add PDF/Word export** of the generated itinerary.
- **Add a chat-style follow-up** ("make day 2 less packed") by keeping
  conversation history and re-running `rag_pipeline.py` with the new ask.

## Notes on model/version choice

Package versions in `requirements.txt` were the latest stable releases
verified at the time this project was generated. LLM model names
(`gemini-2.5-flash`, `gpt-4o-mini`) are also kept as environment
variables in `.env` so you can easily point to whichever current model
your API key has access to without touching the code.
