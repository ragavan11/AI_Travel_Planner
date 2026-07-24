"""
rag_pipeline.py
------------------
This is the heart of the project: a Retrieval-Augmented Generation (RAG)
pipeline that
  1) turns the user's trip preferences into a semantic search query,
  2) retrieves the most relevant chunks from the FAISS vector store,
  3) stuffs those chunks into a prompt as grounding "context", and
  4) asks the LLM to write a structured, day-wise itinerary using only
     (mostly) that retrieved context -- this is what makes the output
     grounded in the knowledge base instead of being a pure hallucination.
"""

from typing import List, Dict, Any

from .embeddings import EmbeddingModel
from .vector_store import FaissVectorStore
from .llm_client import LLMClient
from . import config


SYSTEM_PROMPT = """You are an expert AI travel planner. You generate clear,
practical, day-wise travel itineraries grounded in the CONTEXT provided to
you. Follow these rules:

1. Use the CONTEXT as your primary source of attractions, accommodation,
   food, and transport details. If the CONTEXT does not fully cover the
   destination, you may use general travel knowledge, but prefer CONTEXT
   facts and figures whenever they are available.
2. Respect the traveler's budget level, trip duration, and interests.
3. Structure the answer with the following sections, using Markdown:
   - A short trip overview (2-3 sentences)
   - Day-wise itinerary (Day 1, Day 2, ...), each with morning / afternoon
     / evening plans and approximate costs
   - Estimated total budget breakdown (accommodation, food, transport,
     activities) compared against the traveler's stated budget
   - Accommodation suggestions (budget/mid-range/luxury, matching the
     traveler's preference)
   - 4-6 practical travel tips
4. Be specific with costs and currency where the CONTEXT provides them.
5. Keep the tone friendly and concise -- no filler text."""


class TravelPlannerRAG:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: FaissVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = LLMClient()

    def _build_search_query(
        self, destination: str, interests: List[str], duration_days: int, budget_level: str
    ) -> str:
        interest_str = ", ".join(interests) if interests else "general sightseeing"
        return (
            f"{destination} trip for {duration_days} days, {budget_level} budget, "
            f"interested in {interest_str}: attractions, accommodation, food, "
            f"transport, travel tips, daily budget"
        )

    def retrieve_context(
        self, destination: str, interests: List[str], duration_days: int, budget_level: str
    ) -> List[Dict[str, Any]]:
        query = self._build_search_query(destination, interests, duration_days, budget_level)
        query_embedding = self.embedding_model.encode(query)[0]
        results = self.vector_store.search(query_embedding, top_k=config.TOP_K_RETRIEVAL)

        # Prefer chunks whose destination metadata matches the requested
        # destination (simple metadata filter layered on top of semantic
        # search -- keeps the itinerary from mixing in an unrelated city).
        dest_lower = destination.lower()
        matched = [r for r in results if dest_lower in r["document"]["metadata"]["destination"].lower()]
        return matched if matched else results

    def _format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        lines = []
        for r in retrieved:
            meta = r["document"]["metadata"]
            lines.append(f"[{meta['category']} | {meta['destination']}] {r['document']['text']}")
        return "\n".join(lines) if lines else "No matching context found in the knowledge base."

    def generate_itinerary(
        self,
        destination: str,
        duration_days: int,
        budget_level: str,
        interests: List[str],
        num_travelers: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline and return both the generated itinerary
        and the retrieved context (useful for showing "sources" in the UI).
        """
        retrieved = self.retrieve_context(destination, interests, duration_days, budget_level)
        context_text = self._format_context(retrieved)

        user_prompt = f"""Plan a trip with these details:
- Destination: {destination}
- Duration: {duration_days} day(s)
- Number of travelers: {num_travelers}
- Budget level: {budget_level}
- Interests: {', '.join(interests) if interests else 'general sightseeing'}

CONTEXT (retrieved from the travel knowledge base):
{context_text}

Now generate the itinerary following the structure and rules you were given."""

        itinerary_text = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return {
            "itinerary": itinerary_text,
            "retrieved_context": retrieved,
        }
