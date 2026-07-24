"""
data_loader.py
----------------
Loads the travel knowledge base (JSON) and converts every destination's
attractions, accommodations, food, transport, tips, and budget info into
small, self-contained text "documents". Each document is what actually
gets embedded and stored in FAISS, and metadata is kept alongside it so
the retrieved chunk can be traced back to a destination/category.

Splitting one big destination record into several smaller documents
(rather than embedding the whole JSON blob as one chunk) is what makes
semantic search precise: a query about "budget hotels in Goa" should
match the accommodations chunk, not the whole Goa record.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_destinations(data_path: Path) -> List[Dict[str, Any]]:
    """Load the raw destination records from the JSON knowledge base."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _doc(text: str, destination: str, country: str, category: str) -> Dict[str, Any]:
    """Helper to build a uniform document dict."""
    return {
        "text": text,
        "metadata": {
            "destination": destination,
            "country": country,
            "category": category,
        },
    }


def build_documents(destinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert destination records into a flat list of chunked documents
    ready for embedding. Each chunk focuses on one aspect of one
    destination (overview, one attraction, accommodation tier, etc.)
    """
    documents: List[Dict[str, Any]] = []

    for dest in destinations:
        name = dest["destination"]
        country = dest.get("country", "")

        # 1) Overview chunk
        overview_text = (
            f"{name} ({country}) overview: {dest.get('overview', '')} "
            f"Best time to visit: {dest.get('best_time_to_visit', '')}. "
            f"Currency: {dest.get('currency', '')}. Language: {dest.get('language', '')}."
        )
        documents.append(_doc(overview_text, name, country, "overview"))

        # 2) One chunk per attraction
        for a in dest.get("attractions", []):
            text = (
                f"{name} attraction - {a['name']} ({a.get('category', '')}): "
                f"{a.get('description', '')} Approx cost per person: "
                f"{a.get('estimated_cost_per_person', 'N/A')}. "
                f"Typical visit duration: {a.get('duration_hours', 'N/A')} hours."
            )
            documents.append(_doc(text, name, country, "attraction"))

        # 3) One chunk per accommodation tier
        for acc in dest.get("accommodations", []):
            text = (
                f"{name} accommodation ({acc.get('type', '')}): {acc['name']} in "
                f"{acc.get('area', '')}. Price range per night: "
                f"{acc.get('price_range_per_night', 'N/A')}. Rating: {acc.get('rating', 'N/A')}/5."
            )
            documents.append(_doc(text, name, country, "accommodation"))

        # 4) One chunk per food item
        for f in dest.get("food", []):
            text = (
                f"{name} food - {f['name']} ({f.get('type', '')}): {f.get('description', '')} "
                f"Average cost per person: {f.get('avg_cost_per_person', 'N/A')}."
            )
            documents.append(_doc(text, name, country, "food"))

        # 5) One chunk per transport option
        for t in dest.get("transport", []):
            text = (
                f"{name} transport - {t['mode']}: {t.get('description', '')} "
                f"Average cost per day: {t.get('avg_cost_per_day', 'N/A')}."
            )
            documents.append(_doc(text, name, country, "transport"))

        # 6) Travel tips, grouped into one chunk
        tips = dest.get("travel_tips", [])
        if tips:
            tips_text = f"{name} travel tips: " + " | ".join(tips)
            documents.append(_doc(tips_text, name, country, "tips"))

        # 7) Budget summary chunk
        budget = dest.get("average_daily_budget", {})
        if budget:
            budget_text = (
                f"{name} average daily budget in {budget.get('currency', '')}: "
                f"Budget traveler: {budget.get('budget', 'N/A')}, "
                f"Mid-range traveler: {budget.get('mid_range', 'N/A')}, "
                f"Luxury traveler: {budget.get('luxury', 'N/A')}."
            )
            documents.append(_doc(budget_text, name, country, "budget"))

    return documents


def load_all_documents(data_path: Path) -> List[Dict[str, Any]]:
    """Convenience wrapper: load JSON and return chunked documents in one call."""
    destinations = load_destinations(data_path)
    return build_documents(destinations)
