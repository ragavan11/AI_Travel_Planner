"""
config.py
----------
Centralized configuration for the AI Travel Planner.
All values can be overridden via environment variables (see .env.example).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "travel_knowledge_base.json"
INDEX_DIR = BASE_DIR / "index"

# --- Embedding model (runs locally via sentence-transformers, no API key needed) ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- LLM provider: "gemini" or "openai" ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Gemini settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- RAG settings ---
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))
