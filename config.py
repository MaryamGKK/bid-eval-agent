# config.py
"""Configuration settings for Bid Evaluation Agent POC."""

import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key: str, default: str = None) -> str:
    """Get secret from environment or Streamlit secrets."""
    # First try environment variable
    value = os.getenv(key)
    if value:
        return value
    
    # Then try Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    return default

# API Keys (works with both .env and Streamlit secrets)
GROQ_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")
SERPAPI_API_KEY = get_secret("SERPAPI_API_KEY")

# Model Settings
GROQ_MODEL = "openai/gpt-oss-120b"

# Alternative models (uncomment to use):
# GROQ_MODEL = "llama-3.3-70b-versatile"  # 280 T/s, good balance
# GROQ_MODEL = "openai/gpt-oss-20b"       # 1000 T/s, faster but smaller

# Scoring Weights (risk-prioritized per project requirements)
WEIGHTS = {
    "cost": 0.20,
    "timeline": 0.20,
    "scope": 0.25,
    "risk": 0.25,
    "reputation": 0.10
}

# Project Requirements (LA Commercial Renovation)
MANDATORY_SCOPE = ["Electrical", "HVAC", "Interior"]
TARGET_MONTHS = 6
TARGET_BUDGET = 1_200_000
BUDGET_HARD_CAP = 1_350_000
MAX_TIMELINE_EXTENSION_WEEKS = 2

# Thresholds
CONFIDENCE_THRESHOLD = 0.7
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.7
DOCUMENTS_PER_COMPANY = 5

# RAG Settings
RAG_CONFIG = {
    "vector_store": "ChromaDB",
    "embedding_model": "all-MiniLM-L6-v2",
    "documents_retrieved_per_company": DOCUMENTS_PER_COMPANY,
    "retrieval_confidence_threshold": RETRIEVAL_CONFIDENCE_THRESHOLD
}
