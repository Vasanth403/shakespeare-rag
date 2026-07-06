"""
Configuration for the Assignment 2 starter code.

Students should adjust these values to match their own implementation.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROMPT_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Raw dataset provided with the assignment (one level up from starter code root)
RAW_DATA_DIR = PROJECT_ROOT.parent / "data" / "raw" / "shakespeare_slm_dataset"

PLAY_FILES = {
    "hamlet": DATA_DIR / "hamlet.json",
    "macbeth": DATA_DIR / "macbeth.json",
    "romeo_and_juliet": DATA_DIR / "romeo_and_juliet.json",
}

DEFAULT_TOP_K = 3

# Embedding model for retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama local model for generation
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
