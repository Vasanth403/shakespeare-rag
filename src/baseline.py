"""
Baseline system: prompt-only generation with no retrieval.

The baseline sends the user's question directly to Ollama with a system
prompt that relies entirely on the model's general Shakespeare knowledge.
Comparing this against the RAG system shows the value of grounded retrieval.

Run from the src/ directory:
    python baseline.py
"""

from __future__ import annotations

import requests

from config import OLLAMA_MODEL, OLLAMA_URL


BASELINE_SYSTEM = (
    "You are a Shakespeare expert. Answer the following question about Shakespeare's plays "
    "using only your general knowledge. Keep your answer beginner-friendly and concise."
)


def generate_answer(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Cannot connect to Ollama. "
            "Make sure Ollama is running (ollama serve) and the model is pulled "
            f"(ollama pull {OLLAMA_MODEL})."
        )
    except Exception as exc:
        return f"[ERROR] Generation failed: {exc}"


def baseline_answer(query: str) -> str:
    """
    Prompt-only baseline: no retrieval, no grounding.
    """
    prompt = f"{BASELINE_SYSTEM}\n\nQuestion: {query}\n\nAnswer:"
    return generate_answer(prompt)


if __name__ == "__main__":
    question = "Why does Macbeth kill Duncan?"
    print("Question:", question)
    print("\nBaseline answer (no retrieval):")
    print(baseline_answer(question))
