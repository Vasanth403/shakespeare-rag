"""
RAG chatbot using Ollama for generation.

Run from the src/ directory:
    python rag_chatbot.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import requests

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME, INDEX_DIR, OLLAMA_MODEL, OLLAMA_URL, PROMPT_DIR
from data_loader import load_all_plays
from chunking import create_chunks, format_chunk_for_display
from retrieval import EmbeddingRetriever


Chunk = Dict[str, Any]


def load_system_prompt() -> str:
    prompt_path = PROMPT_DIR / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_rag_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    system_prompt = load_system_prompt()

    context_blocks = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Context {rank} | similarity={score:.4f}]\n"
            f"{format_chunk_for_display(chunk)}"
        )

    context = "\n\n".join(context_blocks)

    return f"""{system_prompt}

Retrieved context:
{context}

User question:
{query}

Answer:
"""


def generate_answer(prompt: str) -> str:
    """
    Generate an answer using the local Ollama model.
    Requires Ollama to be running: https://ollama.com
    Pull a model first with: ollama pull llama3.2
    """
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


def get_retriever() -> EmbeddingRetriever:
    """Build or load cached embedding index."""
    retriever = EmbeddingRetriever(EMBEDDING_MODEL_NAME)
    if not retriever.load_index(INDEX_DIR):
        records = load_all_plays()
        chunks = create_chunks(records)
        retriever.build_index(chunks, cache_dir=INDEX_DIR)
    return retriever


def main() -> None:
    print("Loading retrieval index (this may take a moment on first run)...")
    retriever = get_retriever()

    print("\nShakespeare-aware RAG chatbot.")
    print(f"Model: {OLLAMA_MODEL} via Ollama")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        retrieved = retriever.retrieve(query, top_k=DEFAULT_TOP_K)
        prompt = build_rag_prompt(query, retrieved)
        answer = generate_answer(prompt)

        print("\nGenerated answer:")
        print(answer)

        print("\nRetrieved evidence (sources used to generate the above answer):")
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            print("-" * 80)
            print(f"Rank {rank} | Score: {score:.4f}")
            print(format_chunk_for_display(chunk))
        print("\n")


if __name__ == "__main__":
    main()
