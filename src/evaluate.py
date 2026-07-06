"""
Evaluation script.

Runs all instructor questions through both the baseline and RAG systems,
saves results to results/evaluation_results.csv, and prints a summary.

Scores (correctness, grounding, retrieval_relevance, usefulness, style_quality)
are left blank for manual annotation after reviewing the outputs.

Run from the src/ directory:
    python evaluate.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIR
from baseline import baseline_answer
from rag_chatbot import build_rag_prompt, format_chunk_for_display, generate_answer, get_retriever
from chunking import format_chunk_for_display


QUESTIONS_PATH = RESULTS_DIR / "instructor_questions.json"
OUTPUT_PATH = RESULTS_DIR / "evaluation_results.csv"

FIELDNAMES = [
    "question_id",
    "question",
    "question_type",
    "expected_focus",
    "system",
    "retrieved_passages",
    "generated_response",
    "correctness_score",
    "grounding_score",
    "retrieval_relevance_score",
    "usefulness_score",
    "style_quality_score",
    "comments",
]


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both list and {"questions": [...]} formats
    return data if isinstance(data, list) else data.get("questions", data)


def format_retrieved(retrieved) -> str:
    parts = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        parts.append(f"[Rank {rank} | score={score:.4f}]\n{format_chunk_for_display(chunk)}")
    return "\n\n".join(parts)


def run_evaluation() -> None:
    questions = load_questions()
    retriever = get_retriever()

    rows = []
    for q in questions:
        qid = q.get("question_id") or q.get("id", "")
        question = q.get("question", "")
        qtype = q.get("type") or q.get("question_type", "")
        expected = q.get("expected_focus", "")

        print(f"\n{'='*60}")
        print(f"[{qid}] {question}")

        # --- Baseline ---
        print("  Running baseline...")
        b_answer = baseline_answer(question)
        rows.append({
            "question_id": qid,
            "question": question,
            "question_type": qtype,
            "expected_focus": expected,
            "system": "baseline",
            "retrieved_passages": "",
            "generated_response": b_answer,
            "correctness_score": "",
            "grounding_score": "",
            "retrieval_relevance_score": "",
            "usefulness_score": "",
            "style_quality_score": "",
            "comments": "",
        })

        # --- RAG ---
        print("  Running RAG...")
        from config import DEFAULT_TOP_K
        retrieved = retriever.retrieve(question, top_k=DEFAULT_TOP_K)
        prompt = build_rag_prompt(question, retrieved)
        r_answer = generate_answer(prompt)
        rows.append({
            "question_id": qid,
            "question": question,
            "question_type": qtype,
            "expected_focus": expected,
            "system": "rag",
            "retrieved_passages": format_retrieved(retrieved),
            "generated_response": r_answer,
            "correctness_score": "",
            "grounding_score": "",
            "retrieval_relevance_score": "",
            "usefulness_score": "",
            "style_quality_score": "",
            "comments": "",
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("Open the CSV and fill in the score columns manually after reviewing responses.")


if __name__ == "__main__":
    run_evaluation()
