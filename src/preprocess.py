"""
Preprocessing pipeline.

Reads scene-level JSONL chunks from the raw dataset and saves flat JSON lists
to data/processed/ for use by the RAG and baseline systems.

Chunking strategy: one chunk per scene.
Each scene already contains the full dialogue concatenated as `text`, plus
metadata (scene_summary, keywords, location) that enriches retrieval context.

Run from the src/ directory:
    python preprocess.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from config import DATA_DIR, RAW_DATA_DIR

Record = Dict[str, Any]

PLAY_SLUGS = {
    "hamlet": "hamlet",
    "macbeth": "macbeth",
    "romeo_and_juliet": "romeo_and_juliet",
}


def load_scene_chunks(jsonl_path: Path) -> List[Record]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_play(slug: str) -> List[Record]:
    """
    Load scene chunks for one play and normalise field names.
    Each output record has: play, act, scene, scene_id, location,
    scene_summary, keywords, text, source_id.
    """
    jsonl_path = RAW_DATA_DIR / f"{slug}_scene_chunks.jsonl"
    raw_records = load_scene_chunks(jsonl_path)

    processed = []
    for r in raw_records:
        text = r.get("text", "").strip()
        if not text:
            continue

        record: Record = {
            "play": r.get("play", slug.replace("_", " ").title()),
            "act": r.get("act"),
            "scene": r.get("scene"),
            "scene_id": r.get("scene_id", ""),
            "location": r.get("location", ""),
            "scene_summary": r.get("scene_summary", ""),
            "keywords": r.get("keywords", []),
            "text": text,
            "source_id": r.get("scene_id", ""),
        }
        processed.append(record)

    return processed


def count_utterances(slug: str) -> int:
    jsonl_path = RAW_DATA_DIR / f"{slug}_utterances.jsonl"
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_utterances = 0
    stats = []
    for slug in PLAY_SLUGS:
        records = process_play(slug)
        out_path = DATA_DIR / f"{slug}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        utterances = count_utterances(slug)
        stats.append((slug.replace("_", " ").title(), len(records), utterances))
        total_chunks += len(records)
        total_utterances += utterances

    print(f"\nPreprocessing complete. {total_chunks} total chunks saved to {DATA_DIR}")
    print(f"\n{'Play':<25} {'Scenes':>8} {'Utterances':>12}")
    print("-" * 47)
    for name, scenes, utterances in stats:
        print(f"{name:<25} {scenes:>8} {utterances:>12}")
    print("-" * 47)
    print(f"{'Total':<25} {total_chunks:>8} {total_utterances:>12}")


if __name__ == "__main__":
    run()
