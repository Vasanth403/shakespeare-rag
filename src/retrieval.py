"""
Embedding and retrieval utilities.

Uses sentence-transformers for embeddings and cosine similarity for retrieval.
Embeddings are cached to disk so subsequent runs skip the encoding step.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


Chunk = Dict[str, Any]


class EmbeddingRetriever:
    """
    Embedding-based retriever with disk caching of the index.
    """

    def __init__(self, embedding_model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: List[Chunk], cache_dir: Path | None = None) -> None:
        """
        Create embeddings for all chunks, optionally saving them to cache_dir.
        """
        if not chunks:
            raise ValueError("No chunks supplied to build_index().")

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        print(f"Encoding {len(texts)} chunks with {self.model_name} ...")
        self.embeddings = np.asarray(self.model.encode(texts, show_progress_bar=True))

        if cache_dir is not None:
            self.save_index(cache_dir)

    def save_index(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "embeddings.npy", self.embeddings)
        with (cache_dir / "chunks.pkl").open("wb") as f:
            pickle.dump(self.chunks, f)
        print(f"Index cached to {cache_dir}")

    def load_index(self, cache_dir: Path) -> bool:
        """
        Load a previously saved index. Returns True on success, False if not found.
        """
        emb_path = cache_dir / "embeddings.npy"
        chunks_path = cache_dir / "chunks.pkl"
        if not emb_path.exists() or not chunks_path.exists():
            return False

        self.embeddings = np.load(emb_path)
        with chunks_path.open("rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded cached index from {cache_dir} ({len(self.chunks)} chunks)")
        return True

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query using cosine similarity.
        """
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        query_embedding = np.asarray(self.model.encode([query]))
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
