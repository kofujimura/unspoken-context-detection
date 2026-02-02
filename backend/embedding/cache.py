"""
Embedding cache for reducing API calls and costs.
Based on SPEC.md Section 11.
"""
import json
import hashlib
import os
from pathlib import Path
from typing import Optional, List
import numpy as np


class EmbeddingCache:
    """Disk-based cache for embeddings."""

    def __init__(self, cache_dir: str = ".cache/embeddings"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.jsonl"

        # Load existing cache into memory for fast lookup
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk into memory."""
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = self._make_key(
                            entry['video_id'],
                            entry['segment_id'],
                            entry['text']
                        )
                        cache[key] = entry['embedding']
        return cache

    def _make_key(self, video_id: str, segment_id: int, text: str) -> str:
        """
        Create cache key from video_id, segment_id, and text hash.

        Args:
            video_id: Video identifier
            segment_id: Segment ID
            text: Segment text

        Returns:
            Cache key string
        """
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"{video_id}:{segment_id}:{text_hash}"

    def get(self, video_id: str, segment_id: int, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.

        Args:
            video_id: Video identifier
            segment_id: Segment ID
            text: Segment text

        Returns:
            Embedding vector if cached, None otherwise
        """
        key = self._make_key(video_id, segment_id, text)
        return self._cache.get(key)

    def put(self, video_id: str, segment_id: int, text: str, embedding: List[float]):
        """
        Store embedding in cache.

        Args:
            video_id: Video identifier
            segment_id: Segment ID
            text: Segment text
            embedding: Embedding vector
        """
        key = self._make_key(video_id, segment_id, text)
        self._cache[key] = embedding

        # Append to disk cache
        entry = {
            'video_id': video_id,
            'segment_id': segment_id,
            'text': text,
            'embedding': embedding
        }

        with open(self.cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def get_batch(
        self,
        video_id: str,
        segments: List[tuple]  # [(segment_id, text), ...]
    ) -> tuple[List[Optional[List[float]]], List[int]]:
        """
        Get multiple embeddings from cache.

        Args:
            video_id: Video identifier
            segments: List of (segment_id, text) tuples

        Returns:
            Tuple of (embeddings list with None for misses, list of miss indices)
        """
        embeddings = []
        miss_indices = []

        for idx, (seg_id, text) in enumerate(segments):
            emb = self.get(video_id, seg_id, text)
            embeddings.append(emb)
            if emb is None:
                miss_indices.append(idx)

        return embeddings, miss_indices

    def put_batch(
        self,
        video_id: str,
        segments: List[tuple],  # [(segment_id, text), ...]
        embeddings: List[List[float]]
    ):
        """
        Store multiple embeddings in cache.

        Args:
            video_id: Video identifier
            segments: List of (segment_id, text) tuples
            embeddings: List of embedding vectors
        """
        for (seg_id, text), embedding in zip(segments, embeddings):
            self.put(video_id, seg_id, text, embedding)

    def clear(self):
        """Clear all cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        self._cache = {}
