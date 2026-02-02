"""
OpenAI embedding client with caching and batch processing.
Based on SPEC.md Section 5.
"""
import os
from typing import List, Optional
import asyncio
from openai import AsyncOpenAI
import numpy as np
from backend.core.models import Transcript
from .cache import EmbeddingCache


class OpenAIEmbedder:
    """OpenAI embedding client with caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        cache_dir: str = ".cache/embeddings",
        max_retries: int = 3,
        batch_size: int = 100
    ):
        """
        Initialize embedder.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            model: Embedding model name
            cache_dir: Cache directory path
            max_retries: Maximum retry attempts for API calls
            batch_size: Batch size for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or parameters")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.cache = EmbeddingCache(cache_dir)

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts with batching and retries.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch with exponential backoff retry.

        Args:
            texts: Batch of texts

        Returns:
            List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Embedding API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError("Failed to get embeddings after all retries")

    async def embed_transcript(
        self,
        transcript: Transcript,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Embed all segments in a transcript with caching.

        Args:
            transcript: Input transcript
            use_cache: If True, use cache for embeddings

        Returns:
            Array of embeddings with shape (N, d) where N is number of segments
        """
        segments = [(seg.id, seg.text) for seg in transcript.segments]
        embeddings_list = []

        if use_cache:
            # Check cache first
            cached_embeddings, miss_indices = self.cache.get_batch(
                transcript.video_id,
                segments
            )

            if miss_indices:
                # Fetch missing embeddings
                miss_texts = [segments[idx][1] for idx in miss_indices]
                print(f"Cache miss: {len(miss_indices)}/{len(segments)} segments")
                new_embeddings = await self.embed_batch(miss_texts)

                # Update cache
                miss_segments = [segments[idx] for idx in miss_indices]
                self.cache.put_batch(transcript.video_id, miss_segments, new_embeddings)

                # Fill in missing embeddings
                new_emb_iter = iter(new_embeddings)
                for idx, emb in enumerate(cached_embeddings):
                    if emb is None:
                        embeddings_list.append(next(new_emb_iter))
                    else:
                        embeddings_list.append(emb)
            else:
                print(f"Cache hit: {len(segments)}/{len(segments)} segments")
                embeddings_list = cached_embeddings

        else:
            # No cache, fetch all
            texts = [seg.text for seg in transcript.segments]
            embeddings_list = await self.embed_batch(texts)

        return np.array(embeddings_list)
