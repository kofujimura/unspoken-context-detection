"""
LLM caller with caching and retry logic (SPEC2.md B6).
"""
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
from pydantic import ValidationError

from backend_llm.core.models import ChunkResult
from .prompt_builder import PromptBuilder


class LLMCaller:
    """
    Calls LLM API with caching and retry logic.

    SPEC2.md B6:
    - Cache by (video_id, chunk_id, segments_hash, established_hash, model, prompt_version)
    - Temperature 0.0 for determinism
    - Retry up to RETRY_MAX on invalid JSON
    """

    PROMPT_VERSION = "v1.2"  # Fixed mark.id to use actual segment IDs instead of local indices

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retry_max: int = 2,
        cache_dir: str = ".cache/llm_chunks"
    ):
        """
        Initialize LLM caller.

        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            retry_max: Max retry attempts
            cache_dir: Cache directory path
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.retry_max = retry_max
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(
        self,
        video_id: str,
        chunk_id: int,
        segments: list,
        established_prerequisites: list
    ) -> str:
        """
        Compute cache key for a chunk request.

        Args:
            video_id: Video identifier
            chunk_id: Chunk index
            segments: Segment list
            established_prerequisites: Rolling prerequisites

        Returns:
            Cache key string
        """
        segments_str = json.dumps(segments, ensure_ascii=False, sort_keys=True)
        established_str = json.dumps(established_prerequisites, ensure_ascii=False, sort_keys=True)

        key_data = f"{video_id}|{chunk_id}|{segments_str}|{established_str}|{self.model}|{self.PROMPT_VERSION}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()

        return key_hash

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.json"

    async def _load_from_cache(self, cache_key: str) -> Optional[ChunkResult]:
        """
        Load cached result if exists.

        Args:
            cache_key: Cache key

        Returns:
            ChunkResult if cached, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ChunkResult(**data)
        except Exception:
            return None

    async def _save_to_cache(self, cache_key: str, result: ChunkResult):
        """
        Save result to cache.

        Args:
            cache_key: Cache key
            result: ChunkResult to save
        """
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Cache write failure is non-fatal

    async def call_llm(
        self,
        video_id: str,
        chunk_id: int,
        segments: list,
        established_prerequisites: list
    ) -> ChunkResult:
        """
        Call LLM for a chunk with caching and retry.

        Args:
            video_id: Video identifier
            chunk_id: Chunk index
            segments: List of {id, text} dicts
            established_prerequisites: Rolling prerequisites list

        Returns:
            ChunkResult

        Raises:
            Exception: If all retries fail
        """
        # Check cache first
        cache_key = self._compute_cache_key(
            video_id, chunk_id, segments, established_prerequisites
        )
        cached_result = await self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Build prompt
        user_prompt = PromptBuilder.build_user_prompt(segments, established_prerequisites)

        # Retry loop
        for attempt in range(self.retry_max + 1):
            try:
                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                # Parse response
                content = response.choices[0].message.content
                data = json.loads(content)

                # Validate schema
                result = ChunkResult(**data)

                # Cache and return
                await self._save_to_cache(cache_key, result)
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                # Invalid JSON or schema - retry with warning
                if attempt < self.retry_max:
                    user_prompt = PromptBuilder.build_retry_prompt(user_prompt)
                    await asyncio.sleep(1)  # Brief delay before retry
                else:
                    raise Exception(f"LLM returned invalid JSON after {self.retry_max} retries: {e}")

            except Exception as e:
                # Other errors (API errors, etc)
                if attempt < self.retry_max:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"LLM call failed after {self.retry_max} retries: {e}")

        # Should not reach here
        raise Exception("Unexpected error in LLM call")
