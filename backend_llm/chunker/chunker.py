"""
Chunker for rolling window processing (SPEC2.md B3).
"""
from typing import List, Dict, Any
from pydantic import BaseModel


class Chunk(BaseModel):
    """A chunk of segments for LLM processing."""
    chunk_id: int
    segments: List[Dict[str, Any]]  # List of {id, text}
    start_segment_id: int
    end_segment_id: int


class Chunker:
    """
    Splits transcript segments into overlapping chunks.

    SPEC2.md B3.2 (Updated to match embedding version):
    - CHUNK_SIZE = 15 segments (default, matches k_minus + k_plus)
    - OVERLAP = 3 segments (default)
    - Effective step = CHUNK_SIZE - OVERLAP
    """

    def __init__(self, chunk_size: int = 15, overlap: int = 3):
        """
        Initialize chunker.

        Args:
            chunk_size: Number of segments per chunk
            overlap: Number of overlapping segments between chunks
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def create_chunks(self, segments: List[Dict]) -> List[Chunk]:
        """
        Create overlapping chunks from segments.

        Args:
            segments: List of segment dicts with 'id' and 'text'

        Returns:
            List of Chunk objects
        """
        chunks = []
        total_segments = len(segments)

        start_idx = 0
        chunk_id = 0

        while start_idx < total_segments:
            end_idx = min(start_idx + self.chunk_size, total_segments)

            chunk_segments = []
            for i in range(start_idx, end_idx):
                chunk_segments.append({
                    "id": segments[i]["id"],
                    "text": segments[i]["text"]
                })

            chunk = Chunk(
                chunk_id=chunk_id,
                segments=chunk_segments,
                start_segment_id=segments[start_idx]["id"],
                end_segment_id=segments[end_idx - 1]["id"]
            )
            chunks.append(chunk)

            # Move to next chunk
            start_idx += self.step
            chunk_id += 1

            # Stop if we've reached the end
            if end_idx >= total_segments:
                break

        return chunks
