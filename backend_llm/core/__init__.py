"""
Core models for LLM-direct UCC detection.
"""
from .models import (
    LLMDetectionParams,
    Mark,
    EstablishedUpdate,
    ChunkResult,
    LLMAnalyzeResponse
)

__all__ = [
    "LLMDetectionParams",
    "Mark",
    "EstablishedUpdate",
    "ChunkResult",
    "LLMAnalyzeResponse"
]
