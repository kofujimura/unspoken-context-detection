"""
Data models for LLM-direct UCC detection (SPEC2.md).
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class LLMDetectionParams(BaseModel):
    """Parameters for LLM-direct detection."""
    chunk_size: int = Field(default=15, description="Number of segments per chunk (matches embedding k_minus + k_plus)")
    overlap: int = Field(default=3, description="Overlap between chunks")
    max_established: int = Field(default=200, description="Max rolling prerequisites to keep")
    model: str = Field(default="gpt-4o-mini", description="LLM model name")
    temperature: float = Field(default=0.0, description="LLM temperature")
    retry_max: int = Field(default=2, description="Max retries on invalid JSON")


class Evidence(BaseModel):
    """Evidence for a UCC mark."""
    not_established_before: List[str] = Field(default_factory=list, max_length=3)
    topic_continuity: str = Field(default="medium", pattern="^(high|medium|low)$")


class Mark(BaseModel):
    """A single UCC mark from LLM."""
    id: int = Field(description="Segment ID where UCC occurs")
    severity: int = Field(ge=1, le=5, description="Severity score 1-5")
    new_required_prerequisites: List[str] = Field(default_factory=list, max_length=5)
    evidence: Evidence = Field(default_factory=Evidence)
    micro_suggestion: str = Field(default="", max_length=120)
    mark_span_ids: List[int] = Field(default_factory=list)


class EstablishedUpdate(BaseModel):
    """Updates to established prerequisites."""
    add: List[str] = Field(default_factory=list, max_length=8)
    remove: List[str] = Field(default_factory=list)


class ChunkResult(BaseModel):
    """LLM response for a single chunk."""
    marks: List[Mark] = Field(default_factory=list)
    established_updates: EstablishedUpdate = Field(default_factory=EstablishedUpdate)


class SegmentWithScores(BaseModel):
    """Output segment with LLM-direct scores."""
    id: int
    start_ms: int
    end_ms: int
    text: str
    is_marked: bool = False
    reasons: List[str] = Field(default_factory=list)
    ucc_score: float = Field(default=0.0, ge=0.0, le=1.0)
    severity: Optional[int] = Field(default=None, ge=1, le=5)
    micro_suggestion: Optional[str] = None


class LLMAnalyzeResponse(BaseModel):
    """Response from LLM-direct analyze endpoint."""
    video_id: str
    language: str
    params: Dict[str, Any]
    segments: List[SegmentWithScores]
    total_chunks: int = 0
    total_marks: int = 0
