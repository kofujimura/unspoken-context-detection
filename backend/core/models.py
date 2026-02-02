"""
Data models for UCC detection system.
Based on SPEC.md Section 2.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class Segment(BaseModel):
    """Normalized subtitle segment."""
    id: int = Field(..., ge=0)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., gt=0)
    text: str = Field(..., min_length=1)

    @field_validator('end_ms')
    @classmethod
    def end_must_be_after_start(cls, v: int, info) -> int:
        if 'start_ms' in info.data and v <= info.data['start_ms']:
            raise ValueError('end_ms must be greater than start_ms')
        return v


class Transcript(BaseModel):
    """Normalized transcript object."""
    video_id: str
    language: str = "ja"
    segments: List[Segment] = Field(default_factory=list)

    @field_validator('segments')
    @classmethod
    def segments_must_have_contiguous_ids(cls, v: List[Segment]) -> List[Segment]:
        """Ensure segment IDs are contiguous 0..N-1."""
        if not v:
            return v
        expected_ids = list(range(len(v)))
        actual_ids = [seg.id for seg in v]
        if actual_ids != expected_ids:
            # Auto-fix: reassign IDs
            for i, seg in enumerate(v):
                seg.id = i
        return v


class SegmentScores(BaseModel):
    """Scores for a single segment."""
    topic_similarity: Optional[float] = None
    context_debt: Optional[float] = None
    context_debt_delta: Optional[float] = None
    ucc_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class AnalyzedSegment(Segment):
    """Segment with UCC analysis results."""
    scores: SegmentScores = Field(default_factory=SegmentScores)
    is_marked: bool = False
    reasons: List[str] = Field(default_factory=list)


class DetectionParams(BaseModel):
    """Parameters for UCC detection algorithm."""
    # Window size parameters
    k_minus: int = Field(12, ge=1, description="Past window size (how far back to look)")
    k_plus: int = Field(3, ge=1, description="Current window size (forward-looking)")
    k: Optional[int] = Field(None, ge=1, description="Legacy: sets both k_minus and k_plus (for backward compatibility)")

    lambda_param: float = Field(1.5, ge=0.0, description="Threshold multiplier")
    tau_topic: Optional[float] = Field(0.55, ge=0.0, le=1.0, description="Topic continuity threshold")
    mark_length: int = Field(3, ge=1, description="Number of segments to mark at end of W_plus (last L segments)")
    score_norm: str = Field("zsigmoid", description="Score normalization method")
    reason_mode: str = Field("keyword_diff", description="Reason extraction mode")
    n_reasons: int = Field(5, ge=1, description="Number of reason keywords to extract")

    # Preprocessing params
    min_chars: int = Field(18, ge=1)
    max_chars: int = Field(120, ge=1)
    max_gap_ms: int = Field(800, ge=0)

    @model_validator(mode='after')
    def handle_legacy_k(self):
        """Handle backward compatibility: if k is set, use it for both k_minus and k_plus."""
        if self.k is not None:
            self.k_minus = self.k
            self.k_plus = self.k
        return self


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    video_id: str
    language: str
    params: DetectionParams
    segments: List[AnalyzedSegment]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def marked_segments(self) -> List[AnalyzedSegment]:
        """Get only marked segments."""
        return [seg for seg in self.segments if seg.is_marked]

    @property
    def marked_count(self) -> int:
        """Count of marked segments."""
        return len(self.marked_segments)


class AnalyzeRequest(BaseModel):
    """API request for analysis."""
    video_id: str
    language: str = "ja"
    params: Optional[DetectionParams] = None


class AnalyzeResponse(BaseModel):
    """API response for analysis."""
    success: bool
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
