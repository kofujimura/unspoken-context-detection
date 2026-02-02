"""
Main UCC detection algorithm.
Based on SPEC.md Section 6.
"""
import numpy as np
from typing import List, Optional, Set
from backend.core.models import (
    Transcript,
    AnalyzedSegment,
    SegmentScores,
    DetectionParams,
    AnalysisResult
)
from .metrics import (
    cosine_similarity,
    context_debt,
    normalize_scores_minmax,
    normalize_scores_zsigmoid
)
from .windows import compute_all_context_embeddings


class UCCDetector:
    """UCC detection with sliding window analysis."""

    def __init__(self, params: DetectionParams):
        """
        Initialize detector.

        Args:
            params: Detection parameters
        """
        self.params = params

    def detect(
        self,
        transcript: Transcript,
        embeddings: np.ndarray
    ) -> AnalysisResult:
        """
        Perform UCC detection on transcript.

        Args:
            transcript: Input transcript
            embeddings: Segment embeddings (N, d)

        Returns:
            Analysis result with scored and marked segments
        """
        N = len(transcript.segments)

        min_required = self.params.k_minus + self.params.k_plus + 1
        if N < min_required:
            # Transcript too short, emit warning
            warnings = [f"Transcript too short (N={N}, expected >= {min_required})"]
        else:
            warnings = []

        # Compute context embeddings for all positions
        c_minus_list, c_plus_list = compute_all_context_embeddings(
            embeddings,
            self.params.k_minus,
            self.params.k_plus
        )

        # Initialize score arrays (will be filled by assigning scores to target indices)
        topic_similarities = [None] * N
        context_debts = [None] * N
        context_debt_deltas = [None] * N

        # Temporary storage for computing deltas
        temp_debts = []

        for t in range(N):
            c_minus = c_minus_list[t]
            c_plus = c_plus_list[t]

            # Topic continuity
            s_topic = cosine_similarity(c_minus, c_plus)

            # Context debt
            d_t = context_debt(c_minus, c_plus)
            temp_debts.append(d_t)

            # Compute delta
            if t == 0:
                delta = 0.0
            else:
                delta = temp_debts[t] - temp_debts[t-1]

            # Assign scores to target index (last segment of W_plus)
            target_idx = t + self.params.k_plus - 1
            if 0 <= target_idx < N:
                topic_similarities[target_idx] = s_topic
                context_debts[target_idx] = d_t
                context_debt_deltas[target_idx] = delta

        # Fill any remaining None values with 0.0 (for segments before k-1)
        for i in range(N):
            if topic_similarities[i] is None:
                topic_similarities[i] = 0.0
            if context_debts[i] is None:
                context_debts[i] = 0.0
            if context_debt_deltas[i] is None:
                context_debt_deltas[i] = 0.0

        # Compute statistics for threshold
        deltas_array = np.array(context_debt_deltas)
        mu = np.mean(deltas_array)
        sigma = np.std(deltas_array)

        # Normalize UCC scores
        if self.params.score_norm == "minmax":
            ucc_scores = normalize_scores_minmax(deltas_array)
        elif self.params.score_norm == "zsigmoid":
            ucc_scores = normalize_scores_zsigmoid(deltas_array, mu, sigma)
        else:
            raise ValueError(f"Unknown score_norm: {self.params.score_norm}")

        # Detect candidates
        # Note: Scores at segment index i were computed from position t = i - k + 1
        # When we detect high score at segment i, we mark based on that segment index
        threshold = mu + self.params.lambda_param * sigma
        candidates = set()

        for i in range(N):
            # Jump condition
            if context_debt_deltas[i] > threshold:
                # Optional topic continuity mask
                if self.params.tau_topic is not None:
                    if topic_similarities[i] >= self.params.tau_topic:
                        candidates.add(i)
                else:
                    candidates.add(i)

        # Apply marking span
        marked_segments = self._apply_marking(candidates, N)

        # Create analyzed segments
        analyzed = []
        for t, seg in enumerate(transcript.segments):
            scores = SegmentScores(
                topic_similarity=topic_similarities[t],
                context_debt=context_debts[t],
                context_debt_delta=context_debt_deltas[t],
                ucc_score=float(ucc_scores[t])
            )

            is_marked = t in marked_segments

            analyzed_seg = AnalyzedSegment(
                id=seg.id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=seg.text,
                scores=scores,
                is_marked=is_marked,
                reasons=[]  # Will be filled by reason module
            )
            analyzed.append(analyzed_seg)

        # Create result
        result = AnalysisResult(
            video_id=transcript.video_id,
            language=transcript.language,
            params=self.params,
            segments=analyzed,
            metadata={
                "total_segments": N,
                "marked_count": len(marked_segments),
                "mu": float(mu),
                "sigma": float(sigma),
                "threshold": float(threshold),
                "warnings": warnings
            }
        )

        return result

    def _apply_marking(self, candidates: Set[int], total: int) -> Set[int]:
        """
        Apply marking to candidates.

        When segment i has high score, it was assigned from computation at position t = i - k_plus + 1.
        The score reflects the last L segments of W_plus(t), which end at segment i.
        We mark the last L segments ending at i: [i-L+1, i]

        Args:
            candidates: Set of segment indices with high scores
            total: Total number of segments

        Returns:
            Set of marked segment indices
        """
        marked = set()
        L = self.params.mark_length

        for i in candidates:
            # Segment i is the last of W_plus(t) where t = i - k_plus + 1
            # Mark the last L segments: [i-L+1, i]
            start_idx = i - L + 1
            end_idx = i

            for j in range(start_idx, end_idx + 1):
                if 0 <= j < total:
                    marked.add(j)

        return marked
