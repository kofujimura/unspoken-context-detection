"""
Reason labeling for marked segments.
Based on SPEC.md Section 7.
"""
from typing import List
import numpy as np
from backend.core.models import Transcript, AnalysisResult, DetectionParams
from backend.detector.windows import get_window_indices
from .keyword_extractor import KeywordExtractor


class ReasonLabeler:
    """Add reason labels to marked segments."""

    def __init__(self, params: DetectionParams, language: str = "ja"):
        """
        Initialize labeler.

        Args:
            params: Detection parameters
            language: Language code
        """
        self.params = params
        self.extractor = KeywordExtractor(language=language)

    def add_reasons(
        self,
        result: AnalysisResult,
        transcript: Transcript
    ) -> AnalysisResult:
        """
        Add reason labels to marked segments.

        Args:
            result: Analysis result
            transcript: Original transcript

        Returns:
            Result with reasons filled in
        """
        N = len(transcript.segments)
        L = self.params.mark_length

        # First pass: identify which high-score segment caused each marking
        # For each marked segment i, find all segments j where i is in [j-L+1, j]
        marked_by = {}  # i -> list of j that caused marking of i

        for j, seg_j in enumerate(result.segments):
            # Check if j has high enough score to be a detection candidate
            # We'll consider any segment with relatively high delta
            if seg_j.scores.context_debt_delta > 0:
                # This segment j would mark range [j-L+1, j]
                start_mark = max(0, j - L + 1)
                for i in range(start_mark, j + 1):
                    if i < N and result.segments[i].is_marked:
                        if i not in marked_by:
                            marked_by[i] = []
                        marked_by[i].append(j)

        # Second pass: compute reasons for each marked segment
        for i, seg in enumerate(result.segments):
            if seg.is_marked:
                # Find the segment j with highest score that caused marking of i
                if i in marked_by and marked_by[i]:
                    # Use the j with maximum context_debt_delta
                    j = max(marked_by[i], key=lambda x: result.segments[x].scores.context_debt_delta)
                else:
                    # Fallback: assume i itself is the detection point
                    j = i

                # Compute position t from j
                t = j - self.params.k_plus + 1

                # Ensure t is in valid range
                if t < 0:
                    t = 0
                if t >= N:
                    t = N - 1

                # Get window texts for the computation position t
                minus_indices = get_window_indices(t, self.params.k_minus, N, mode="minus")
                plus_indices = get_window_indices(t, self.params.k_plus, N, mode="plus")

                text_minus = " ".join([
                    transcript.segments[m].text for m in minus_indices
                ])
                text_plus = " ".join([
                    transcript.segments[m].text for m in plus_indices
                ])

                # Compute keyword diff
                reasons = self.extractor.compute_keyword_diff(
                    text_minus,
                    text_plus,
                    top_n=self.params.n_reasons
                )

                seg.reasons = reasons

        return result
