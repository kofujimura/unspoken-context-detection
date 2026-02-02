"""
Mark merger for overlapping chunks (SPEC2.md B4.3).
"""
from typing import List, Dict
from collections import defaultdict

from backend_llm.core.models import Mark


class MarkMerger:
    """
    Merges marks from overlapping chunks.

    SPEC2.md B4.3:
    - For each segment ID, keep mark with max severity
    - Union new_required_prerequisites (deduplicate)
    - Keep shortest micro_suggestion (or prefer earlier chunk)
    - Union mark_span_ids
    """

    @staticmethod
    def merge_marks(all_marks: List[Mark]) -> List[Mark]:
        """
        Merge marks across chunks.

        Args:
            all_marks: List of all marks from all chunks

        Returns:
            List of merged marks (one per unique segment ID)
        """
        # Group marks by segment ID
        marks_by_id: Dict[int, List[Mark]] = defaultdict(list)
        for mark in all_marks:
            marks_by_id[mark.id].append(mark)

        # Merge each group
        merged_marks = []
        for segment_id, marks in marks_by_id.items():
            merged_mark = MarkMerger._merge_mark_group(segment_id, marks)
            merged_marks.append(merged_mark)

        # Sort by segment ID
        merged_marks.sort(key=lambda m: m.id)

        return merged_marks

    @staticmethod
    def _merge_mark_group(segment_id: int, marks: List[Mark]) -> Mark:
        """
        Merge multiple marks for the same segment ID.

        Args:
            segment_id: Segment ID
            marks: List of marks for this segment

        Returns:
            Merged mark
        """
        # Find max severity
        max_severity = max(m.severity for m in marks)

        # Union prerequisites (deduplicate, preserve order)
        prerequisites_set = set()
        prerequisites = []
        for mark in marks:
            for prereq in mark.new_required_prerequisites:
                prereq_normalized = MarkMerger._normalize_prerequisite(prereq)
                if prereq_normalized not in prerequisites_set:
                    prerequisites_set.add(prereq_normalized)
                    prerequisites.append(prereq)

        # Union mark_span_ids (deduplicate and sort)
        span_ids_set = set()
        for mark in marks:
            span_ids_set.update(mark.mark_span_ids)
        span_ids = sorted(list(span_ids_set))

        # Choose shortest micro_suggestion (or first if tie)
        suggestions = [m.micro_suggestion for m in marks if m.micro_suggestion]
        if suggestions:
            micro_suggestion = min(suggestions, key=len)
        else:
            micro_suggestion = ""

        # Use evidence from mark with max severity
        max_severity_mark = next(m for m in marks if m.severity == max_severity)

        return Mark(
            id=segment_id,
            severity=max_severity,
            new_required_prerequisites=prerequisites[:5],  # Limit to 5
            evidence=max_severity_mark.evidence,
            micro_suggestion=micro_suggestion,
            mark_span_ids=span_ids
        )

    @staticmethod
    def _normalize_prerequisite(prereq: str) -> str:
        """
        Normalize prerequisite for deduplication.

        Args:
            prereq: Prerequisite string

        Returns:
            Normalized string
        """
        # Strip whitespace and convert to lowercase for Latin
        normalized = prereq.strip()

        # For Japanese, keep as-is (exact match)
        # For Latin, lowercase for case-insensitive match
        if any(ord(c) > 127 for c in normalized):
            # Contains non-ASCII (likely Japanese)
            return normalized
        else:
            # ASCII only (likely English)
            return normalized.lower()

    @staticmethod
    def fill_mark_spans(marks: List[Mark], total_segments: int) -> List[Mark]:
        """
        Fill missing mark_span_ids with default [id, id+1, id+2].

        SPEC2.md B4.2: If mark_span_ids is missing or empty,
        set to [id, id+1, id+2] clipped to segment bounds.

        Args:
            marks: List of marks
            total_segments: Total number of segments

        Returns:
            List of marks with filled spans
        """
        filled_marks = []
        for mark in marks:
            if not mark.mark_span_ids:
                # Default span: [id, id+1, id+2]
                span = [mark.id, mark.id + 1, mark.id + 2]
                # Clip to bounds
                span = [s for s in span if 0 <= s < total_segments]
                mark.mark_span_ids = span

            filled_marks.append(mark)

        return filled_marks
