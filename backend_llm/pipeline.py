"""
LLM-direct UCC detection pipeline (SPEC2.md).
"""
from typing import List, Dict
from collections import defaultdict

# Reuse existing parser and preprocess from backend
from backend.parser import parse_vtt_from_string, parse_srt_from_string
from backend.preprocess import merge_short_segments, merge_incomplete_sentences, normalize_text

from .core.models import (
    LLMDetectionParams,
    SegmentWithScores,
    LLMAnalyzeResponse,
    Mark
)
from .chunker import Chunker
from .llm import LLMCaller
from .merger import MarkMerger


class LLMDirectPipeline:
    """
    Full pipeline for LLM-direct UCC detection.

    Steps:
    1. Parse subtitle file (reuse existing parser)
    2. Preprocess segments (reuse existing preprocess)
    3. Create overlapping chunks
    4. Process each chunk with LLM (with rolling prerequisites)
    5. Merge marks across chunks
    6. Build final output
    """

    def __init__(self, llm_caller: LLMCaller, params: LLMDetectionParams):
        """
        Initialize pipeline.

        Args:
            llm_caller: LLM caller instance
            params: Detection parameters
        """
        self.llm_caller = llm_caller
        self.params = params
        self.chunker = Chunker(
            chunk_size=params.chunk_size,
            overlap=params.overlap
        )

    async def analyze_content(
        self,
        content: str,
        file_format: str,
        video_id: str,
        language: str
    ) -> LLMAnalyzeResponse:
        """
        Analyze subtitle content.

        Args:
            content: Raw subtitle content
            file_format: 'vtt' or 'srt'
            video_id: Video identifier
            language: Language code

        Returns:
            LLMAnalyzeResponse with results
        """
        # Step 1: Parse
        if file_format == 'vtt':
            transcript = parse_vtt_from_string(content, video_id, language)
        elif file_format == 'srt':
            transcript = parse_srt_from_string(content, video_id, language)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        # Step 2: Preprocess
        transcript = merge_short_segments(transcript)
        # Merge incomplete sentences to improve consistency
        transcript = merge_incomplete_sentences(
            transcript,
            max_gap_ms=800,
            language=language
        )
        transcript = normalize_text(transcript)

        # Step 3: Create chunks
        segment_dicts = [
            {"id": seg.id, "text": seg.text}
            for seg in transcript.segments
        ]
        chunks = self.chunker.create_chunks(segment_dicts)

        # Step 4: Process chunks with rolling prerequisites
        all_marks: List[Mark] = []
        established_prerequisites: List[str] = []

        for chunk in chunks:
            # Call LLM
            chunk_result = await self.llm_caller.call_llm(
                video_id=video_id,
                chunk_id=chunk.chunk_id,
                segments=chunk.segments,
                established_prerequisites=established_prerequisites
            )

            # Collect marks
            all_marks.extend(chunk_result.marks)

            # Update rolling prerequisites
            new_prereqs = chunk_result.established_updates.add
            for prereq in new_prereqs:
                # Deduplicate
                prereq_normalized = self._normalize_prerequisite(prereq)
                if prereq_normalized not in {self._normalize_prerequisite(p) for p in established_prerequisites}:
                    established_prerequisites.append(prereq)

            # Limit size
            if len(established_prerequisites) > self.params.max_established:
                # Keep most recent
                established_prerequisites = established_prerequisites[-self.params.max_established:]

        # Step 5: Fill default mark spans and merge
        all_marks = MarkMerger.fill_mark_spans(all_marks, len(transcript.segments))
        merged_marks = MarkMerger.merge_marks(all_marks)

        # Step 6: Build output segments
        output_segments = self._build_output_segments(
            transcript.segments,
            merged_marks
        )

        return LLMAnalyzeResponse(
            video_id=video_id,
            language=language,
            params={
                "detection_mode": "llm_direct",
                "chunk_size": self.params.chunk_size,
                "overlap": self.params.overlap,
                "max_established": self.params.max_established,
                "model": self.params.model,
                "temperature": self.params.temperature
            },
            segments=output_segments,
            total_chunks=len(chunks),
            total_marks=len(merged_marks)
        )

    def _build_output_segments(
        self,
        original_segments: list,
        marks: List[Mark]
    ) -> List[SegmentWithScores]:
        """
        Build output segments with marking.

        Args:
            original_segments: Original transcript segments
            marks: Merged marks

        Returns:
            List of SegmentWithScores
        """
        # Build lookup: segment_id -> mark
        mark_by_id: Dict[int, Mark] = {m.id: m for m in marks}

        # Build lookup: segment_id -> is in any mark_span
        segment_in_span: Dict[int, List[Mark]] = defaultdict(list)
        for mark in marks:
            for span_id in mark.mark_span_ids:
                segment_in_span[span_id].append(mark)

        # Build output segments
        output_segments = []
        for seg in original_segments:
            # Check if segment is marked
            marks_for_seg = segment_in_span.get(seg.id, [])
            is_marked = len(marks_for_seg) > 0

            if is_marked:
                # Aggregate reasons (union)
                reasons_set = set()
                for mark in marks_for_seg:
                    reasons_set.update(mark.new_required_prerequisites)
                reasons = list(reasons_set)

                # Max severity
                max_severity = max(m.severity for m in marks_for_seg)

                # UCC score from severity (SPEC2.md B5)
                ucc_score = max_severity / 5.0

                # Micro suggestion: prioritize mark where this segment is the detection point (mark.id)
                # Otherwise use the mark with highest severity
                primary_mark = None
                if seg.id in mark_by_id:
                    # This segment is a detection point
                    primary_mark = mark_by_id[seg.id]
                else:
                    # Use the mark with highest severity
                    primary_mark = max(marks_for_seg, key=lambda m: m.severity)

                micro_suggestion = primary_mark.micro_suggestion if primary_mark and primary_mark.micro_suggestion else None

            else:
                reasons = []
                max_severity = None
                ucc_score = 0.0
                micro_suggestion = None

            output_segments.append(SegmentWithScores(
                id=seg.id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=seg.text,
                is_marked=is_marked,
                reasons=reasons,
                ucc_score=ucc_score,
                severity=max_severity,
                micro_suggestion=micro_suggestion
            ))

        return output_segments

    @staticmethod
    def _normalize_prerequisite(prereq: str) -> str:
        """Normalize prerequisite for deduplication."""
        normalized = prereq.strip()
        if any(ord(c) > 127 for c in normalized):
            return normalized  # Japanese: exact match
        else:
            return normalized.lower()  # English: case-insensitive
