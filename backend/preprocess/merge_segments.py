"""
Segment merging for short subtitle segments.
Based on SPEC.md Section 4.1.
"""
from typing import List
import re
from backend.core.models import Segment, Transcript


def merge_short_segments(
    transcript: Transcript,
    min_chars: int = 18,
    max_chars: int = 120,
    max_gap_ms: int = 800
) -> Transcript:
    """
    Merge short segments to reduce noise.

    Algorithm:
    1. If segment is shorter than min_chars, try to merge with next or previous
    2. Merging allowed if gap <= max_gap_ms and combined length <= max_chars
    3. Reassign contiguous IDs after merging

    Args:
        transcript: Input transcript
        min_chars: Minimum character count for standalone segment
        max_chars: Maximum character count for merged segment
        max_gap_ms: Maximum time gap for merging

    Returns:
        Transcript with merged segments
    """
    if not transcript.segments:
        return transcript

    segments = list(transcript.segments)  # Make a copy
    merged = []
    i = 0

    while i < len(segments):
        current = segments[i]

        # Check if current segment is short
        if len(current.text) < min_chars:
            merged_segment = None

            # Try merge with next
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                gap_ms = next_seg.start_ms - current.end_ms
                combined_text = f"{current.text} {next_seg.text}"

                if gap_ms <= max_gap_ms and len(combined_text) <= max_chars:
                    merged_segment = Segment(
                        id=0,  # Will be reassigned later
                        start_ms=current.start_ms,
                        end_ms=next_seg.end_ms,
                        text=combined_text.strip()
                    )
                    i += 2  # Skip both segments
                    merged.append(merged_segment)
                    continue

            # Try merge with previous
            if merged_segment is None and merged:
                prev_seg = merged[-1]
                gap_ms = current.start_ms - prev_seg.end_ms
                combined_text = f"{prev_seg.text} {current.text}"

                if gap_ms <= max_gap_ms and len(combined_text) <= max_chars:
                    # Update previous segment
                    merged[-1] = Segment(
                        id=0,
                        start_ms=prev_seg.start_ms,
                        end_ms=current.end_ms,
                        text=combined_text.strip()
                    )
                    i += 1
                    continue

        # No merge possible, add as-is
        merged.append(current)
        i += 1

    # Reassign contiguous IDs
    for idx, seg in enumerate(merged):
        seg.id = idx

    return Transcript(
        video_id=transcript.video_id,
        language=transcript.language,
        segments=merged
    )


def merge_incomplete_sentences(
    transcript: Transcript,
    max_gap_ms: int = 2000,
    language: str = "ja"
) -> Transcript:
    """
    Merge segments that end mid-sentence with the next segment.

    For Japanese: Segments not ending with sentence terminators (。？！) are
    considered incomplete and will be merged with the next segment if the gap is small.

    For English: Segments not ending with sentence terminators (.?!) are merged.

    Args:
        transcript: Input transcript
        max_gap_ms: Maximum time gap to allow merging (default: 2000ms)
        language: Language code (ja or en)

    Returns:
        Transcript with merged incomplete sentences
    """
    if not transcript.segments:
        return transcript

    # Define sentence ending patterns based on language
    if language == 'ja':
        # Japanese sentence enders: 。？！and common endings like です、ます、た、だ
        sentence_enders = re.compile(r'[。？！]$|です$|ます$|でした$|ました$|だ$|た$|ね$|よ$|か$')
    else:
        # English sentence enders
        sentence_enders = re.compile(r'[.?!]$')

    segments = list(transcript.segments)
    merged = []
    i = 0

    while i < len(segments):
        current = segments[i]

        # Check if current segment ends with a sentence terminator
        ends_complete = sentence_enders.search(current.text.strip()) is not None

        # If incomplete and there's a next segment, try to merge
        if not ends_complete and i + 1 < len(segments):
            next_seg = segments[i + 1]
            gap_ms = next_seg.start_ms - current.end_ms

            # Only merge if gap is reasonable
            if gap_ms <= max_gap_ms:
                # Merge current with next
                if language == 'ja':
                    combined_text = f"{current.text}{next_seg.text}"
                else:
                    combined_text = f"{current.text} {next_seg.text}"

                merged_segment = Segment(
                    id=0,  # Will be reassigned later
                    start_ms=current.start_ms,
                    end_ms=next_seg.end_ms,
                    text=combined_text.strip()
                )
                merged.append(merged_segment)
                i += 2  # Skip both segments
                continue

        # Add segment as-is
        merged.append(current)
        i += 1

    # Reassign contiguous IDs
    for idx, seg in enumerate(merged):
        seg.id = idx

    return Transcript(
        video_id=transcript.video_id,
        language=transcript.language,
        segments=merged
    )
