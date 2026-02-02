"""
Text normalization for subtitle segments.
Based on SPEC.md Section 4.2.
"""
import re
from typing import List
from backend.core.models import Transcript


def normalize_text(transcript: Transcript, remove_fillers: bool = False) -> Transcript:
    """
    Normalize text in all segments.

    - Collapse multiple spaces to one
    - Trim leading/trailing spaces
    - Keep punctuation
    - Optionally remove Japanese fillers

    Args:
        transcript: Input transcript
        remove_fillers: If True, remove Japanese fillers like "えー", "あの"

    Returns:
        Transcript with normalized text
    """
    normalized_segments = []

    for seg in transcript.segments:
        text = seg.text

        # Remove fillers if requested (Japanese)
        if remove_fillers and transcript.language == "ja":
            text = _remove_japanese_fillers(text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Trim
        text = text.strip()

        # Skip empty segments after normalization
        if not text:
            continue

        # Create new segment with normalized text
        normalized_segments.append(
            seg.model_copy(update={"text": text})
        )

    # Reassign contiguous IDs
    for idx, seg in enumerate(normalized_segments):
        seg.id = idx

    return Transcript(
        video_id=transcript.video_id,
        language=transcript.language,
        segments=normalized_segments
    )


def _remove_japanese_fillers(text: str) -> str:
    """
    Remove common Japanese filler words.

    Args:
        text: Input text

    Returns:
        Text with fillers removed
    """
    fillers = [
        r'えー+',
        r'あー+',
        r'うー+',
        r'んー+',
        r'あの+',
        r'その+',
        r'まあ',
    ]

    for filler in fillers:
        text = re.sub(filler, '', text)

    return text
