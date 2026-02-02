"""WebVTT subtitle parser."""
import webvtt
from typing import List
from backend.core.models import Segment, Transcript


def parse_vtt(file_path: str, video_id: str, language: str = "ja") -> Transcript:
    """
    Parse WebVTT file into normalized transcript.

    Args:
        file_path: Path to .vtt file
        video_id: Video identifier
        language: Language code (default: "ja")

    Returns:
        Transcript object with normalized segments
    """
    vtt = webvtt.read(file_path)
    segments: List[Segment] = []

    for idx, caption in enumerate(vtt):
        # Parse timestamps to milliseconds
        start_ms = _time_to_ms(caption.start)
        end_ms = _time_to_ms(caption.end)

        # Extract and normalize text
        # For Japanese, remove newlines; for English, replace with space
        if language == 'ja':
            text = caption.text.replace('\n', '').strip()
        else:
            text = caption.text.replace('\n', ' ').strip()

        if not text:  # Skip empty segments
            continue

        segments.append(Segment(
            id=idx,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text
        ))

    # Reassign contiguous IDs after filtering
    for i, seg in enumerate(segments):
        seg.id = i

    return Transcript(
        video_id=video_id,
        language=language,
        segments=segments
    )


def parse_vtt_from_string(content: str, video_id: str, language: str = "ja") -> Transcript:
    """
    Parse WebVTT content from string.

    Args:
        content: VTT file content as string
        video_id: Video identifier
        language: Language code

    Returns:
        Transcript object
    """
    # Write to temp file and parse
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    try:
        return parse_vtt(temp_path, video_id, language)
    finally:
        os.unlink(temp_path)


def _time_to_ms(time_str: str) -> int:
    """
    Convert WebVTT timestamp to milliseconds.

    Format: HH:MM:SS.mmm or MM:SS.mmm

    Args:
        time_str: Timestamp string

    Returns:
        Milliseconds as integer
    """
    parts = time_str.split(':')

    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = '0'
        minutes, seconds = parts
    else:
        raise ValueError(f"Invalid timestamp format: {time_str}")

    # Split seconds and milliseconds
    if '.' in seconds:
        sec, ms = seconds.split('.')
        ms = ms.ljust(3, '0')[:3]  # Ensure 3 digits
    else:
        sec = seconds
        ms = '000'

    total_ms = (
        int(hours) * 3600 * 1000 +
        int(minutes) * 60 * 1000 +
        int(sec) * 1000 +
        int(ms)
    )

    return total_ms
