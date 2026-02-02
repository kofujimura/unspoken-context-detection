"""SRT subtitle parser."""
import pysrt
from typing import List
from backend.core.models import Segment, Transcript


def parse_srt(file_path: str, video_id: str, language: str = "ja") -> Transcript:
    """
    Parse SRT file into normalized transcript.

    Args:
        file_path: Path to .srt file
        video_id: Video identifier
        language: Language code (default: "ja")

    Returns:
        Transcript object with normalized segments
    """
    subs = pysrt.open(file_path, encoding='utf-8')
    segments: List[Segment] = []

    for idx, sub in enumerate(subs):
        # Convert to milliseconds
        start_ms = _time_to_ms(sub.start)
        end_ms = _time_to_ms(sub.end)

        # Extract and normalize text
        # For Japanese, remove newlines; for English, replace with space
        if language == 'ja':
            text = sub.text.replace('\n', '').strip()
        else:
            text = sub.text.replace('\n', ' ').strip()

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


def parse_srt_from_string(content: str, video_id: str, language: str = "ja") -> Transcript:
    """
    Parse SRT content from string.

    Args:
        content: SRT file content as string
        video_id: Video identifier
        language: Language code

    Returns:
        Transcript object
    """
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    try:
        return parse_srt(temp_path, video_id, language)
    finally:
        os.unlink(temp_path)


def _time_to_ms(time_obj) -> int:
    """
    Convert pysrt SubRipTime to milliseconds.

    Args:
        time_obj: SubRipTime object

    Returns:
        Milliseconds as integer
    """
    return (
        time_obj.hours * 3600 * 1000 +
        time_obj.minutes * 60 * 1000 +
        time_obj.seconds * 1000 +
        time_obj.milliseconds
    )
