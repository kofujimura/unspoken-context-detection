"""
Batch processing pipeline for UCC detection.
"""
from pathlib import Path
from backend.core.models import (
    Transcript,
    DetectionParams,
    AnalysisResult,
    AnalyzeResponse
)
from backend.parser import parse_vtt, parse_srt, parse_vtt_from_string, parse_srt_from_string
from backend.preprocess import merge_short_segments, merge_incomplete_sentences, normalize_text
from backend.embedding import OpenAIEmbedder
from backend.detector import UCCDetector
from backend.reasons import ReasonLabeler


class UCCPipeline:
    """End-to-end UCC detection pipeline."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        params: DetectionParams = None
    ):
        """
        Initialize pipeline.

        Args:
            embedder: Embedding client
            params: Detection parameters (uses defaults if None)
        """
        self.embedder = embedder
        self.params = params or DetectionParams()

    async def analyze_file(
        self,
        file_path: str,
        video_id: str,
        language: str = "ja"
    ) -> AnalyzeResponse:
        """
        Analyze subtitle file.

        Args:
            file_path: Path to subtitle file (.vtt or .srt)
            video_id: Video identifier
            language: Language code

        Returns:
            Analysis response
        """
        try:
            # Parse
            path = Path(file_path)
            if path.suffix == '.vtt':
                transcript = parse_vtt(file_path, video_id, language)
            elif path.suffix == '.srt':
                transcript = parse_srt(file_path, video_id, language)
            else:
                return AnalyzeResponse(
                    success=False,
                    error=f"Unsupported file format: {path.suffix}"
                )

            return await self.analyze_transcript(transcript)

        except Exception as e:
            return AnalyzeResponse(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )

    async def analyze_content(
        self,
        content: str,
        file_format: str,
        video_id: str,
        language: str = "ja"
    ) -> AnalyzeResponse:
        """
        Analyze subtitle content from string.

        Args:
            content: Subtitle file content
            file_format: Format ("vtt" or "srt")
            video_id: Video identifier
            language: Language code

        Returns:
            Analysis response
        """
        try:
            # Parse
            if file_format == 'vtt':
                transcript = parse_vtt_from_string(content, video_id, language)
            elif file_format == 'srt':
                transcript = parse_srt_from_string(content, video_id, language)
            else:
                return AnalyzeResponse(
                    success=False,
                    error=f"Unsupported format: {file_format}"
                )

            return await self.analyze_transcript(transcript)

        except Exception as e:
            return AnalyzeResponse(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )

    async def analyze_transcript(self, transcript: Transcript) -> AnalyzeResponse:
        """
        Analyze transcript object.

        Args:
            transcript: Parsed transcript

        Returns:
            Analysis response
        """
        try:
            warnings = []

            # Preprocess
            transcript = merge_short_segments(
                transcript,
                min_chars=self.params.min_chars,
                max_chars=self.params.max_chars,
                max_gap_ms=self.params.max_gap_ms
            )
            # Merge incomplete sentences to improve embedding quality
            transcript = merge_incomplete_sentences(
                transcript,
                max_gap_ms=self.params.max_gap_ms,
                language=transcript.language
            )
            transcript = normalize_text(transcript, remove_fillers=False)

            if len(transcript.segments) == 0:
                return AnalyzeResponse(
                    success=False,
                    error="No segments remaining after preprocessing"
                )

            # Embed
            embeddings = await self.embedder.embed_transcript(transcript, use_cache=True)

            # Detect UCC
            detector = UCCDetector(self.params)
            result = detector.detect(transcript, embeddings)

            # Add reasons
            labeler = ReasonLabeler(self.params, language=transcript.language)
            result = labeler.add_reasons(result, transcript)

            # Collect warnings
            if result.metadata.get("warnings"):
                warnings.extend(result.metadata["warnings"])

            return AnalyzeResponse(
                success=True,
                result=result,
                warnings=warnings
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"=== Analysis error traceback ===\n{error_details}")
            return AnalyzeResponse(
                success=False,
                error=f"Analysis failed: {str(e)}"
            )
