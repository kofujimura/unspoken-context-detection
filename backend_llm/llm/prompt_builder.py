"""
Prompt builder for LLM-direct UCC detection (SPEC2.md B2).
"""
import json
from typing import List, Dict


class PromptBuilder:
    """
    Builds prompts for LLM-direct UCC detection.

    SPEC2.md B2: Fixed prompts with variable substitution.
    """

    # Fixed system prompt (SPEC2.md B2.1)
    SYSTEM_PROMPT = """You detect Unspoken Context Change (UCC) in expert-to-non-expert explanations.
UCC is the earliest point where the speaker implicitly starts requiring new prerequisite knowledge that has NOT been explicitly established earlier, even if the topic remains continuous.
Be conservative: prefer fewer, high-precision change points.
Output JSON only and follow the schema exactly."""

    # User prompt template (SPEC2.md B2.2)
    USER_PROMPT_TEMPLATE = """Task: Detect Unspoken Context Change points in the following subtitle segments and propose minimal micro-suggestions.

Definitions:

* "Explicitly established" means the prerequisite was defined, explained, or exemplified earlier in a way a non-expert could follow.
* "Missing prerequisite" must be a concrete concept/assumption (avoid vague labels like "more detail").
* A "change point" is the FIRST segment where the new prerequisite becomes required.

You are given a rolling list of prerequisites already established earlier:
{established_prerequisites_json}
Treat it as ground truth and do NOT re-infer earlier context outside this list.

Output JSON only. Schema:
{{
  "marks": [
    {{
      "id": <segment_id>,
      "severity": 1,
      "new_required_prerequisites": ["string (1-5)"],
      "evidence": {{
        "not_established_before": ["string (1-3)"],
        "topic_continuity": "high|medium|low"
      }},
      "micro_suggestion": "string (<=120 chars, polite optional phrasing)",
      "mark_span_ids": [<segment_id>, <segment_id+1>, <segment_id+2>]
    }}
  ],
  "established_updates": {{
    "add": ["string", "... (0-8)"],
    "remove": []
  }}
}}

CRITICAL: The "id" field MUST be the actual segment ID from the provided segments list, NOT a zero-based index.
For example, if the change point is at the segment with id=42, use "id": 42 (not "id": 0).
The "mark_span_ids" should also use actual segment IDs from the segments list.

Subtitle segments (id + text):
{segments_json}"""

    @staticmethod
    def build_user_prompt(
        segments: List[Dict[str, any]],
        established_prerequisites: List[str]
    ) -> str:
        """
        Build user prompt with variable substitution.

        Args:
            segments: List of {id, text} dicts
            established_prerequisites: Rolling list of established prerequisites

        Returns:
            Formatted user prompt string
        """
        # Format established prerequisites
        established_json = json.dumps(
            established_prerequisites,
            ensure_ascii=False,
            indent=2
        )

        # Format segments
        segments_json = json.dumps(
            segments,
            ensure_ascii=False,
            indent=2
        )

        # Substitute variables
        user_prompt = PromptBuilder.USER_PROMPT_TEMPLATE.format(
            established_prerequisites_json=established_json,
            segments_json=segments_json
        )

        return user_prompt

    @staticmethod
    def build_retry_prompt(original_prompt: str) -> str:
        """
        Build retry prompt after invalid JSON response.

        Args:
            original_prompt: Original user prompt

        Returns:
            Retry prompt with warning prepended
        """
        retry_warning = "Your last output was invalid JSON. Output JSON only, matching the schema.\n\n"
        return retry_warning + original_prompt
