## Appendix B — LLM-Direct UCC Detection (No Embeddings) for YouTube Subtitle Segments

This appendix specifies an **LLM-direct** Unspoken Context Change (UCC) detector that operates on **YouTube subtitle segments** (VTT/SRT normalized to segments) **without embeddings**.  
It is intended as an **alternative / comparative method** and is designed to integrate into the existing system defined in `SPEC.md`.

---

### B0) Purpose

- Input: YouTube subtitle segments (id/start/end/text)
- Output: Segment-level UCC markings + missing prerequisite candidates + micro-suggestions
- Constraint: **No embedding computation**
- Recommended use:
  - As a **comparison (oracle-like)** baseline
  - As a “reason generator” for marked segments
  - As a prototype for interaction logic before implementing embedding pipeline

---

## B1) Inputs and Internal Data Model

### B1.1 Normalized segment format (same as core spec)
```json
{
  "video_id": "string",
  "language": "ja",
  "segments": [
    {"id": 0, "start_ms": 0, "end_ms": 2100, "text": "..." }
  ]
}
````

### B1.2 LLM call input unit

The LLM processes segments in **chunks** (windows), each chunk includes:

* `chunk_id`: int
* `segments`: array of `{id, text}`
* `established_prerequisites`: array of strings (rolling memory, see B3.3)
* `instructions`: fixed prompt (see B2)

```json
{
  "chunk_id": 3,
  "segments": [{"id": 40, "text": "..."}, ...],
  "established_prerequisites": ["...", "..."]
}
```

---

## B2) Fixed Prompt Specification

### B2.1 System prompt (fixed)

**SYSTEM_PROMPT_UCC_DIRECT**

> You detect Unspoken Context Change (UCC) in expert-to-non-expert explanations.
> UCC is the earliest point where the speaker implicitly starts requiring new prerequisite knowledge that has NOT been explicitly established earlier, even if the topic remains continuous.
> Be conservative: prefer fewer, high-precision change points.
> Output JSON only and follow the schema exactly.

### B2.2 User prompt template (fixed)

**USER_PROMPT_TEMPLATE_UCC_DIRECT**
(Variables in `{{...}}` are substituted by the system)

> Task: Detect Unspoken Context Change points in the following subtitle segments and propose minimal micro-suggestions.
>
> Definitions:
>
> * “Explicitly established” means the prerequisite was defined, explained, or exemplified earlier in a way a non-expert could follow.
> * “Missing prerequisite” must be a concrete concept/assumption (avoid vague labels like “more detail”).
> * A “change point” is the FIRST segment where the new prerequisite becomes required.
>
> Severity guide (integer 1-5):
> * 1 — Minor gap: an unfamiliar term or single concept; a non-expert can still follow the overall flow.
> * 2 — Moderate gap: a missing concept causes noticeable confusion, but the gist of the section can still be inferred.
> * 3 — Significant gap: the missing prerequisite is necessary to understand WHY the conclusion or next step holds; without it the reasoning feels arbitrary or unjustified even if the words are understood.
> * 4 — Severe gap: the section is nearly incomprehensible without prior knowledge not covered anywhere in the video.
> * 5 — Critical gap: the entire point is inaccessible; the non-expert is completely lost.
> Use severity 3 when a non-expert could repeat the words but could not explain why they are true or why they matter.
>
> You are given a rolling list of prerequisites already established earlier:
> {{ESTABLISHED_PREREQUISITES_JSON}}
> Treat it as ground truth and do NOT re-infer earlier context outside this list.
>
> Output JSON only. Schema:
> {
> "marks": [
> {
> "id": <segment_id>,
> "severity": <integer 1-5>,
> "new_required_prerequisites": ["string (1-5)"],
> "evidence": {
> "not_established_before": ["string (1-3)"],
> "topic_continuity": "high|medium|low"
> },
> "micro_suggestion": "string (<=120 chars, polite optional phrasing)",
> "mark_span_ids": [<segment_id>, <segment_id+1>, <segment_id+2>]
> }
> ],
> "established_updates": {
> "add": ["string", "... (0-8)"],
> "remove": []
> }
> }
>
> CRITICAL: The "id" field MUST be the actual segment ID from the provided segments list, NOT a zero-based index.
> For example, if the change point is at the segment with id=42, use "id": 42 (not "id": 0).
> The "mark_span_ids" should also use actual segment IDs from the segments list.
>
> Subtitle segments (id + text):
> {{SEGMENTS_JSON}}

### Notes

* `established_updates.add` is used to update rolling prerequisites after each chunk (B3.3).
* `remove` is reserved for future; keep it empty in v1.

---

## B3) Chunking and Rolling Memory

### B3.1 Why chunking is required

Subtitle transcripts can be long; the system must process in chunks within token limits.

### B3.2 Chunk parameters (defaults)

* `CHUNK_SIZE = 15` segments (matches embedding version k_minus + k_plus)
* `OVERLAP = 3` segments (to reduce boundary misses)
* Effective step = `CHUNK_SIZE - OVERLAP` = 12 segments

### B3.3 Rolling prerequisite memory

Maintain a running list:

* `established_prerequisites: string[]`

Initialization:

* empty `[]` (or optionally seed with known basics per domain)

For each chunk:

1. Send `established_prerequisites` in the prompt (as ground truth)
2. Receive `established_updates.add`
3. Append new items (deduplicate, keep max length `MAX_ESTABLISHED = 200`)
4. Use updated list for next chunk

**Dedup rule:** case-insensitive match for Latin, exact match for Japanese; strip whitespace.

---

## B4) Marking Semantics

### B4.1 Mark definition

A `mark` indicates an **earliest segment** where a prerequisite becomes required but is not established.

### B4.2 Mark span ids

`mark_span_ids` represent the range affected by the prerequisite shift.
Default guidance to LLM (implicit via prompt): mark the earliest segment, span should include that segment and the next few segments where the concept is used.

**Post-processing rule (system-side):**

* If `mark_span_ids` is missing or empty, set:

  * `mark_span_ids = [id, id+1, id+2]` clipped to segment bounds

### B4.3 Overlap resolution across chunks

Because chunks overlap, the same segment may be marked multiple times.

System-side merge rules:

* For a given segment `id`:

  * keep the mark with maximum `severity`
  * union `new_required_prerequisites` (deduplicate)
  * keep the shortest `micro_suggestion` (or prefer earlier chunk)
  * union span ids

---

## B5) Output Integration into Main `result.json`

The LLM-direct output should be integrated into the main output shape defined in `SPEC.md`:

For each segment:

* `is_marked = true` if segment id appears in any mark’s `mark_span_ids`
* `reasons = mark.new_required_prerequisites` (aggregate)
* `ucc_score` can be derived from severity (if no numeric score exists):

  * `ucc_score = severity / 5.0`

Add to `params`:

```json
{
  "detection_mode": "llm_direct",
  "chunk_size": 15,
  "overlap": 3,
  "max_established": 200,
  "model": "gpt-5.2",
  "temperature": 0.0
}
```

---

## B6) Rate Limits, Caching, and Determinism

### B6.1 Caching

Cache each chunk request/response:

* Key: `(video_id, chunk_id, segments_hash, established_hash, model_name, prompt_version)`
* Store: request JSON + response JSON

### B6.2 Determinism (recommended)

* Use lowest possible temperature (e.g., `temperature = 0.0` or equivalent)
* Fix `prompt_version` string and include it in cache key
  * Current version: `v1.3` (added 5-level severity guide to user prompt)
  * When prompt logic changes, increment version to invalidate old cache
  * Version history:
    * `v1.0`: Initial implementation
    * `v1.1`: Updated for chunk_size=15 and micro_suggestion fix
    * `v1.2`: Fixed mark.id to use actual segment IDs instead of local indices
    * `v1.3`: Added severity guide (integer 1-5 with descriptions) to user prompt template
* Enforce JSON-only output (reject and retry if invalid)

### B6.3 Retry policy

* If response is not valid JSON or fails schema:

  * retry up to `RETRY_MAX = 2`
  * on retry, prepend a short instruction:

    * “Your last output was invalid JSON. Output JSON only, matching the schema.”

---

## B7) Security & Safety Considerations

* Treat transcript text as potentially sensitive:

  * do not log raw text in plaintext unless explicitly enabled
* If using external LLM APIs:

  * allow a redaction option (emails, phone numbers)

---

## B8) Acceptance Criteria (LLM-direct mode)

* System can process `.vtt/.srt` -> normalized segments -> LLM-direct chunking -> merged marks
* Generates `result.json` with:

  * `is_marked` on segments in marked spans
  * `reasons` aggregated from marks
  * `ucc_score` derived from severity
* Produces stable results under deterministic settings (same input -> same output, within cache)

---

## B9) Recommended Minimal Implementation Tasks

1. Implement chunker: `(segments, CHUNK_SIZE, OVERLAP) -> chunk list`
2. Implement prompt builder: fill `ESTABLISHED_PREREQUISITES_JSON` + `SEGMENTS_JSON`
3. Implement LLM caller with deterministic settings + caching
4. Implement schema validator + retry
5. Implement mark merger across chunks
6. Integrate into core export pipeline (`result.json`, optional `viewer.html`)

---
