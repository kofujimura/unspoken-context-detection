````markdown
# SPEC.md — Unspoken Context Change Detection (YouTube Transcript Marking System)

**Goal:** Build an experimental system that takes YouTube lecture video transcripts (subtitle-equivalent text) and performs **Unspoken Context Change (UCC) Detection**, then **marks** subtitle segments where **implicit prerequisite shifts / missing prerequisite information** are likely.

This spec is designed to be handed to a coding agent. It is language-agnostic and focuses on **precise I/O, algorithm, parameters, and edge cases**.

---

## 0) System Summary

### Input
- YouTube subtitle transcripts (preferred: `.vtt`, supported: `.srt`, optional: JSON)

### Processing
- Normalize & merge short subtitle segments
- Compute embeddings per segment with an embedding model
- Sliding-window detection of UCC via **Context Debt** and its change
- Attach **marking** to relevant subtitle segments
- Produce optional “reason labels” (missing concept candidates)

### Output
- `result.json` (primary)
- `viewer.html` (optional but strongly recommended for inspection)

---

## 1) Definitions

### Unspoken Context Change (UCC)
A point in an explanation where **topic remains coherent**, but **implicit prerequisite demands shift/increase** without being explicitly explained.

### Context Debt (Embedding Proxy)
A lightweight embedding-based approximation of implicit prerequisite load, computed by contrasting “past explained” vs “current required” context windows.

---

## 2) Data Model (Normalized Internal Format)

### 2.1 Normalized transcript object
```json
{
  "video_id": "string",
  "language": "ja",
  "segments": [
    {
      "id": 0,
      "start_ms": 1230,
      "end_ms": 3560,
      "text": "string"
    }
  ]
}
````

### Requirements

* `id` must be contiguous `0..N-1` after preprocessing
* `start_ms < end_ms`
* `text` is trimmed; internal newlines removed

---

## 3) Input Formats & Parsing

### 3.1 Supported formats

* **WebVTT** (`.vtt`) — recommended
* **SRT** (`.srt`)
* Optional: `transcript.json` in normalized format (bypass parsing)

### 3.2 Parsing requirements

* Parse timestamps into milliseconds
* Extract text lines per cue, join with spaces
* Preserve punctuation (important for meaning boundaries)

### 3.3 Parser outputs

Return normalized transcript object (Section 2.1).

---

## 4) Preprocessing

### 4.1 Segment merge (stability + readability)

YouTube subtitles often have very short segments; merge to reduce noise.

#### Parameters (defaults)

* `MIN_CHARS = 18`  (segments shorter than this are merge candidates)
* `MAX_CHARS = 120` (merged segment must not exceed this)
* `MAX_GAP_MS = 800` (merge allowed if next start - current end <= this)

#### Merge algorithm

For each segment `i`:

1. If `len(text_i) < MIN_CHARS`:

   * Try merge with `i+1` if:

     * `gap_ms(i, i+1) <= MAX_GAP_MS` AND `len(text_i + text_{i+1}) <= MAX_CHARS`
   * Else try merge with `i-1` with same constraints.
2. When merging:

   * `start_ms` = min of merged
   * `end_ms` = max of merged
   * `text` = `"text_a text_b"` (single space join)
3. Reassign ids to contiguous `0..N-1`.

### 4.2 Text normalization

* Collapse multiple spaces to one
* Trim leading/trailing spaces
* Keep punctuation
* Optional (default OFF): remove fillers (“えー”, “あの”) for Japanese transcripts

---

## 5) Embedding Module

### 5.1 Embedding function interface

```
embed(texts: string[]) -> float[d][]
```

### Requirements

* Must support batch embedding for speed/cost
* Must handle max token length:

  * Strategy A (default): truncate
  * Strategy B (optional): summarize then embed
* Deterministic settings recommended where applicable

### 5.2 What gets embedded

* Each preprocessed segment text `u_t.text` is embedded into `e_t ∈ R^d`

---

## 6) Core Detection Algorithm (Embedding-based)

### 6.1 Notation

* Segments: `u_0..u_{T-1}`
* Embeddings: `e_t ∈ R^d`
* Window sizes:
  * `k_minus`: Past window size (how far back to look)
  * `k_plus`: Current window size (forward-looking)
  * `k` (legacy): When specified, sets both k_minus and k_plus to same value for backward compatibility

### 6.2 Context windows

For each index `t`:

* Past window:

  * `W_minus(t) = {t-k_minus, ..., t-1}` (size: k_minus segments before t)
* Current window:

  * `W_plus(t)  = {t, ..., t+k_plus-1}` (size: k_plus segments starting from t)

Boundary handling:

* Clip indices to `[0, T-1]`
* If window is empty (e.g., t=0 and past window empty), use available elements only.
* If both windows are too small (e.g., transcript too short), detection still runs but may be noisy; system should warn.
* Minimum required segments: `T >= k_minus + k_plus + 1`

### 6.3 Context embeddings (mean pooling)

Let `mean(E)` be arithmetic mean of vectors in E.

* `c_minus(t) = mean({e_i | i ∈ W_minus(t)})`
* `c_plus(t)  = mean({e_i | i ∈ W_plus(t)})`

### 6.4 Topic continuity (reference signal)

Cosine similarity:

* `S_topic(t) = cos(c_minus(t), c_plus(t))`

This is used to optionally **exclude strong topic shifts**, since UCC aims for “topic continuity but prerequisite discontinuity.”

### 6.5 Context Debt (proxy)

* `d_t = c_plus(t) - c_minus(t)`
* `D(t) = ||d_t||_2`

### 6.6 Change magnitude

* `ΔD(t) = D(t) - D(t-1)` for `t>=1`
* For `t=0`, define `ΔD(0)=0`

### 6.7 Candidate UCC rule (threshold)

Compute mean and std of `ΔD(t)` over transcript:

* `mu = mean(ΔD)`
* `sigma = std(ΔD)`

Candidate if:

1. Jump condition:

* `ΔD(t) > mu + lambda_param * sigma`

2. Optional topic continuity mask:

* `S_topic(t) >= tau_topic`

#### Default parameters

* `k_minus = 12` (past window: look back 12 segments)
* `k_plus = 3` (current window: forward-looking 3 segments)
* `lambda_param = 1.5` (threshold multiplier)
* `tau_topic = 0.55` (optional; can be disabled)

Note: Legacy parameter `k = 3` can be used to set both k_minus and k_plus to 3 for simpler symmetric windows.

### 6.8 Scoring

For each `t`, compute:

* `context_debt = D(t)`
* `context_debt_delta = ΔD(t)`
* `topic_similarity = S_topic(t)`
* `ucc_score` normalized to `[0,1]`

**IMPORTANT: Score Assignment**

The scores computed at index `t` are assigned to the **last segment** of the current window:
* Assign all scores to segment index `t+k_plus-1`
* This ensures scores reflect the impact of newly added context at the end of W_plus(t)
* Example: If t=10 and k_plus=3, W_plus(10)={10,11,12}, scores are assigned to segment 12

Normalization options:

* `minmax`: `ucc_score = (ΔD - min)/(max-min)` (clip 0..1)
* `zsigmoid` (recommended): `ucc_score = sigmoid((ΔD - mu)/sigma)`

Default: `zsigmoid` (more stable across videos)

### 6.9 Marking policy

When segment `i` has a high UCC score (detected as candidate), we mark a range ending at that segment.

#### Marking span logic

Since scores at segment `i` were computed from position `t = i - k_plus + 1`, and reflect the context shift in W_plus(t), we mark the last L segments ending at i:

* `MARK_SPAN = [i-L+1, i]` where `L` is `mark_length` parameter (default: 3)
* Example: If segment i=15 is detected and L=3, mark segments [13, 14, 15]

**Rationale:** The UCC score spike at segment i is caused by newly introduced concepts in the **tail** of W_plus(t), which ends at segment i. Marking these segments highlights the actual source of the prerequisite shift.

#### Default marking span

* `mark_length = 3` (last 3 segments)
* This typically marks the k_plus window when k_plus=3

Alternative option:

* Set `mark_length = k_plus` to mark the entire current window

### 6.10 Overlap handling

When multiple candidates mark the same segment:

* `ucc_score` = max
* `reasons` = union (deduplicate)

---

## 7) “Reason” Labels (Missing Concept Candidates)

Purpose: provide human-readable hints for why a segment was marked.

### 7.1 Mode A (default, low cost): Keyword diff

For each candidate `t`:

* `text_minus = concat texts in W_minus(t)`
* `text_plus  = concat texts in W_plus(t)`
  Extract keywords from each, then compute `plus - minus` top N.

#### Keyword extraction

Japanese recommended:

* morphological analysis (Sudachi/Mecab) → nouns and compound nouns

Fallback (no morphology available):

* simple tokenization by whitespace/punctuation + heuristic noun detection (low quality but acceptable for prototype)

Parameters:

* `N_REASONS = 5`

### 7.2 Mode B (optional, oracle): LLM reasoning

For each candidate `t`, pass window text and ask the LLM to infer missing prerequisite concepts.

* This is **optional** and must be toggled.
* Used for analysis, not required for base system.

Output format:

```json
{
  "missing_prerequisites": ["concept1", "concept2", "..."],
  "explanation": "short"
}
```

---

## 8) Output Specification

### 8.1 Primary output JSON (`result.json`)

```json
{
  "video_id": "xxxx",
  "language": "ja",
  "params": {
    "k_minus": 12,
    "k_plus": 3,
    "k": null,
    "lambda_param": 1.5,
    "tau_topic": 0.55,
    "mark_length": 3,
    "score_norm": "zsigmoid",
    "reason_mode": "keyword_diff",
    "n_reasons": 5,
    "min_chars": 18,
    "max_chars": 120,
    "max_gap_ms": 800
  },
  "segments": [
    {
      "id": 12,
      "start_ms": 45100,
      "end_ms": 48600,
      "text": "コサイン類似度は内積を使います。",
      "scores": {
        "topic_similarity": 0.82,
        "context_debt": 1.73,
        "context_debt_delta": 0.61,
        "ucc_score": 0.88
      },
      "is_marked": true,
      "reasons": ["内積", "ベクトルのノルム", "正規化"]
    }
  ]
}
```

### 8.2 Optional HTML viewer (`viewer.html`)

Requirements:

* Render segments in time order
* Highlight `is_marked` segments
* On hover/click: show `ucc_score`, `ΔD`, `topic_similarity`, `reasons`
* Provide filter slider for `ucc_score` threshold (post hoc exploration)

---

## 9) CLI / API Requirements

### 9.1 CLI (recommended minimal)

```
ucc_detect \
  --input path/to/subtitles.vtt \
  --video_id XXXX \
  --language ja \
  --k_minus 12 \
  --k_plus 3 \
  --lambda_param 1.5 \
  --tau_topic 0.55 \
  --mark_length 3 \
  --reason_mode keyword_diff \
  --output_dir out/
```

Or use legacy symmetric window:

```
ucc_detect \
  --input path/to/subtitles.vtt \
  --video_id XXXX \
  --language ja \
  --k 3 \
  --lambda_param 1.5 \
  --output_dir out/
```

Outputs:

* `out/result.json`
* `out/viewer.html` (if enabled)

### 9.2 Programmatic API (optional)

* `parse_transcript(path) -> transcript`
* `preprocess(transcript, params) -> transcript`
* `compute_embeddings(transcript, embedder) -> embeddings`
* `detect_ucc(transcript, embeddings, params) -> result`
* `export_json(result, path)`
* `export_html(result, path)`

---

## 10) Module Breakdown (Recommended)

```
/parser
  vtt_parser.*
  srt_parser.*
/preprocess
  merge_segments.*
  normalize_text.*
/embedding
  embedder_interface.*
  openai_embedder.* (optional)
  local_embedder.* (optional)
/detector
  windows.*
  metrics.*
  ucc_threshold.*
  marker.*
/reasons
  keyword_diff.*
  llm_oracle.* (optional)
/export
  json_export.*
  html_viewer_template.*
/cli
  main.*
```

---

## 11) Edge Cases & Error Handling

* Transcript too short (`T < k_minus + k_plus + 1`):

  * Proceed with clipped windows
  * Emit warning: `"short_transcript_warning": true`
* Missing timestamps:

  * Reject input (hard error)
* Empty text segments after normalization:

  * Drop them and reindex
* Embedding failures (rate limits/timeouts):

  * Retry with exponential backoff (bounded)
  * Cache embeddings to disk (`.cache/embeddings.jsonl`) by `(video_id, segment_id, text_hash)`

---

## 12) Acceptance Criteria

Must-have:

* Parse `.vtt` and `.srt` into normalized segments
* Preprocess merge + normalization works
* Compute embeddings for all segments
* Produce `ucc_score` per segment
* Mark segments with `is_marked` based on detection + marking span
* Produce `result.json`

Nice-to-have:

* HTML viewer with hover details
* Optional LLM oracle for reasons
* Embedding cache

---

## 13) Default Parameters (Current Implementation)

### Window parameters
* `k_minus = 12` (past window: look back 12 segments)
* `k_plus = 3` (current window: forward-looking 3 segments)
* `k = null` (legacy parameter for backward compatibility)

### Detection parameters
* `lambda_param = 1.5` (threshold multiplier)
* `tau_topic = 0.55` (topic continuity threshold, optional)
* `mark_length = 3` (marks last 3 segments ending at detected segment)
* `score_norm = zsigmoid` (score normalization method)

### Preprocessing parameters
* `min_chars = 18` (merge threshold)
* `max_chars = 120` (max merged segment length)
* `max_gap_ms = 800` (max gap for merging)

### Reason extraction parameters
* `reason_mode = keyword_diff` (extraction method)
* `n_reasons = 5` (max number of reason keywords)

### Notes on window size choice
* `k_minus = 12`: Provides sufficient context history for prerequisite establishment
* `k_plus = 3`: Focuses on immediate forward context to detect prerequisite shifts
* Asymmetric windows (k_minus > k_plus) improve detection by contrasting larger past context against focused current context
* Use legacy `k = 3` for symmetric windows in simpler scenarios

---

## 14) Notes for Future Extensions (Non-blocking)

* Near-real-time mode: process streaming ASR chunks, maintain rolling window
* Audience-adaptive mode: personalize thresholds per target audience
* Evaluation hooks: export candidate indices for annotation tooling

---

```
```
