# Unspoken Context Change Detection System

An experimental system for detecting **Unspoken Context Change (UCC)** points in lecture transcripts, where speakers implicitly introduce new prerequisite knowledge without explicit explanation.

## What is Unspoken Context Change (UCC)?

**Unspoken Context Change (UCC)** occurs when an expert speaker transitions to requiring new prerequisite knowledge that has **not been explicitly established earlier**, even when the topic appears to remain continuous. This phenomenon is particularly problematic in educational content where:

- The speaker assumes prior knowledge that learners may lack
- New concepts or terminology are introduced without definition
- The prerequisite shift is implicit rather than explicit
- Topic continuity masks the conceptual gap

**Example:**
> "We use cosine similarity to measure semantic relationships. The inner product of the normalized vectors gives us the similarity score."

The speaker jumps from "cosine similarity" to "inner product" and "normalized vectors" without explaining these prerequisites, creating an unspoken context change.

## System Overview

This system provides **two complementary detection approaches** for identifying UCC points in subtitle transcripts:

### 1. Embedding-based Detection (Fast, Cost-Effective)

Uses **semantic embeddings** to detect prerequisite shifts through context debt analysis.

**Key Features:**
- Fast processing with batch embeddings
- Low cost (embedding API only)
- Sliding window analysis with asymmetric windows
- Context Debt metric: measures semantic gap between past and current context
- Automatic keyword extraction for missing prerequisites

**Best for:**
- Large-scale transcript analysis
- Quick prototyping and experimentation
- Cost-sensitive applications
- Real-time or near-real-time processing

**See `SPEC.md` for detailed algorithm specification.**

### 2. LLM-Direct Detection (High-Precision, Reasoning-Based)

Uses **direct LLM reasoning** with structured prompts to identify prerequisite shifts.

**Key Features:**
- High-precision detection with explicit reasoning
- Severity scoring (1-5) for each detected shift
- Micro-suggestions for learners (≤120 chars)
- Rolling prerequisite memory across chunks
- Explicit identification of missing concepts
- Deterministic caching for reproducibility

**Best for:**
- High-stakes educational content
- Generating explanatory interventions
- Oracle/baseline for evaluation
- Research and fine-grained analysis

**See `SPEC2.md` for detailed specification.**

## Comparison: Embedding vs LLM-Direct

| Aspect | Embedding-based | LLM-Direct |
|--------|----------------|------------|
| **Speed** | Fast (batch embeddings) | Moderate (sequential LLM calls) |
| **Cost** | Low (embedding only) | Higher (chat API calls) |
| **Precision** | Good (statistical detection) | Excellent (explicit reasoning) |
| **Explainability** | Keyword extraction | Explicit prerequisites + suggestions |
| **Severity scoring** | Continuous (0-1) | Discrete (1-5) + rationale |
| **Cache support** | Embeddings cached | Full response cached |
| **Best use case** | Large-scale analysis | High-precision interventions |

## Quick Start

### Prerequisites

1. Python 3.8+
2. OpenAI API key

### Installation

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Note:** If you see warnings about scripts not being on PATH, add `~/.local/bin` to your PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Running the System

#### Embedding-based Detection (Port 8000)

```bash
# Using the startup script
./start_embedding.sh

# Or manually
uvicorn backend.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser to access the interactive viewer.

#### LLM-Direct Detection (Port 8001)

```bash
# Using the startup script
./start_llm.sh

# Or manually
uvicorn backend_llm.main:app --reload --port 8001
```

Open http://localhost:8001 in your browser to access the interactive viewer.

## Usage Examples

### Web Interface

Both modes provide an interactive HTML viewer where you can:
1. Upload subtitle files (.vtt or .srt)
2. Configure detection parameters
3. View marked segments with scores
4. Filter results by threshold
5. Inspect detailed scores and reasons

### API Usage

#### Embedding-based API

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@subtitles.vtt" \
  -F "video_id=XXXX" \
  -F "language=ja" \
  -F "k_minus=12" \
  -F "k_plus=3" \
  -F "lambda_param=1.5"
```

**Parameters:**
- `k_minus`: Past window size (default: 12)
- `k_plus`: Current window size (default: 3)
- `lambda_param`: Threshold multiplier (default: 1.5)
- `tau_topic`: Topic continuity threshold (default: 0.55)

#### LLM-Direct API

```bash
curl -X POST "http://localhost:8001/api/analyze" \
  -F "file=@subtitles.vtt" \
  -F "video_id=XXXX" \
  -F "language=ja" \
  -F "chunk_size=15" \
  -F "overlap=3" \
  -F "model=gpt-4o-mini"
```

**Parameters:**
- `chunk_size`: Segments per chunk (default: 15)
- `overlap`: Overlap between chunks (default: 3)
- `model`: LLM model name (default: gpt-4o-mini)

### Command-Line Interface

```bash
python cli.py analyze \
  --input path/to/subtitles.vtt \
  --video-id XXXX \
  --language ja \
  --output-dir results/
```

## Project Structure

```
unspoken-context-detection/
├── backend/              # Embedding-based detection (SPEC.md)
│   ├── main.py          # FastAPI server (port 8000)
│   ├── core/            # Data models
│   ├── parser/          # VTT/SRT parsers
│   ├── preprocess/      # Segment normalization & merging
│   ├── embedding/       # OpenAI embeddings with caching
│   ├── detector/        # UCC detection algorithm
│   │   ├── windows.py   # Context window computation
│   │   ├── metrics.py   # Context Debt & scoring
│   │   └── ucc_detector.py
│   ├── reasons/         # Keyword-based reason extraction
│   └── batch/           # Pipeline orchestration
│
├── backend_llm/         # LLM-direct detection (SPEC2.md)
│   ├── main.py          # FastAPI server (port 8001)
│   ├── core/            # Data models (marks, severity)
│   ├── chunker/         # Overlapping chunk creation
│   ├── llm/             # LLM caller with caching
│   │   ├── llm_caller.py    # API wrapper
│   │   └── prompt_builder.py # Fixed prompts
│   ├── merger/          # Mark merging across chunks
│   └── pipeline.py      # Full pipeline with rolling memory
│
├── frontend/            # Embedding-based viewer
│   ├── viewer.html
│   └── viewer.js
│
├── frontend_llm/        # LLM-direct viewer
│   ├── viewer.html
│   └── viewer.js        # Displays severity & suggestions
│
├── examples/            # Example subtitle files
├── .cache/              # Cached embeddings & LLM responses
├── cli.py               # Command-line interface
├── start_embedding.sh   # Start embedding server
├── start_llm.sh         # Start LLM server
├── SPEC.md              # Embedding-based specification
├── SPEC2.md             # LLM-direct specification
└── README.md            # This file
```

## Detailed Specifications

For complete technical specifications, algorithm details, and parameter tuning:

- **Embedding-based Detection**: See [`SPEC.md`](./SPEC.md)
  - Detailed algorithm description
  - Context Debt computation
  - Sliding window mechanics
  - Score normalization methods
  - Parameter tuning guidelines

- **LLM-Direct Detection**: See [`SPEC2.md`](./SPEC2.md)
  - Prompt engineering details
  - Chunking strategy with overlap
  - Rolling prerequisite memory
  - Mark merging rules
  - Caching and determinism

## Architecture Details

### Embedding-based Mode Architecture

```
Input (VTT/SRT) → Parser → Preprocessing → Embeddings
                                              ↓
Result JSON ← Keyword Reasons ← UCC Detector ← Context Windows
              ↓
         HTML Viewer
```

**Core Algorithm:**
1. Parse and preprocess subtitle segments
2. Compute embeddings for all segments (with caching)
3. Sliding window: compute context embeddings (mean pooling)
4. Calculate Context Debt: `D(t) = ||c_plus(t) - c_minus(t)||`
5. Detect spikes: `ΔD(t) > μ + λσ`
6. Mark segments and extract keyword reasons

### LLM-Direct Mode Architecture

```
Input (VTT/SRT) → Parser → Preprocessing → Chunker
                                              ↓
                           Rolling Prerequisites ← LLM (cached)
                                              ↓
Result JSON ← Mark Merger ← Chunk Results (marks + updates)
              ↓
         HTML Viewer
```

**Core Algorithm:**
1. Parse and preprocess subtitle segments
2. Create overlapping chunks (size 15, overlap 3)
3. For each chunk:
   - Build prompt with segments + established prerequisites
   - Call LLM with deterministic settings (temp=0.0)
   - Extract marks (id, severity, prerequisites, suggestion)
   - Update rolling prerequisites
4. Merge marks across overlapping chunks
5. Assign scores to segments based on mark spans

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - Embedding mode
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_CACHE_DIR=.cache/embeddings

# Optional - LLM mode
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_CACHE_DIR=.cache/llm_chunks
```

### Detection Parameters

Both modes support parameter customization through the API or web interface. See `SPEC.md` and `SPEC2.md` for complete parameter descriptions.

## Output Format

Both modes produce compatible JSON output:

```json
{
  "video_id": "xxxx",
  "language": "ja",
  "params": { ... },
  "segments": [
    {
      "id": 42,
      "start_ms": 45100,
      "end_ms": 48600,
      "text": "コサイン類似度は内積を使います。",
      "is_marked": true,
      "reasons": ["内積", "ベクトルのノルム", "正規化"],
      "scores": {
        "ucc_score": 0.88,
        "context_debt_delta": 0.61
      }
    }
  ]
}
```

LLM mode additionally includes:
- `severity`: 1-5 scoring
- `micro_suggestion`: Short hint for learners

## Development

### Running Tests

```bash
# Run test script
python test_changes.py
```

### Adding New Features

1. **Embedding mode**: Modify files in `backend/`
2. **LLM mode**: Modify files in `backend_llm/`
3. Update corresponding SPEC.md or SPEC2.md
4. Test with example subtitle files in `examples/`

## Use Cases

### Educational Content Analysis
- Identify prerequisite gaps in lecture videos
- Generate learner interventions at UCC points
- Assess content difficulty and prerequisite density

### Content Authoring
- Quality assurance for educational videos
- Prerequisite dependency mapping
- Suggesting points for additional explanation

### Adaptive Learning Systems
- Trigger prerequisite reviews at UCC points
- Personalize content based on learner background
- Generate Just-In-Time interventions

## Limitations

### Embedding-based Mode
- May miss subtle conceptual shifts
- Keyword extraction can be noisy
- Requires manual threshold tuning

### LLM-Direct Mode
- Higher API costs for long transcripts
- Processing time scales with transcript length
- Quality depends on LLM model capability

## License

[Specify your license here]

## Citation

If you use this system in your research, please cite:

```bibtex
[Add citation information]
```

## Contributing

Contributions are welcome! Please:
1. Read `SPEC.md` and `SPEC2.md` for implementation details
2. Follow existing code structure
3. Add tests for new features
4. Update documentation

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See `SPEC.md` for algorithm details
- See `SPEC2.md` for LLM mode details

---

**Note:** This is an experimental research system. Detection accuracy depends on transcript quality, language, and domain characteristics. Always validate results with domain experts.
