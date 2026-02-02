# Quick Start Guide

## Installation

1. **Run setup script:**
```bash
./setup.sh
```

2. **Configure environment:**
```bash
# Edit .env file and add your OpenAI API key
nano .env
```

Add your API key:
```
OPENAI_API_KEY=sk-...
```

3. **Activate virtual environment:**
```bash
source venv/bin/activate
```

## Usage

### Option 1: Web UI (Recommended)

1. **Start the server:**
```bash
cd backend
python -m uvicorn main:app --reload
```

2. **Open your browser:**
```
http://localhost:8000
```

3. **Upload a subtitle file** (.vtt or .srt) and analyze!

### Option 2: CLI

**Basic usage:**
```bash
python cli.py analyze \
  --input examples/sample.vtt \
  --video-id sample_lecture \
  --language ja
```

**Advanced options:**
```bash
python cli.py analyze \
  --input path/to/subtitles.vtt \
  --video-id lecture_001 \
  --language ja \
  --k-minus 12 \
  --k-plus 3 \
  --lambda-param 1.5 \
  --tau-topic 0.6 \
  --output-dir output/
```

**Parameters:**
- `--k-minus`: Past window size (default: 12)
- `--k-plus`: Current window size (default: 3)
- `--k`: Legacy parameter to set both windows (backward compatibility)
- `--lambda-param`: Threshold multiplier (default: 1.5)
- `--tau-topic`: Topic continuity threshold (default: 0.55)
- `--mark-length`: Number of segments to mark (default: 3)
- `--n-reasons`: Number of reason keywords (default: 5)
- `--no-cache`: Disable embedding cache
- `--no-html`: Skip HTML viewer export

## Output

### CLI Output Files

The CLI creates two files in the output directory:

1. **`{video_id}_result.json`** - Complete analysis data
2. **`{video_id}_viewer.html`** - Standalone HTML viewer (can be opened in browser)

### JSON Structure

```json
{
  "video_id": "sample_lecture",
  "language": "ja",
  "params": { ... },
  "segments": [
    {
      "id": 0,
      "start_ms": 0,
      "end_ms": 3500,
      "text": "今日は機械学習の基礎について学びます。",
      "scores": {
        "topic_similarity": 0.85,
        "context_debt": 1.23,
        "context_debt_delta": 0.45,
        "ucc_score": 0.72
      },
      "is_marked": true,
      "reasons": ["機械学習", "基礎"]
    }
  ],
  "metadata": {
    "total_segments": 24,
    "marked_count": 5,
    "threshold": 0.234
  }
}
```

## Understanding the Results

### Metrics

- **Context Debt (D)**: Magnitude of implicit prerequisite load at each point
- **Context Debt Delta (ΔD)**: Change in context debt from previous segment
- **UCC Score**: Normalized score (0-1) indicating likelihood of unspoken context change
- **Topic Similarity**: Cosine similarity between past and current context windows

### Marked Segments

Segments are marked when:
1. ΔD > μ + λ × σ (jump condition)
2. Topic similarity ≥ tau_topic (optional continuity mask)

### Reason Keywords

For each marked segment, the system extracts keywords that appear in the current window but not in the past window, suggesting new concepts that may require prerequisite knowledge.

## API Usage

### Python API

```python
import asyncio
from backend.core.models import DetectionParams
from backend.embedding import OpenAIEmbedder
from backend.batch import UCCPipeline

async def main():
    embedder = OpenAIEmbedder()
    params = DetectionParams(k_minus=12, k_plus=3, lambda_param=1.5)
    pipeline = UCCPipeline(embedder, params)

    result = await pipeline.analyze_file(
        "examples/sample.vtt",
        "sample_lecture",
        "ja"
    )

    if result.success:
        print(f"Marked segments: {result.result.marked_count}")

asyncio.run(main())
```

### REST API

**Analyze endpoint:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@examples/sample.vtt" \
  -F "video_id=sample_lecture" \
  -F "language=ja" \
  -F "k_minus=12" \
  -F "k_plus=3" \
  -F "lambda_param=1.5"
```

**Get default parameters:**
```bash
curl http://localhost:8000/api/params
```

## Future: Real-time Mode

The architecture is designed to support real-time streaming in the future. The core `UCCDetector` can process segments one at a time with a rolling window, enabling live lecture analysis as audio is transcribed.

## Troubleshooting

**Error: OPENAI_API_KEY not found**
- Make sure you've created `.env` and added your API key

**Error: sudachipy not available**
- The system will fall back to simple tokenization
- For better Japanese analysis, ensure sudachipy is installed: `pip install sudachipy sudachidict-core`

**Transcript too short warning**
- Increase the window size or use a longer transcript
- Minimum recommended: N ≥ k_minus + k_plus + 1 segments

**Cache location**
- Embeddings are cached in `.cache/embeddings/`
- To clear cache: `rm -rf .cache/embeddings/`

## Next Steps

- Experiment with different parameters (k_minus, k_plus, lambda_param, tau_topic)
- Try your own lecture transcripts
- Compare results with human annotations
- Adjust mark_length for your use case
- Compare with LLM-Direct mode (see QUICKSTART_LLM.md)

For more details, see `SPEC.md` for the complete algorithm specification.
