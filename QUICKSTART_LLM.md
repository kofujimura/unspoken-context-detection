# Quick Start Guide - LLM-Direct Mode

This guide will help you get started with the LLM-Direct UCC detection mode (SPEC2.md).

## What is LLM-Direct Mode?

LLM-Direct mode uses OpenAI's Chat API to directly analyze transcript chunks and identify:
- Unspoken context changes (UCC points)
- Missing prerequisites at each change point
- Severity ratings (1-5)
- Micro-suggestions for improvement

## Prerequisites

- Python 3.8+
- OpenAI API key
- Subtitle file (.vtt or .srt)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r backend_llm/requirements.txt
```

### 2. Configure API Key

Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_CACHE_DIR=.cache/llm_chunks
```

### 3. Start the Server

```bash
./start_llm.sh
```

Or manually:
```bash
uvicorn backend_llm.main:app --reload --port 8001
```

### 4. Open the Viewer

Navigate to: http://localhost:8001

### 5. Upload and Analyze

1. Enter a Video ID
2. Select language (Japanese/English)
3. Adjust parameters (optional):
   - **Chunk Size**: Number of segments per chunk (default: 15)
   - **Overlap**: Overlapping segments between chunks (default: 3)
   - **Model**: LLM model to use (default: gpt-4o-mini)
4. Upload your subtitle file
5. Click "Analyze"

## Understanding the Results

### Stats

- **Total Segments**: Number of subtitle segments
- **Marked Segments**: Segments identified as UCC points
- **Total Chunks**: Number of chunks processed
- **Total Marks**: Number of unique UCC detections

### Chart

- **UCC Score**: Normalized severity score (0-1)
- **Severity**: Raw severity rating (1-5)

### Segments List

For each marked segment:
- **Score**: UCC confidence score
- **Severity Badge**: Color-coded severity (1=green, 5=red)
- **Missing Prerequisites**: Identified missing concepts
- **Micro-suggestion**: Optional suggestion for improvement

## Parameters

### Chunk Size (default: 15)
- Matches embedding mode window size (k_minus + k_plus = 12 + 3)
- Larger chunks: More context, slower processing, higher API cost
- Smaller chunks: Less context, faster processing, lower API cost

### Overlap (default: 3)
- Reduces boundary effects and ensures smooth mark merging
- Should be < chunk_size
- Typical range: 2-5 segments

### Model Options
- `gpt-4o-mini`: Fast, cost-effective (recommended)
- `gpt-4o`: Higher quality, more expensive
- `gpt-4-turbo`: Balance of speed and quality

## Caching

Results are automatically cached in `.cache/llm_chunks/`.

To clear cache:
```bash
rm -rf .cache/llm_chunks/
```

## Comparison with Embedding Mode

| Feature | Embedding Mode | LLM-Direct Mode |
|---------|----------------|-----------------|
| Speed | Fast | Slower |
| Cost | Lower | Higher |
| Precision | Good | Excellent |
| Explanations | Keywords only | Full reasoning |
| Best for | Quick analysis | Detailed analysis |

## Troubleshooting

### API Key Error
Make sure your `.env` file contains a valid OpenAI API key.

### Rate Limit Error
Reduce chunk_size or add delays. The system will automatically retry with exponential backoff.

### Cache Issues
Clear the cache directory if you suspect stale results.

## Next Steps

- See `SPEC2.md` for detailed algorithm specification
- Compare results with embedding mode (port 8000, see QUICKSTART.md)
- Experiment with different chunk_size and overlap values
- Try different LLM models for quality/cost tradeoffs
- Clear cache if you update prompts or change PROMPT_VERSION
