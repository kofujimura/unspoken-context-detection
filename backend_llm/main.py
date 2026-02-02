"""
FastAPI application for LLM-direct UCC detection (SPEC2.md).
"""
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from backend_llm.core.models import LLMDetectionParams, LLMAnalyzeResponse
from backend_llm.llm import LLMCaller
from backend_llm.pipeline import LLMDirectPipeline

# Load environment
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Unspoken Context Detection API (LLM-Direct)",
    description="LLM-direct API for detecting implicit prerequisite shifts in lecture transcripts",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend_llm)
frontend_path = Path(__file__).parent.parent / "frontend_llm"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Initialize LLM caller (singleton)
llm_caller = LLMCaller(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    cache_dir=os.getenv("LLM_CACHE_DIR", ".cache/llm_chunks")
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve viewer HTML."""
    viewer_path = Path(__file__).parent.parent / "frontend_llm" / "viewer.html"
    if viewer_path.exists():
        return FileResponse(viewer_path)
    return HTMLResponse("<h1>UCC Detection API (LLM-Direct)</h1><p>Upload subtitle file to /api/analyze</p>")


@app.post("/api/analyze", response_model=LLMAnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    video_id: str = Form(...),
    language: str = Form("ja"),
    chunk_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    model: Optional[str] = Form(None)
):
    """
    Analyze subtitle file for UCC using LLM-direct mode.

    Args:
        file: Subtitle file (.vtt or .srt)
        video_id: Video identifier
        language: Language code
        chunk_size: Chunk size (optional)
        overlap: Chunk overlap (optional)
        model: LLM model name (optional)

    Returns:
        Analysis response with results
    """
    # Validate file format
    filename = file.filename.lower()
    if not (filename.endswith('.vtt') or filename.endswith('.srt')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use .vtt or .srt"
        )

    # Read content
    content = await file.read()
    content = content.decode('utf-8')

    # Determine format
    file_format = 'vtt' if filename.endswith('.vtt') else 'srt'

    # Create params
    params = LLMDetectionParams()
    if chunk_size is not None:
        params.chunk_size = chunk_size
    if overlap is not None:
        params.overlap = overlap
    if model is not None:
        params.model = model

    # Run pipeline
    pipeline = LLMDirectPipeline(llm_caller, params)
    response = await pipeline.analyze_content(
        content,
        file_format,
        video_id,
        language
    )

    return response


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "ucc-detection-llm", "mode": "llm_direct"}


@app.get("/api/params")
async def get_default_params():
    """Get default detection parameters."""
    return LLMDetectionParams()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
