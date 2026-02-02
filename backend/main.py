"""
FastAPI application for UCC detection.
"""
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

from backend.core.models import DetectionParams, AnalyzeResponse
from backend.embedding import OpenAIEmbedder
from backend.batch import UCCPipeline

# Load environment
load_dotenv()


# CSP Middleware to allow Chart.js and other scripts
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Allow Chart.js from CDN and inline scripts
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self' data:; "
            "img-src 'self' data:; "
            "connect-src 'self' https://cdn.jsdelivr.net;"
        )
        return response


# Initialize FastAPI
app = FastAPI(
    title="Unspoken Context Detection API",
    description="API for detecting implicit prerequisite shifts in lecture transcripts",
    version="1.0.0"
)

# Add CSP middleware
app.add_middleware(CSPMiddleware)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Initialize embedder (singleton)
embedder = OpenAIEmbedder(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    cache_dir=os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")
)


@app.get("/")
async def root():
    """Serve viewer HTML."""
    viewer_path = Path(__file__).parent.parent / "frontend" / "viewer.html"
    if viewer_path.exists():
        with open(viewer_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return HTMLResponse("<h1>UCC Detection API</h1><p>Upload subtitle file to /api/analyze</p>")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    video_id: str = Form(...),
    language: str = Form("ja"),
    k: Optional[int] = Form(None),
    k_minus: Optional[int] = Form(None),
    k_plus: Optional[int] = Form(None),
    lambda_param: Optional[float] = Form(None),
    tau_topic: Optional[float] = Form(None)
):
    """
    Analyze subtitle file for UCC.

    Args:
        file: Subtitle file (.vtt or .srt)
        video_id: Video identifier
        language: Language code
        k: Legacy window size (sets both k_minus and k_plus, optional)
        k_minus: Past window size (optional)
        k_plus: Current window size (optional)
        lambda_param: Threshold multiplier (optional)
        tau_topic: Topic continuity threshold (optional)

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
    params_dict = {}

    # Handle window size parameters
    # Priority: k_minus/k_plus override k
    # If both k_minus and k_plus are provided, use them; otherwise use k
    if k_minus is not None and k_plus is not None:
        # User specified both - use them directly
        params_dict['k_minus'] = k_minus
        params_dict['k_plus'] = k_plus
    elif k_minus is not None or k_plus is not None:
        # User specified only one - set that one, other uses default
        if k_minus is not None:
            params_dict['k_minus'] = k_minus
        if k_plus is not None:
            params_dict['k_plus'] = k_plus
    elif k is not None:
        # Legacy: user specified k - set it (validator will copy to k_minus/k_plus)
        params_dict['k'] = k

    # Other parameters
    if lambda_param is not None:
        params_dict['lambda_param'] = lambda_param
    if tau_topic is not None:
        params_dict['tau_topic'] = tau_topic

    # Create params object with all values at once
    params = DetectionParams(**params_dict)

    # Debug: log the params
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Created DetectionParams: k_minus={params.k_minus}, k_plus={params.k_plus}, k={params.k}")

    # Run pipeline
    pipeline = UCCPipeline(embedder, params)
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
    return {"status": "ok", "service": "ucc-detection"}


@app.get("/api/params")
async def get_default_params():
    """Get default detection parameters."""
    return DetectionParams()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
