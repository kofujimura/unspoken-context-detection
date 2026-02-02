#!/bin/bash
# Start LLM-direct UCC detection server (SPEC2.md)

echo "Starting LLM-Direct UCC Detection Server (port 8001)..."
echo "Open http://localhost:8001 in your browser"
echo ""
echo "Note: This mode uses LLM API calls and may take longer to process."
echo ""

uvicorn backend_llm.main:app --reload --port 8001
