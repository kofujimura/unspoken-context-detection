#!/bin/bash
# Start embedding-based UCC detection server (SPEC.md)

echo "Starting Embedding-based UCC Detection Server (port 8000)..."
echo "Open http://localhost:8000 in your browser"
echo ""

uvicorn backend.main:app --reload --port 8000
