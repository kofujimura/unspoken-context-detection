#!/bin/bash
# Setup script for UCC Detection System

set -e

echo "==================================="
echo "UCC Detection System Setup"
echo "==================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# Setup environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OPENAI_API_KEY"
    echo ""
else
    echo ".env file already exists"
fi

# Create cache directory
mkdir -p .cache/embeddings

# Create output directory
mkdir -p output

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the server: python -m uvicorn backend.main:app --reload"
echo "4. Or use the CLI: python cli.py analyze --help"
echo ""
echo "Example CLI usage:"
echo "  python cli.py analyze --input examples/sample.vtt --video-id sample_lecture"
echo ""
