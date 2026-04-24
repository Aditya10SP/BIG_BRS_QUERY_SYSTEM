#!/bin/bash

# Start Backend API for Graph RAG Query System

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  📚 Graph RAG Query System - Starting Backend API         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Load environment variables from .env file
if [ -f .env ]; then
    echo "✅ Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  Warning: .env file not found!"
    echo "   Creating from .env.example..."
    cp .env.example .env
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "🔍 Checking required services..."

# Check Ollama
if curl -s http://localhost:11434/ > /dev/null 2>&1; then
    echo "✅ Ollama is running at http://localhost:11434"
else
    echo "⚠️  Warning: Ollama not detected at http://localhost:11434"
    echo "   Please start Ollama: ollama serve"
fi

# Check Qdrant
if curl -s http://localhost:6333/ > /dev/null 2>&1; then
    echo "✅ Qdrant is running at http://localhost:6333"
else
    echo "⚠️  Warning: Qdrant not detected at http://localhost:6333"
fi

# Check Neo4j
if curl -s http://localhost:7474/ > /dev/null 2>&1; then
    echo "✅ Neo4j is running at http://localhost:7474"
else
    echo "⚠️  Warning: Neo4j not detected at http://localhost:7474"
fi

echo ""
echo "🚀 Starting backend API on http://localhost:8000..."
echo "   Press Ctrl+C to stop"
echo ""

# Start the API
python -m uvicorn src.main:app --reload --port 8000
