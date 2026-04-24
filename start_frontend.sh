#!/bin/bash

# Start Frontend Server for Graph RAG Query System

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║  📚 Graph RAG Query System - Starting Frontend            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if backend is running
echo "🔍 Checking if backend API is running..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Backend API is running at http://localhost:8000"
else
    echo "⚠️  Warning: Backend API not detected at http://localhost:8000"
    echo "   Please start the backend API first:"
    echo "   python -m uvicorn src.main:app --reload --port 8000"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🚀 Starting frontend server..."
echo ""

cd frontend && python server.py
