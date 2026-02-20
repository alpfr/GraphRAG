#!/bin/bash
set -e

echo "🚀 Starting OpenSearch-Docling-GraphRAG setup..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running or not installed."
    exit 1
fi

echo "📦 Starting Docker services (OpenSearch & Neo4j)..."
docker-compose up -d

echo "✅ Docker services rolling out! Waiting for them to become healthy..."

if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "📦 Installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

echo "🤖 Ensure Ollama is running and models are pulled!"
echo "Run: ollama pull ibm/granite4:latest"
echo "Run: ollama pull granite-embedding:278m"

echo ""
echo "🎉 Setup complete! Run the app with: streamlit run app.py"
