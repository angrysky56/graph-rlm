#!/bin/bash
set -e

echo "=== Graph RLM Environment Setup ==="

# 1. Check Python
echo "[+] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found."
    exit 1
fi

# 2. Check UV (Package Manager)
echo "[+] Checking uv..."
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 3. Backend Setup
# 3. Backend Setup
echo "[+] Setting up Backend (Root)..."

# Check for venv in root
if [ -d ".venv" ]; then
    echo "Found venv in root. Activating..."
    source .venv/bin/activate
else
    echo "Creating venv in root..."
    uv venv
    source .venv/bin/activate
fi

echo "Installing dependencies..."
uv sync
# Verify backend package is importable or just rely on path
# uv pip install -e . # logic removed as pyproject handles deps


# 4. Frontend Setup
echo "[+] Setting up Frontend..."
if ! command -v npm &> /dev/null; then
    echo "npm not found. Please install Node.js."
    exit 1
fi
cd graph_rlm/frontend
npm install
cd ..

# 5. Configuration Setup
echo "[+] Configuring Environment..."
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
fi

# Load current env
set -a
source .env
set +a

# 6. Check FalkorDB Connection (Optional)
echo "[+] Environment Setup Complete."
echo "Note: Ensure FalkorDB is running (e.g. via Docker) before starting the system."


echo "=== Setup Complete ==="
echo "Run ./start_system.sh to launch."
