#!/bin/bash

# Configuration

REDIS_PORT=6380
API_PORT=8000

echo "=== Starting Graph RLM System ==="

# Trap functionality for cleanup
cleanup() {
    echo ""
    echo "=== Shutting Down ==="

    if [ ! -z "$API_PID" ]; then
        echo "[-] Stopping Backend API (PID: $API_PID)..."
        kill $API_PID 2>/dev/null
    fi

    if [ ! -z "$REDIS_PID" ]; then
        if [ "$REDIS_PID" == "DOCKER_CONTAINER" ]; then
             echo "[-] Stopping Docker Container..."
             docker stop graph-rlm-db >/dev/null
        else
             echo "[-] Stopping Redis/FalkorDB (PID: $REDIS_PID)..."
             kill $REDIS_PID 2>/dev/null
        fi
    fi

    if [ ! -z "$FRONTEND_PID" ]; then
        echo "[-] Stopping Frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi

    echo "=== Goodbye ==="
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# 1. Start Database
# Load environment (DISABLED: Let Python Backend handle .env reloading natively)
# if [ -f .env ]; then
#     set -a
#     source .env
#     set +a
# fi

echo "[+] Launching Database on port $REDIS_PORT..."

if command -v docker &> /dev/null; then
    echo "    Checking for existing Database container..."
    if [ "$(docker ps -q -f name=graph-rlm-db)" ]; then
        echo "    -> Found active graph-rlm-db container. Reusing it."
        REDIS_PID="DOCKER_CONTAINER"
    else
        echo "    -> Launching new FalkorDB container..."
        # Remove dead container if exists
        docker rm -f graph-rlm-db >/dev/null 2>&1
        # Create data directory if it doesn't exist
        mkdir -p falkordb_data
        echo "    -> Launching new FalkorDB container with persistence..."
        docker run -d --name graph-rlm-db -p $REDIS_PORT:6379 -v $(pwd)/falkordb_data:/data falkordb/falkordb
        REDIS_PID="DOCKER_CONTAINER"
        echo "    -> Container started. Waiting 5s for initialization..."
        sleep 5
    fi
else
    echo "WARNING: Docker not found. Assuming local Redis/FalkorDB is already running on port $REDIS_PORT."
    REDIS_PID=""
fi

# Wait for Redis to be ready (Double Check)
echo "    ...verifying Database connectivity on port $REDIS_PORT..."
for i in {1..10}; do
    if (echo > /dev/tcp/127.0.0.1/$REDIS_PORT) >/dev/null 2>&1; then
        echo "    -> Database is connecting."
        break
    fi
    echo -n "."
    sleep 1
done
echo ""


# 1.5 Setup Agent Venv
AGENT_VENV="graph_rlm/backend/agent_venv"
if [ ! -d "$AGENT_VENV" ]; then
    echo "[+] Creating dedicated Agent Venv at $AGENT_VENV..."
    uv venv "$AGENT_VENV"
    echo "    -> Environment created."
fi

# 2. Start Backend API
# Note: Assuming using uvicorn directly or via module
echo "[+] Launching Backend API on port $API_PORT..."
# Determine python interpreter from venv
if [ -f ".venv/bin/python" ]; then
    PY_CMD=".venv/bin/python"
else
    PY_CMD="python3"
fi

echo "[+] Launching Backend API (Module Mode)..."
$PY_CMD -m uvicorn graph_rlm.backend.main:app --port $API_PORT --reload &
API_PID=$!
echo "    -> API PID: $API_PID"

# 3. Start Frontend
echo "[+] Launching Frontend..."
cd graph_rlm/frontend/ui
npm run dev -- --host &
FRONTEND_PID=$!
echo "    -> Frontend PID: $FRONTEND_PID"
cd ../..

echo "=== System Operational ==="
echo "Press Ctrl+C to stop all services."

# Wait indefinitely so trap can catch signals
wait
