#!/bin/bash

# Configuration
REDIS_PORT=6380

echo "=== Starting Graph RLM System (PyQt Edition) ==="

# Trap functionality for cleanup
cleanup() {
	echo ""
	echo "=== Shutting Down ==="

	if [[ -n ${REDIS_PID} ]]; then
		if [[ ${REDIS_PID} == "DOCKER_CONTAINER" ]]; then
			echo "[-] Stopping Docker Container..."
			docker stop graph-rlm-db >/dev/null
		else
			echo "[-] Stopping Redis/FalkorDB (PID: ${REDIS_PID})..."
			kill "${REDIS_PID}" 2>/dev/null
		fi
	fi

    # Kill the Python process if it's still running (though closing the window usually does this)
    if [[ -n ${UI_PID} ]]; then
        kill "${UI_PID}" 2>/dev/null
    fi

	echo "=== Goodbye ==="
	exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# 1. Start Database
echo "[+] Launching Database on port ${REDIS_PORT}..."

if command -v docker &>/dev/null; then
	echo "    Checking for existing Database container..."
	existing_container=$(docker ps -aq -f name=graph-rlm-db)
	if [[ -n ${existing_container} ]]; then
		echo "    -> Found existing graph-rlm-db container. Starting/Reusing it."
		docker start graph-rlm-db >/dev/null 2>&1
		REDIS_PID="DOCKER_CONTAINER"
	else
		echo "    -> Launching new FalkorDB container..."
		docker rm -f graph-rlm-db >/dev/null 2>&1
		mkdir -p falkordb_data
		echo "    -> Launching new FalkorDB container with persistence..."
		docker run -d --name graph-rlm-db -p "${REDIS_PORT}":6379 -v "${PWD}"/falkordb_data:/data falkordb/falkordb falkordb-server --appendonly yes
		REDIS_PID="DOCKER_CONTAINER"
		echo "    -> Container started. Waiting 5s for initialization..."
		sleep 5
	fi
else
	echo "WARNING: Docker not found. Assuming local Redis/FalkorDB is already running on port ${REDIS_PORT}."
	REDIS_PID=""
fi

# Wait for Redis to be ready
echo "    ...verifying Database connectivity on port ${REDIS_PORT}..."
for _ in {1..10}; do
	if (echo >/dev/tcp/127.0.0.1/"${REDIS_PORT}") >/dev/null 2>&1; then
		echo "    -> Database is connecting."
		break
	fi
	echo -n "."
	sleep 1
done
echo ""

# 1.5 Setup Agent Venv (for skills)
AGENT_VENV="graph_rlm/backend/agent_venv"
if [[ ! -d ${AGENT_VENV} ]]; then
	echo "[+] Creating dedicated Agent Venv at ${AGENT_VENV}..."
	uv venv "${AGENT_VENV}"
	echo "    -> Environment created."
fi

# 2. Start PyQt UI
echo "[+] Launching Graph RLM UI..."

# Use 'uv run' to ensure dependencies from pyproject.toml are used
uv run python -m graph_rlm.ui.main &
UI_PID=$!

echo "=== System Operational ==="
echo "Application PID: ${UI_PID}"

# Wait for UI process to exit
wait ${UI_PID}
cleanup
