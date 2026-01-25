"""
Configuration management for MCP-Coordinator.

Handles loading MCP server configuration from:
1. Explicit path parameter
2. MCP_JSON environment variable
3. .env file in project root
4. ./mcp.json in project root (default)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ConfigManager:
    """
    Manages MCP server configuration with multiple loading strategies.

    Priority order:
    1. Explicit config_path parameter
    2. MCP_JSON environment variable
    3. mcp_servers.json in project root
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Initialize configuration manager.

        Args:
            project_root: Root directory of the project (defaults to cwd)
        """
        self.project_root = project_root or Path.cwd()

        # Load .env file from project root if it exists
        env_file = self.project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def get_config_path(self, explicit_path: str | Path | None = None) -> Path:
        """
        Resolve configuration file path using priority order.

        Args:
            explicit_path: Explicitly provided config path (highest priority)

        Returns:
            Resolved Path to configuration file

        Raises:
            FileNotFoundError: If no configuration file found
        """
        # 1. Explicit path parameter (highest priority)
        if explicit_path is not None:
            path = Path(explicit_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"Explicitly provided config file not found: {path}"
                )
            return path

        # 2. MCP_JSON or MCP_SERVERS_CONFIG environment variable
        mcp_json_env = os.getenv("MCP_JSON") or os.getenv("MCP_SERVERS_CONFIG")
        if mcp_json_env:
            path = Path(mcp_json_env).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(
                    f"Environment variable points to non-existent file: {path}"
                )
            return path

        # 3. mcp.json or mcp_servers.json in project root (default)
        for filename in ["mcp.json", "mcp_servers.json"]:
            default_path = self.project_root / filename
            if default_path.exists():
                return default_path

        # No configuration found
        raise FileNotFoundError(
            f"No MCP configuration found. Tried:\n  - MCP_JSON environment variable (not set)\n  - {default_path} (not found)\n\nPlease create mcp.json in your project root or set MCP_JSON environment variable."
        )

    def get_executor_type(self) -> str:
        """
        Get executor type from environment or use default.

        Returns:
            Executor type: 'local', 'docker', 'e2b', 'modal', or 'wasm'
        """
        return os.getenv("MCP_EXECUTOR_TYPE", "local")

    # Removed unused get_ollama_config

    def get_docker_config(self) -> dict[str, Any]:
        """
        Get Docker executor configuration from environment.

        Returns:
            Dictionary with Docker configuration
        """
        return {
            "image": os.getenv("DOCKER_IMAGE", "python:3.12-slim"),
            "mem_limit": os.getenv("DOCKER_MEM_LIMIT", "512m"),
            "cpu_quota": int(os.getenv("DOCKER_CPU_QUOTA", "50000")),
        }

    def get_port_config(self) -> dict[str, int]:
        """
        Get port allocation configuration from environment.

        Returns:
            Dictionary with port range configuration
        """
        return {
            "port_range_start": int(os.getenv("MCP_PORT_RANGE_START", "49152")),
            "port_range_end": int(os.getenv("MCP_PORT_RANGE_END", "65535")),
        }

    def get_timeouts(self) -> dict[str, float]:
        """
        Get timeout configuration.

        Returns:
            Dictionary with timeout values in seconds
        """
        return {
            "connect": float(os.getenv("MCP_CONNECT_TIMEOUT", "45.0")),
            "read": float(os.getenv("MCP_READ_TIMEOUT", "60.0")),
            "discovery": float(os.getenv("MCP_DISCOVERY_TIMEOUT", "45.0")),
        }


def create_default_env_file(project_root: Path) -> None:
    """
    Create a default .env file with example configuration.

    Args:
        project_root: Root directory of the project
    """
    env_file = project_root / ".env"

    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file}")
        return

    env_content = f"""# MCP Coordinator Configuration
# Generated on {datetime.now().isoformat()}

# --- Core Configuration ---
# 1. Output Buffering (CRITICAL)
# Ensure stdout is flushed immediately for real-time agent logs
PYTHONUNBUFFERED=1

# 2. Model Selection
# =============================================================================
# Settings
# =============================================================================


# Timeouts (seconds)
MCP_CONNECT_TIMEOUT=30.0   # Connection establishment
MCP_READ_TIMEOUT=600.0     # Tool execution / read timeout (10 minutes)
MCP_DISCOVERY_TIMEOUT=60.0 # Server discovery timeout

# Enable network isolation (Linux only)
# NETWORK_ISOLATION=false

# Workspace directory
# Used for temporary file operations during agent execution
MCP_WORKSPACE_DIR=./workspace

# Tools output directory
# TOOLS_OUTPUT_DIR=./mcp_tools


# Knowledge Base Configuration
# Used for persistent memory (skills, graphs, vectors)
MCP_KNOWLEDGE_BASE_DIR=./knowledge_base
MCP_CHROMA_PATH=./knowledge_base/chroma

MAX_TOOL_OUTPUT_CHARS=100000
# Allow the agent to import ANY module (removes whitelisting security)
MCP_ALLOW_ALL_IMPORTS=false

# Path to MCP server configuration file
# Default: ./mcp.json in project root
# Priority: Explicit path > MCP_JSON env > ./mcp_servers.json
# MCP_JSON=/path/to/your/mcp_servers.json

# Executor type: 'local', 'docker', 'e2b', 'modal', or 'wasm'
# Recommended: docker (best security/performance balance)
# Default: docker
MCP_EXECUTOR_TYPE=docker

# =============================================================================
# Docker Executor Configuration (when MCP_EXECUTOR_TYPE=docker)
# =============================================================================

# Docker image for execution
# Recommended: mcp-homebase:latest (Custom image with Python+Node+Tools)
# Requires running build_container.sh to bootstrap from the default Home Base Lite.
# **Ask your AI to do it for you!** Say:
# "Ask the MCP Coordinator to build Home Base with the bootstrap_environment skill."

# It may take 5 to 10 minutes to build. Then set it here:
DOCKER_IMAGE=python:3.12-slim
# for Limited resources keep: DOCKER_IMAGE=python:3.12-slim

# Memory limit for Docker container
DOCKER_MEM_LIMIT=512m

# CPU quota (50000 = 50% of one core)
DOCKER_CPU_QUOTA=50000

# =============================================================================
# Port Allocation (for sub-servers spawned by executed code)
# =============================================================================

# Number of ports to pre-allocate for container use (default: 5)
# These ports are available via MCP_ALLOCATED_PORTS env var inside container
# MCP_PREALLOCATE_PORTS=5

# Port range for dynamic allocation (default: IANA dynamic ports)
# MCP_PORT_RANGE_START=49152
# MCP_PORT_RANGE_END=65535


# =============================================================================
# Pick a provider for your Coordinator Agent Configuration (Required)
# =============================================================================
# (OpenRouter/OpenAI)- Recommended: OpenRouter

# OpenRouter API Key (Compatible with OpenAI SDK)
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx

# Base URL for OpenRouter (default: https://openrouter.ai/api/v1)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Model to use for internal Coordinator Agents
# Recommended: google/gemini-2.0-flash-exp:free (Rate Limited), x-ai/grok-4.1-fast
OPENROUTER_MODEL=google/gemini-2.0-flash-exp:free

# Set your own API endpoint and use your own:

# OpenAI API key (for OpenAI models via LiteLLM)
# OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

# Anthropic API key (for Claude models via LiteLLM)
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx

# =============================================================================
# Ollama Configuration (Cheapest Local or Cloud Option)
# Set your own API endpoint and use your your own local models
# i.e. LM Studio http://localhost:1234

# Ollama API endpoint
# OLLAMA_API_BASE=http://localhost:11434

# Ollama model to use
# Local models: llama3.2:3b, qwen2.5-coder:7b, etc.
# Cloud models (requires auth): qwen3-coder:480b-cloud, gpt-oss:120b-cloud
# OLLAMA_MODEL=qwen3:latest


# =============================================================================
# Cloud Services (Optional)

# HuggingFace token for cloud models
# Required for: InferenceClientModel, hosted models
# Get token: https://huggingface.co/settings/tokens
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# E2B API key (when MCP_EXECUTOR_TYPE=e2b)
# Get key: https://e2b.dev/
# E2B_API_KEY=your_e2b_key_here

# Modal API credentials (when MCP_EXECUTOR_TYPE=modal)
# Get token: https://modal.com/settings
# MODAL_TOKEN_ID=your_token_id
# MODAL_TOKEN_SECRET=your_token_secret
"""

    env_file.write_text(env_content)
    print(f"✓ Created default .env file at {env_file}")
    print("  Edit this file to configure MCP-Coordinator for your setup")
