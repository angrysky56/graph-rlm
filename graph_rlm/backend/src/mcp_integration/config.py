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

    def get_timeouts(self) -> dict[str, float]:
        """
        Get timeout configuration.

        Returns:
            Dictionary with timeout values in seconds
        """
        return {
            "connect": float(os.getenv("MCP_CONNECT_TIMEOUT", "60.0")),
            "read": float(os.getenv("MCP_READ_TIMEOUT", "300.0")),
            "discovery": float(os.getenv("MCP_DISCOVERY_TIMEOUT", "300.0")),
        }


def create_default_env_file(project_root: Path) -> None:
    """
    Create a default .env file with values matching core.config.Settings.
    """
    env_file = project_root / ".env"

    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file}")
        return

    env_content = f"""# Graph-RLM Configuration
# Generated on {datetime.now().isoformat()}

# --- Core ---
PROJECT_NAME=Graph-RLM
# GRAPH_NAME=rlm_graph

# --- Database ---
FALKOR_HOST=localhost
FALKOR_PORT=6380
# FALKORDB_PATH=./falkordb_data # Optional, for local persistent store

# --- LLM Provider Selection ---
# Options: ollama, openrouter, lmstudio, openai
LLM_PROVIDER=ollama

# --- Ollama Settings ---
OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=gemma3:latest
# OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# --- OpenRouter Settings ---
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
# OPENROUTER_MODEL=google/gemini-3-flash-preview
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# --- OpenAI Settings ---
# OPENAI_API_KEY=sk-xxxx
# OPENAI_MODEL=gpt-4o-mini

# --- MCP Tool Settings ---
# MCP_JSON=./mcp_servers.json
"""

    env_file.write_text(env_content)
    print(f"✓ Created default .env file at {env_file}")
