import subprocess  # nosec
from pathlib import Path


def build_environment(mode: str = "full") -> str:
    """
    Bootstrap the MCP execution environment by building custom Docker images.

    This allows the agent to upgrade its own runtime capabilities.

    Args:
        mode: "full" (Python+Node+DataScience) or "lite" (Python+Node+MCP essentials)

    Returns:
        Log of build process and success message.
    """
    # Define Dockerfiles
    dockerfile_lite = """
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y curl git nodejs npm && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv smolagents mcp python-dotenv httpx anyio pydantic
WORKDIR /app
ENV PYTHONPATH=/app/src:/app
CMD ["python3"]
"""

    dockerfile_full = """
FROM python:3.12-slim-bookworm
# System Tools & Node.js
RUN apt-get update && apt-get install -y curl git build-essential nodejs npm && rm -rf /var/lib/apt/lists/*
# Python Essentials + Data Science + MCP
RUN pip install --no-cache-dir uv smolagents mcp python-dotenv httpx anyio pydantic \\
    numpy pandas sympy scipy requests beautifulsoup4
WORKDIR /app
ENV PYTHONPATH=/app/src:/app
CMD ["python3"]
"""

    # Select mode
    if mode == "lite":
        content = dockerfile_lite
        tag = "mcp-homebase:lite"
    else:
        content = dockerfile_full
        tag = "mcp-homebase:latest"

    # Write Dockerfile to temp location
    df_path = Path("Dockerfile.bootstrap")
    df_path.write_text(content)

    try:
        print(f"🏗️  Building Docker Image: {tag} ({mode} mode)...")
        # Run docker build
        subprocess.check_call(
            ["docker", "build", "-t", tag, "-f", "Dockerfile.bootstrap", "."]
        )  # nosec

        # Update .env if successful
        env_path = Path(".env")
        if env_path.exists():
            env_content = env_path.read_text()
            # Simple replace or append
            if "DOCKER_IMAGE=" in env_content:
                lines = env_content.splitlines()
                new_lines = []
                for line in lines:
                    if line.startswith("DOCKER_IMAGE="):
                        new_lines.append(f"DOCKER_IMAGE={tag}")
                    else:
                        new_lines.append(line)
                env_path.write_text("\n".join(new_lines))
            else:
                with env_path.open("a") as f:
                    f.write(f"\nDOCKER_IMAGE={tag}\n")

        return (
            f"✅ Successfully built {tag}. Please restart the server to apply changes."
        )

    except subprocess.CalledProcessError as e:
        return f"❌ Build failed: {e}"
    finally:
        if df_path.exists():
            df_path.unlink()
