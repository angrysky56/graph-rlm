import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables ASAP to ensure settings are correct
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.endpoints import router as api_router

project_root = Path(__file__).parent.parent.parent.resolve()
load_dotenv(project_root / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for Graph-RLM Backend."""
    from graph_rlm.backend.src.core.db import db
    from graph_rlm.backend.src.mcp_integration.config import create_default_env_file
    from graph_rlm.backend.src.mcp_integration.discovery import discover_all_servers
    from graph_rlm.backend.src.mcp_integration.generator import ToolGenerator

    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        create_default_env_file(project_root)

        # Initialize Database Indexes
        db.create_vector_indexes()

        # Arm RepE Safety Monitor (Async Calibration)
        from graph_rlm.backend.src.core.repe import repe

        await repe.calibrate()

        # Initialize Log Streaming (captures all backend output for frontend)
        from graph_rlm.backend.src.core.log_stream import setup_log_streaming

        setup_log_streaming()
        print("Log streaming initialized for frontend.")

        # --- SKILLS & AXIOMS SYNC ---
        from graph_rlm.backend.src.mcp_integration.skills import (
            get_axioms_manager,
            get_skills_manager,
        )

        skills_mgr = get_skills_manager()
        axioms_mgr = get_axioms_manager()
        await skills_mgr.sync_from_disk()
        await axioms_mgr.sync_from_disk()

        config_path = project_root / "mcp_servers.json"
        if config_path.exists():
            output_dir = Path(__file__).parent / "mcp_tools"

            output_dir = Path(__file__).parent / "mcp_tools"

            should_regenerate = True
            if output_dir.exists():
                # Load config to check if any servers are missing files
                try:
                    import json

                    with open(config_path) as f:
                        config_data = json.load(f)
                    config_servers = config_data.get("mcpServers", {})

                    # Map to snake_case filenames
                    gen = ToolGenerator(output_dir)
                    expected_modules = {
                        gen._sanitize_name(name) for name in config_servers
                    }
                    existing_modules = {
                        f.stem
                        for f in output_dir.glob("*.py")
                        if f.stem not in ["__init__", "skills"]
                    }

                    missing = expected_modules - existing_modules

                    # If everything is there and the dir is newer than config, we can skip
                    if (
                        not missing
                        and output_dir.stat().st_mtime > config_path.stat().st_mtime
                    ):
                        if any(output_dir.iterdir()):
                            # Count cached servers and tools for logging
                            server_files = list(output_dir.glob("*.py"))
                            server_count = 0
                            tool_count = 0
                            for f in server_files:
                                if f.stem in ["__init__", "skills"]:
                                    continue
                                server_count += 1
                                # Quick count of tools
                                try:
                                    content = f.read_text()
                                    match = re.search(r"return\s+\[(.*?)\]", content)
                                    if match:
                                        tools = [
                                            t.strip().strip("'").strip('"')
                                            for t in match.group(1).split(",")
                                            if t.strip()
                                        ]
                                        tool_count += len(tools)
                                except Exception:
                                    pass

                            print(
                                f"MCP: Cached - Found {server_count} servers and {tool_count} tools in {output_dir.name}/"
                            )
                            should_regenerate = False
                except Exception as e:
                    logging.getLogger(__name__).error(
                        f"Failed to verify MCP cache consistency: {e}"
                    )
                    should_regenerate = True

            if should_regenerate:
                print(f"MCP: Discovering tools from {config_path}...")
                servers_info = await discover_all_servers(config_path)
                gen = ToolGenerator(output_dir)
                count, t_count = gen.generate_all(servers_info)
                print(
                    f"MCP: Generated {count} server modules with {t_count} tools in {output_dir}"
                )
        else:
            print("MCP: No mcp_servers.json found, skipping tool generation.")

    except Exception as e:
        print(f"MCP Initialization Failed: {e}")

    # --- STARTUP COMPLETE ---
    yield
    # --- SHUTDOWN STARTING ---

    print("\n[-] Shutting down Graph-RLM Backend...")
    try:
        from graph_rlm.backend.src.core.agent import agent

        agent.stop_generation()
        print("    -> Agent told to stop.")
    except Exception as e:
        print(f"Cleanup Failed: {e}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "Graph-RLM Backend is Running", "docs": "/docs"}
