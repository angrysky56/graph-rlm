from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables ASAP to ensure settings are correct
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent.parent.resolve()
load_dotenv(project_root / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP Tools on startup."""
    from graph_rlm.backend.src.mcp_integration.config import create_default_env_file
    from graph_rlm.backend.src.mcp_integration.discovery import discover_all_servers
    from graph_rlm.backend.src.mcp_integration.generator import ToolGenerator

    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        create_default_env_file(project_root)

        config_path = project_root / "mcp_servers.json"
        if config_path.exists():
            print(f"MCP: Discovering tools from {config_path}...")
            # Run discovery
            servers_info = await discover_all_servers(config_path)

            # Generate tools
            output_dir = Path(__file__).parent / "mcp_tools" # graph_rlm/backend/mcp_tools
            gen = ToolGenerator(output_dir)
            count = gen.generate_all(servers_info)
            print(f"MCP: Generated {count} server modules in {output_dir}")
        else:
            print("MCP: No mcp_servers.json found, skipping tool generation.")

    except Exception as e:
        print(f"MCP Initialization Failed: {e}")

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
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
