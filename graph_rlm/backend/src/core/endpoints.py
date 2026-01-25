
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time
import asyncio
import uuid

from .db import db
from .llm import llm
from .agent import agent
from .config import settings

router = APIRouter()

# --- Data Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    session_id: Optional[str] = None

# --- Endpoints ---

@router.get("/system/models")
async def list_models(provider: Optional[str] = None):
    """
    List available models.
    """
    return llm.list_models(provider=provider)

@router.get("/system/config")
async def get_config():
    """
    Get current system configuration (safe subset).
    """
    # Expose configs so UI knows what's active
    # Pydantic .dict() or .model_dump()
    return {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "OLLAMA_BASE_URL": settings.OLLAMA_BASE_URL,
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
        "OLLAMA_EMBEDDING_MODEL": settings.OLLAMA_EMBEDDING_MODEL,
        "OPENROUTER_API_KEY": settings.OPENROUTER_API_KEY, # Return it so it populates UI (over OS SSL if needed, but this is local)
        "OPENROUTER_MODEL": settings.OPENROUTER_MODEL,
        "OPENROUTER_EMBEDDING_MODEL": settings.OPENROUTER_EMBEDDING_MODEL,
        # "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        # "OPENAI_MODEL": settings.OPENAI_MODEL,
        "provider": settings.LLM_PROVIDER # Alias for UI
    }

@router.post("/system/config")
async def update_config(request: Request):
    data = await request.json()
    # Validate allowed keys (security overlap)
    allowed_keys = {
        "LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_EMBEDDING_MODEL",
        "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_EMBEDDING_MODEL",
        # "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL"
    }

    updates = {}
    for k, v in data.items():
        if k in allowed_keys:
            updates[k] = str(v)

    if updates:
        # Save to .env and reload
        success = settings.save_to_env(updates)
        if not success:
             raise HTTPException(status_code=500, detail="Failed to persist config to .env")

    # Refresh llm service if needed (re-init singleton?)
    # ideally we re-init the llm client here if provider changed
    # For MVP, we might need to rely on next request reading settings,
    # but llm.py init happens once.
    # TODO: Trigger LLM re-init.
    # Quick fix: Modifying settings.py in-place works for future calls if LLMService reads on usage?
    # LLMService reads provider in __init__. So we need to re-init it.
    from .llm import llm
    llm.__init__()

    return {"status": "updated", "config": settings.get_llm_config()}

@router.get("/chat/sessions")
async def list_sessions():
    """
    List "Sessions".
    """
    try:
        q = """
        MATCH (t:Thought)
        WHERE NOT ()-[:DECOMPOSES_INTO]->(t)
        RETURN t.id AS id, t.prompt AS prompt, t.created_at AS created_at
        ORDER BY t.created_at DESC
        LIMIT 20
        """
        res = db.query(q)
        sessions = []
        for row in res:
            if isinstance(row, dict):
                sessions.append({
                    "id": row.get("id"),
                    "title": row.get("prompt")[:50] if row.get("prompt") else "Untitled Session",
                    "created_at": row.get("created_at")
                })
            elif isinstance(row, (list, tuple)) and len(row) >= 3:
                # Fallback for list/tuple results (id, prompt, created_at)
                prompt_text = row[1] if row[1] else "Untitled Session"
                sessions.append({
                    "id": row[0],
                    "title": prompt_text[:50],
                    "created_at": row[2]
                })
        return sessions
    except Exception as e:
        print(f"Session list error: {e}")
        return []

@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """
    Get message history for a session.
    """
    q = "MATCH (t:Thought {id: $id}) RETURN t.prompt AS prompt, t.result AS result"
    res = db.query(q, {"id": session_id})

    messages = []
    if res:
        row = res[0]
        p_text = row.get("prompt")
        r_text = row.get("result")
        if p_text:
            messages.append({"role": "user", "content": p_text})
        if r_text:
            messages.append({"role": "assistant", "content": r_text})

    return messages

@router.get("/chat/graph")
async def get_graph(session_id: Optional[str] = None):
    """
    Get entire graph state for visualization.
    Optionally filter by session_id.
    """
    try:
        # If session_id is provided, filter nodes
        if session_id:
            # We want all nodes in this session + their rels
            # Note: We assume edges don't cross sessions generally?
            # Or if they do, we want that context?
            # Safe bet: MATCH (n:Thought) WHERE n.session_id = $sid
            # We query on root_session_id OR session_id to catch everything relevant
            cypher = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $sid OR n.session_id = $sid
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
            """
            raw_data = db.query(cypher, {"sid": session_id})
        else:
            raw_data = db.get_graph_state()

        nodes = {}
        links = []

        for row in raw_data:
            # Row is [node, rel, target_node] or just [node, None, None]
            # FalkorDB python client returns Nodes/Relationships as objects or dicts depending on version
            # We assume dict-like or object with properties

            # Safe extraction
            if isinstance(row, (list, tuple)):
                source = row[0]
                rel = row[1] if len(row) > 1 else None
                target = row[2] if len(row) > 2 else None
            elif isinstance(row, dict):
                # FalkorDB wrapper might return keys 'n', 'r', 'm' based on query
                # Query was `RETURN n, r, m`
                source = row.get('n') or row.get('source')
                rel = row.get('r') or row.get('rel')
                target = row.get('m') or row.get('target')
            else:
                 # Fallback
                 source = row
                 rel = None
                 target = None

            # Helper to extract props
            def get_props(entity):
                if hasattr(entity, 'properties'): return entity.properties
                if isinstance(entity, dict): return entity
                # Falkor sometimes returns (id, ['Label'], {props}) tuple in old versions?
                # But simplified client wrapper usually returns objects.
                # Let's assume object with .id and .properties or dict
                return {}

            def get_id(entity):
                if hasattr(entity, 'id'): # internal ID
                     if 'id' in entity.properties: return entity.properties['id']
                if isinstance(entity, dict): return entity.get('id')
                return str(entity)

            # Process Source Node
            s_props = get_props(source)
            s_id = s_props.get('id')
            if s_id and s_id not in nodes:
                nodes[s_id] = {
                    "id": s_id,
                    "label": s_props.get('prompt', 'Unknown')[:30] + "...",
                    "group": 2 if 'DECOMPOSES_INTO' in str(rel) else 1, # heuristic
                    "val": 5,
                    "status": s_props.get('status', 'pending')
                }

            # Process Relationship
            if rel and target:
                t_props = get_props(target)
                t_id = t_props.get('id')

                if t_id:
                    # Upgrade coloring if we see relationships
                    if s_id in nodes: nodes[s_id]['group'] = 1 # Root-ish?
                    if t_id not in nodes:
                         nodes[t_id] = {
                            "id": t_id,
                            "label": t_props.get('prompt', 'Unknown')[:30] + "...",
                            "group": 2, # Child
                            "val": 3,
                            "status": t_props.get('status', 'pending')
                         }

                    link_id = f"{s_id}-{t_id}"
                    links.append({"source": s_id, "target": t_id})

        return {
            "nodes": list(nodes.values()),
            "links": links
        }
    except Exception as e:
        print(f"Graph fetch error: {e}")
        return {"nodes": [], "links": []}

@router.post("/chat/completions")
async def chat_completions(chat_req: ChatCompletionRequest, req: Request):
    """
    The main chat endpoint.
    Triggers Agent.query().
    """
    if not chat_req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_msg = chat_req.messages[-1]
    prompt = last_msg.content

    import logging
    logger = logging.getLogger("graph_rlm.endpoints")
    logger.info(f"Processing Prompt: {prompt}")

    async def response_stream():
        logger.info("DEBUG: Inside response_stream")
        # 1. Start Event
        yield f"data: {json.dumps({'type': 'thinking', 'data': 'Initializing agent recursion...'})}\n\n"

        # 2. Execute Stream (Yields real events from nested recursion)
        try:
            logger.info("DEBUG: Calling agent.stream_query")
            async for event in agent.stream_query(prompt, parent_id=None):
                if await req.is_disconnected():
                    logger.info("Client disconnected. Stopping agent.")
                    agent.stop_generation()
                    break

                logger.info(f"DEBUG: Yielding event {event.get('type')}")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"DEBUG: Exception in response_stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        # 3. Finish
        logger.info("DEBUG: response_stream finished")
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


# --- MCP Integration Endpoints ---

@router.get("/mcp/status")
async def mcp_status():
    """List detected MCP servers and tools (Optimized)."""
    from pathlib import Path
    import json
    import re
    import importlib

    # Resolve project root
    project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
    config_path = project_root / "mcp_servers.json"

    if not config_path.exists():
        return {"servers": [], "status": "not configured"}

    try:
        # 1. Read Config
        with open(config_path) as f:
            data = json.load(f)

        configured_servers = data.get("mcpServers", {}).keys()

        servers = []

        # 2. Inspect Generated Modules
        for name in configured_servers:
            # Sanitize name to find module
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            if not clean_name[0].isalpha() and clean_name[0] != '_':
                clean_name = '_' + clean_name
            module_name = clean_name.lower()

            error = None
            tools = []
            enabled = False

            try:
                # Dynamic Import
                mod = importlib.import_module(f"graph_rlm.backend.mcp_tools.{module_name}")
                if hasattr(mod, "list_tools"):
                    tools = mod.list_tools()
                    enabled = True
            except ImportError:
                # Module not generated yet or failed
                error = "Tool wrapper not found (may need restart)"
            except Exception as e:
                error = str(e)

            servers.append({
                "name": name,
                "enabled": enabled,
                "configured": True,
                "tools": tools,
                "error": error
            })

        return {"servers": servers}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/skills")
async def list_skills_endpoint():
    """List available skills from the library."""
    try:
        from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
        mgr = get_skills_manager()
        # Returns dict {name: metadata}
        skills_dict = mgr.list_skills()
        # Convert to list for UI
        skills_list = []
        for name, meta in skills_dict.items():
            skills_list.append({
                "name": name,
                "description": meta.get("description"),
                "tags": meta.get("tags", []),
                "version": meta.get("version")
            })
        return skills_list
    except Exception as e:
        print(f"Error listing skills: {e}")
        return []

