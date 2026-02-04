import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph_rlm.backend.src.core.context_index import context_index

from .agent import agent
from .config import settings
from .db import db
from .llm import llm
from .log_stream import log_buffer
from .logger import get_logger
from .trace import banner, trace_action

logger = get_logger("graph_rlm.endpoints")

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
        "OPENROUTER_API_KEY": settings.OPENROUTER_API_KEY,  # Return it so it populates UI (over OS SSL if needed, but this is local)
        "OPENROUTER_MODEL": settings.OPENROUTER_MODEL,
        "OPENROUTER_EMBEDDING_MODEL": settings.OPENROUTER_EMBEDDING_MODEL,
        # "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        # "OPENAI_MODEL": settings.OPENAI_MODEL,
        "provider": settings.LLM_PROVIDER,  # Alias for UI
    }


@router.post("/system/config")
async def update_config(request: Request):
    data = await request.json()
    # Validate allowed keys (security overlap)
    allowed_keys = {
        "LLM_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "OPENROUTER_EMBEDDING_MODEL",
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
            raise HTTPException(
                status_code=500, detail="Failed to persist config to .env"
            )

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


@router.post("/system/stop")
async def stop_generation():
    """
    Explicitly stop the agent generation loop.
    """
    try:
        agent.stop_generation()
        return {"status": "success", "message": "Stop signal sent to agent."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/system/status")
async def get_system_status(session_id: Optional[str] = None):
    """
    Get real-time agent status and scratchpad data.
    """
    try:
        # We really need the frontend to pass the active session ID
        if not session_id:
            return {"scratchpad": []}

        state = agent._get_state()
        scratchpad = context_index.get_active_scratchpad_data(session_id)

        return {
            "status": "active" if state.current_thought_id else "idle",
            "current_thought": state.current_thought_id,
            "scratchpad": scratchpad,
        }
    except Exception as e:
        return {"scratchpad": [], "error": str(e)}


@router.post("/system/reembed")
async def reembed_graph():
    """
    Trigger a graph-wide re-embedding process using the current model.
    """
    try:
        # Use the global db and llm from the core
        count = db.reembed_all_thoughts(llm)
        return {
            "status": "success",
            "message": f"Successfully re-embedded {count} thoughts.",
            "count": count,
        }
    except Exception as e:
        logger.error(f"Re-embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chat/sessions")
async def list_sessions():
    """
    List "Sessions" grouped by root_session_id.
    """
    try:
        # Bundle thoughts by root_session_id to show unique conversations
        # We want to sort by LAST activity (max(created_at)), but Keep Title from FIRST node (min(created_at))
        q = """
        MATCH (t:Thought)
        WITH t.root_session_id AS session_id,
             min(t.created_at) AS started_at,
             max(t.created_at) AS last_active
        ORDER BY last_active DESC
        LIMIT 20
        MATCH (root:Thought {session_id: session_id})
        WITH session_id, last_active, root
        ORDER BY root.created_at ASC
        WITH session_id, last_active, collect(root)[0] AS root_node
        RETURN session_id, root_node.prompt AS title, last_active
        """
        res = db.query(q)

        sessions = []
        for row in res:
            sid = ""
            title = "Untitled Session"
            last_active = None

            if isinstance(row, dict):
                sid = row.get("session_id")
                title = row.get("title", "Untitled Session")
                last_active = row.get("last_active")
            elif isinstance(row, (list, tuple)) and len(row) >= 3:
                sid = row[0]
                title = row[1] if row[1] else "Untitled Session"
                last_active = row[2]

            if not sid:
                continue

            sessions.append(
                {
                    "id": sid,
                    "title": title if title else "Untitled Session",
                    "created_at": last_active,  # Use last_active for sorting in UI
                }
            )
        return sessions
    except Exception as e:
        print(f"Session list error: {e}")
        return []


@router.get("/sessions/{session_id}/thoughts")
async def get_session_thoughts(session_id: str):
    """
    Get all Thought nodes for a session, ordered chronologically.
    Used by the Scratchpad UI for full visibility.
    """
    return context_index.get_session_thoughts(session_id)


@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    """
    Get message history for a session by traversing the Thought Chain.
    Reconstructs the conversation from the Graph of Thoughts.
    """
    # Query: Match root and all subsequent thoughts in the chain
    # We use *0.. to include the root itself
    q = """
    MATCH (root:Thought {id: $id})
    OPTIONAL MATCH (root)-[:DECOMPOSES_INTO*]->(child:Thought)
    WITH root, child
    # If child is null (no chain), we just have root. If chain, we have pairs.
    # actually *0.. handles root as child.
    MATCH (n:Thought) WHERE n.id = root.id OR n.id = child.id
    RETURN DISTINCT n.prompt AS content, n.created_at AS created_at, n.id AS id
    ORDER BY n.created_at ASC
    """

    # Better Query for linear chain reconstruction:
    # We want to walk the tree.
    # Note: FalkorDB might not support full path traversal robustly in one simple return if branching exists.
    # But we enforced linear referencing in agent.py primarily.

    # Query: Match all thoughts belonging to this root session
    # Query: Match all thoughts belonging to this root session
    # We use a robust timestamp-based ordering, but ideally we should follow the DECOMPOSES_INTO chain if possible.
    # However, for the chat view, a chronological flat list of thoughts in the session is usually sufficient and more robust to graph fragments.
    q = """
    MATCH (node:Thought)
    WHERE node.root_session_id = $id OR node.session_id = $id
    RETURN node.prompt AS content,
           node.result AS result,
           node.created_at AS created_at,
           node.status AS status,
           node.id AS id,
           node.repl_id AS repl_id,
           node.execution_summary AS execution_summary
    ORDER BY node.created_at ASC
    """

    res = db.query(q, {"id": session_id})

    messages = []
    seen = set()

    for row in res:
        # Check format
        content = ""
        result = ""
        if isinstance(row, dict):
            content = row.get("content", "")
            result = row.get("result", "")
        elif isinstance(row, (list, tuple)):
            content = row[0]
            result = row[1] if len(row) > 1 else ""

        # Avoid duplicates just in case graph has cycles (shouldn't with DAG)
        if content in seen:
            continue
        seen.add(content)

        # Format:
        # The 'content' (node.prompt) in agent.py holds "Thought + [Output]" often.
        # But the User Prompt is only in the Root Node usually?
        # agent.py: create_thought_node(task_id, prompt, ...) -> Root
        # then create_thought_node(tid, full_content, ...) -> Thoughts

        # So:
        # 1. Root Node = User Prompt
        # 2. Subsequent Nodes = Assistant Thoughts/Actions

        if not messages:
            # First node is User
            messages.append({"role": "user", "content": content})
        else:
            # Subsequent nodes are Assistant
            # If content is empty but result exists?
            final_text = content
            if result:
                # If result is stored separately (old version)
                final_text += f"\n\nResult: {result}"

            messages.append({"role": "assistant", "content": final_text})

    return messages


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete an entire session and its history.
    """
    try:
        db.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/system/prune")
async def prune_orphans(hours: int = 1):
    """
    Prune orphaned thoughts older than N hours.
    """
    try:
        count = db.prune_orphans(older_than_hours=hours)
        return {
            "status": "success",
            "count": count,
            "message": f"Pruned {count} orphan nodes",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/system/reset")
async def reset_database():
    """
    Wipe the database. Used to clean up garbage nodes.
    """
    try:
        # Delete all nodes and relationships
        db.reset_graph()
        return {"status": "success", "message": "Database wiped."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset DB: {e}") from e


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
            if isinstance(row, dict):
                # FalkorDB wrapper might return keys 'n', 'r', 'm' based on query
                # Query was `RETURN n, r, m`
                source = row.get("n") or row.get("source")
                rel = row.get("r") or row.get("rel")
                target = row.get("m") or row.get("target")
            elif isinstance(row, (list, tuple)):
                source = row[0] if len(row) > 0 else None
                rel = row[1] if len(row) > 1 else None
                target = row[2] if len(row) > 2 else None
            else:
                # Fallback
                source = row
                rel = None
                target = None

            # Helper to extract props
            def get_props(entity):
                if hasattr(entity, "properties"):
                    return entity.properties
                if isinstance(entity, dict):
                    return entity
                # Falkor sometimes returns (id, ['Label'], {props}) tuple in old versions?
                # But simplified client wrapper usually returns objects.
                # Let's assume object with .id and .properties or dict
                return {}

            def get_id(entity):
                if hasattr(entity, "id"):  # internal ID
                    if "id" in entity.properties:
                        return entity.properties["id"]
                if isinstance(entity, dict):
                    return entity.get("id")
                return str(entity)

            # Process Source Node
            s_props = get_props(source)
            s_id = s_props.get("id")
            if s_id and s_id not in nodes:
                nodes[s_id] = {
                    "id": s_id,
                    "label": s_props.get("prompt", "Unknown"),
                    "prompt": s_props.get("prompt", ""),
                    "result": s_props.get("result", ""),
                    "group": 2 if "DECOMPOSES_INTO" in str(rel) else 1,  # heuristic
                    "val": 5,
                    "status": s_props.get("status", "pending"),
                }

            # Process Relationship
            if rel and target:
                t_props = get_props(target)
                t_id = t_props.get("id")

                if t_id:
                    # Upgrade coloring if we see relationships
                    if s_id in nodes:
                        nodes[s_id]["group"] = 1  # Root-ish?
                    if t_id not in nodes:
                        nodes[t_id] = {
                            "id": t_id,
                            "label": t_props.get("prompt", "Unknown"),
                            "group": 2,  # Child
                            "val": 3,
                            "status": t_props.get("status", "pending"),
                        }

                    links.append({"source": s_id, "target": t_id})

        return {"nodes": list(nodes.values()), "links": links}
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
    sid = chat_req.session_id or "default"
    model_name = llm.config.get("model")

    banner(f"SESSION START: {sid} | MODEL: {model_name}")
    trace_action("API", "QUERY", result=prompt, tag="AGENT")

    import logging

    logger = logging.getLogger("graph_rlm.endpoints")
    logger.info(f"Processing Prompt: {prompt}")

    async def response_stream():
        # 1. Start Event
        yield f"data: {json.dumps({'type': 'thinking', 'data': 'Initializing agent recursion...'})}\n\n"

        # 2. Execute Stream (Yields real events from nested recursion)
        try:
            # Use provided session_id or fallback to default
            sid = chat_req.session_id or "default"
            async for event in agent.stream_query(
                prompt, parent_id=None, session_id=sid
            ):
                if await req.is_disconnected():
                    logger.info("Client disconnected. Stopping agent.")
                    agent.stop_generation()
                    break

                # Mirror to terminal for observability
                e_type = event.get("type", "unknown")
                if e_type not in ["token", "usage"]:
                    trace_action(
                        "EVENT",
                        e_type.upper(),
                        result=event.get("content", ""),
                        tag="AGENT",
                    )

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Exception in response_stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        # 3. Finish
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


# --- MCP Integration Endpoints ---


@router.get("/mcp/status")
async def mcp_status():
    """List detected MCP servers and tools (Optimized)."""
    import importlib
    import json
    import re
    from pathlib import Path

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
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            if not clean_name[0].isalpha() and clean_name[0] != "_":
                clean_name = "_" + clean_name
            module_name = clean_name.lower()

            error = None
            tools = []
            enabled = False

            try:
                # Dynamic Import
                mod = importlib.import_module(
                    f"graph_rlm.backend.mcp_tools.{module_name}"
                )
                if hasattr(mod, "list_tools"):
                    tools = mod.list_tools()
                    enabled = True
            except ImportError:
                # Module not generated yet or failed
                error = "Tool wrapper not found (may need restart)"
            except Exception as e:
                error = str(e)

            servers.append(
                {
                    "name": name,
                    "enabled": enabled,
                    "configured": True,
                    "tools": tools,
                    "error": error,
                }
            )

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
            skills_list.append(
                {
                    "name": name,
                    "description": meta.get("description"),
                    "tags": meta.get("tags", []),
                    "version": meta.get("version"),
                }
            )
        return skills_list
    except Exception as e:
        print(f"Error listing skills: {e}")
        return []


# --- Log Stream Endpoint ---


@router.websocket("/ws/logs")
async def websocket_log_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming backend terminal logs.
    Sends log history on connect, then streams new logs in real-time.
    """
    await websocket.accept()

    # Create a queue for this connection
    log_queue: asyncio.Queue = asyncio.Queue()

    def on_log(message: str):
        """Callback when new log message arrives."""
        try:
            log_queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    # Subscribe to log updates
    log_buffer.subscribe(on_log)

    try:
        # Send log history first
        history = log_buffer.get_history()
        for msg in history:
            await websocket.send_text(msg)

        # Stream new logs
        while True:
            try:
                # Wait for new log message with timeout
                msg = await asyncio.wait_for(log_queue.get(), timeout=30.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_text("")
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        log_buffer.unsubscribe(on_log)
