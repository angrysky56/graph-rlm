"""
FastAPI Endpoints for Graph-RLM.
Handles chat completions, session management, and system configuration.
"""

import asyncio
import importlib
import json
import os
import re
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import httpx
import redis
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph_rlm.backend.src.core.context_index import context_index
from graph_rlm.backend.src.mcp_integration.skill_storage import get_skills_manager

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
    """
    Representation of a single message in a chat conversation.
    """

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    Schema for an OpenAI-compatible chat completion request.
    """

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    session_id: Optional[str] = None
    metadata: Optional[dict] = None


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
        "SUMMARY_MODEL": settings.SUMMARY_MODEL,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "OPENAI_MODEL": settings.OPENAI_MODEL,
        "provider": settings.LLM_PROVIDER,  # Alias for UI
    }


@router.post("/system/config")
async def update_config(request: Request):
    """
    Update system configuration settings and persist them to the .env file.
    Only strictly allowed keys are updated for security.
    """
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
        "SUMMARY_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
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

    # Refresh llm service
    await llm.refresh()

    return {"status": "updated", "config": settings.get_llm_config()}


@router.post("/system/stop")
async def stop_generation():
    """
    Explicitly stop the agent generation loop.
    """
    try:
        agent.stop_generation()
        return {"status": "success", "message": "Stop signal sent to agent."}
    except (AttributeError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/system/shutdown")
async def shutdown_system():
    """
    Trigger a graceful shutdown of the backend.
    """

    def kill_server():
        time.sleep(1)  # Give time for the response to return
        os.kill(os.getpid(), signal.SIGTERM)

    # Schedule kill in a separate thread to allow response to be sent
    threading.Thread(target=kill_server).start()
    return {"status": "success", "message": "Backend shutting down..."}


@router.get("/system/status")
async def get_system_status(session_id: Optional[str] = None):
    """
    Get real-time agent status and scratchpad data.
    """
    try:
        # We really need the frontend to pass the active session ID
        if not session_id:
            return {"scratchpad": []}

        state = agent.get_state()
        scratchpad = context_index.get_active_scratchpad_data(session_id)

        return {
            "status": "active" if state.current_thought_id else "idle",
            "current_thought": state.current_thought_id,
            "scratchpad": scratchpad,
        }
    except (AttributeError, KeyError, ValueError) as e:
        logger.error("System status check failed: %s", e)
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
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
    ) as e:
        logger.error("Re-embedding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/chat/sessions")
async def list_sessions():
    """
    List "Sessions" grouped by root_session_id.
    """
    try:
        # Bundle thoughts by root_session_id to show unique conversations
        # We want to sort by LAST activity (max(created_at)),
        # but Keep Title from FIRST node (min(created_at))
        q = """
        MATCH (t:Thought)
        WITH COALESCE(t.root_session_id, t.session_id) AS sid,
             max(t.created_at) AS last_active
        ORDER BY last_active DESC
        LIMIT 20
        MATCH (root:Thought)
        WHERE (root.session_id = sid OR root.id = sid)
          AND root.status IN ['task', 'success', 'pending']
        WITH sid, last_active, root
        ORDER BY root.created_at ASC
        WITH sid, last_active, collect(root)[0] AS root_node
        RETURN sid AS session_id, root_node.prompt AS title, last_active
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
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
        KeyError,
    ) as e:
        logger.error("Session list error: %s", e)
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
    Get message history for a session.
    Prioritizes 'Round' nodes (stable turn-level summaries).
    Also fetches unbundled 'Thought' nodes for incomplete sessions.
    """
    messages = []

    # 1. Try to get completed Rounds
    q_rounds = """
    MATCH (r:Round)
    WHERE r.root_session_id = $id OR r.session_id = $id
    RETURN r.user_prompt AS prompt,
           r.final_response AS response,
           r.started_at AS started_at,
           r.ended_at AS ended_at,
           r.round_id AS id
    ORDER BY r.started_at ASC
    """
    res_rounds = db.query(q_rounds, {"id": session_id})

    if res_rounds:
        for row in res_rounds:
            # Add User Message
            messages.append(
                {
                    "role": "user",
                    "content": row.get("prompt", ""),
                    "created_at": row.get("started_at"),
                    "id": row.get("id", "") + ":user",
                }
            )
            # Add Assistant Message
            messages.append(
                {
                    "role": "assistant",
                    "content": row.get("response", ""),
                    "created_at": row.get("ended_at"),
                    "id": row.get("id", "") + ":assistant",
                    "status": "success",
                }
            )

    # 2. Fetch unbundled nodes (active/incomplete turns not yet in a Round)
    q_nodes = """
    MATCH (node)
    WHERE (node:Thought OR node:Insight OR node:Axiom)
    AND (node.root_session_id = $id OR node.session_id = $id)
    AND NOT (:Round)-[:CONTAINS]->(node)
    RETURN node.prompt AS prompt,
           node.content AS content,
           node.result AS result,
           node.created_at AS created_at,
           node.status AS status,
           node.id AS id,
           node.repl_id AS repl_id,
           node.execution_summary AS execution_summary,
           labels(node) AS labels
    ORDER BY node.created_at ASC
    """
    res_nodes = db.query(q_nodes, {"id": session_id})

    for row in res_nodes:
        # Field fallback: Insight/Axiom use 'content', Thought uses 'prompt'
        content = row.get("prompt") or row.get("content") or ""
        result = row.get("result", "")
        labels = row.get("labels", [])
        status = row.get("status")
        repl_id = row.get("repl_id")
        execution_summary = row.get("execution_summary")

        # --- Status-based role/type classification ---
        # System-level nodes: dreamer, reflexion, sheaf, meta-agent events
        system_statuses = {
            "system",
            "reflexion",
            "sheaf",
            "meta_agent",
            "dreamer_rejection",
            "dreamer_validation",
        }

        if status == "task":
            # User input
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    "created_at": row.get("created_at"),
                    "id": row.get("id"),
                }
            )
        elif not messages and "Thought" in labels:
            # First node in a legacy session is usually the user's first prompt
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    "created_at": row.get("created_at"),
                    "id": row.get("id"),
                }
            )
        elif status in system_statuses or repl_id in ("BRK", "SYS"):
            # System transparency events — dreamer, sheaf, reflexion, meta-agents
            # Detect subsystem tag from content prefix (e.g., "🔴 [Dreamer]")
            tag = "SYSTEM"
            if "[Dreamer]" in content or status == "dreamer_rejection":
                tag = "DREAMER"
            elif "[Reflexion]" in content or status == "reflexion":
                tag = "REFLEXION"
            elif "[Sheaf]" in content or status == "sheaf":
                tag = "SHEAF"
            elif "[SOAR]" in content or status == "meta_agent":
                tag = "META_AGENT"
            elif repl_id == "BRK":
                tag = "SYSTEM"

            messages.append(
                {
                    "role": "system",
                    "content": content,
                    "created_at": row.get("created_at"),
                    "id": row.get("id"),
                    "status": status,
                    "subsystem": tag,
                }
            )
        elif status in ("valid", "success"):
            # Validated/successful output
            final_text = content
            if result and str(result) != "None":
                final_text += f"\n\nResult: {result}"
            messages.append(
                {
                    "role": "assistant",
                    "content": final_text,
                    "created_at": row.get("created_at"),
                    "id": row.get("id"),
                    "status": "success",
                    "repl_id": repl_id,
                    "execution_summary": execution_summary,
                }
            )
        else:
            # Default assistant output (code execution, intermediate steps)
            final_text = content
            if result and str(result) != "None":
                final_text += f"\n\nResult: {result}"

            messages.append(
                {
                    "role": "assistant",
                    "content": final_text,
                    "created_at": row.get("created_at"),
                    "id": row.get("id"),
                    "status": status,
                    "repl_id": repl_id,
                    "execution_summary": execution_summary,
                }
            )

    return messages


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete an entire session and its history.
    """
    try:
        db.delete_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
    ) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/chat/thoughts/{thought_id}")
async def delete_thought(thought_id: str):
    """
    Delete a single thought node.
    """
    try:
        db.delete_thought_node(thought_id)
        return {"status": "success", "message": f"Thought {thought_id} deleted"}
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
    ) as e:
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
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
    ) as e:
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
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
    ) as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset DB: {e}") from e


@router.get("/system/eval")
async def get_eval_metrics():
    """
    Get agent evaluation metrics.
    Returns success/failure counts, step counts, and dreamer interventions.
    """
    total = agent.eval_success_count + agent.eval_failure_count
    success_rate = (agent.eval_success_count / total * 100) if total > 0 else 0.0

    return {
        "success_count": agent.eval_success_count,
        "failure_count": agent.eval_failure_count,
        "total_tasks": total,
        "success_rate": round(success_rate, 2),
        "dreamer_interventions": agent.eval_dreamer_interventions,
        "step_count": agent.eval_step_count,
    }


@router.get("/chat/graph")
async def get_graph(session_id: Optional[str] = None):
    """
    Get entire graph state for visualization.
    Optionally filter by session_id.
    """
    try:
        # Use a consistent ID filtering logic
        where_clause = "WHERE (n:Thought OR n:Insight OR n:Axiom OR n:Round)"
        params = {}
        if session_id:
            where_clause += " AND (n.root_session_id = $sid OR n.session_id = $sid)"
            params = {"sid": session_id}

        # 1. Fetch Nodes
        node_query = f"""
        MATCH (n)
        {where_clause}
        RETURN n
        """
        raw_nodes = db.query(node_query, params)

        # 2. Fetch Relationships (DIRECTED to preserve DAG causality)
        # Use directed pattern -> to maintain DECOMPOSES_INTO parent→child flow
        rel_query = f"""
        MATCH (n)-[r]->(m)
        {where_clause}
        AND (m:Thought OR m:Insight OR m:Axiom OR m:Round)
        RETURN n.id as sid, m.id as tid, type(r) as type
        """
        raw_rels = db.query(rel_query, params)

        nodes = {}
        links = []
        seen_links = set()

        # Helper to extract and format node properties
        def process_node(entity):
            if entity is None:
                return None
            props = entity.properties if hasattr(entity, "properties") else entity
            node_id = props.get("id")
            if not node_id:
                return None

            # Result fallback logic
            res = props.get("result", "")
            if not res or res == "None" or res == "":
                res = props.get("execution_summary", "") or props.get(
                    "final_response", ""
                )

            # label/prompt fallback
            lbl = (
                props.get("prompt")
                or props.get("content")
                or props.get("user_prompt")
                or "Unknown"
            )
            pmpt = (
                props.get("prompt")
                or props.get("content")
                or props.get("user_prompt")
                or ""
            )

            return {
                "id": node_id,
                "label": lbl,
                "prompt": pmpt,
                "result": res,
                "status": props.get("status", "pending"),
                "sheaf_score": props.get("sheaf_score"),
                "spectral_energy": props.get("spectral_energy"),
                "h0_rank": props.get("h0_rank"),
                "repe_shakiness": props.get("repe_shakiness"),
                "omcd_score": props.get("omcd_score"),
                "round_id": props.get("round_id"),
                "turn_id": props.get("turn_id"),
                "step_id": props.get("step_id"),
                "parent_id": props.get("parent_id"),
                "val": 5,
            }

        for row in raw_nodes:
            entity = row.get("n")
            data = process_node(entity)
            if data:
                nodes[data["id"]] = data

        for row in raw_rels:
            sid = row.get("sid")
            tid = row.get("tid")
            rtype = row.get("type")

            if sid and tid and sid in nodes and tid in nodes:
                # Directed deduplication: (source, target) is an ordered pair
                # A→B and B→A are preserved as distinct edges
                directed_pair = (sid, tid)
                if directed_pair not in seen_links:
                    links.append(
                        {
                            "source": sid,
                            "target": tid,
                            "type": rtype,
                            "directed": True,
                        }
                    )
                    seen_links.add(directed_pair)

        # Dynamic group assignment for coloring
        for node in nodes.values():
            if node.get("status") == "consolidated":
                node["group"] = 3
            elif node.get("status") in ["error", "failed"]:
                node["group"] = 4
            else:
                node["group"] = 1

        return {"nodes": list(nodes.values()), "links": links}

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to get graph: %s", e)
        logger.error("%s", traceback.format_exc())
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

    # Calculate turn_id properly based on completed rounds
    try:
        completed_rounds = db.get_completed_rounds(sid)
        turn_id = len(completed_rounds) + 1
    except Exception as e:
        logger.warning(f"Failed to calculate turn_id: {e}")
        turn_id = 1

    banner(f"SESSION START: {sid} | MODEL: {model_name} | TURN: {turn_id}")
    trace_action("API", "QUERY", result=prompt, tag="AGENT")

    logger.info("Processing Prompt: %s", prompt)

    async def response_stream():
        # 1. Start Event
        yield f"data: {json.dumps({'type': 'thinking', 'ui_target': 'TERMINAL_RAW', 'data': 'Initializing agent recursion...'})}\n\n"

        # 2. Execute Stream (Yields real events from nested recursion)
        try:
            # Use provided session_id or fallback to default
            sid = chat_req.session_id or "default"
            async for event in agent.stream_query(
                prompt,
                parent_id=None,
                session_id=sid,
                root_session_id=sid,
                turn_id=turn_id,
                metadata=chat_req.metadata,
            ):
                if await req.is_disconnected():
                    logger.info("Client disconnected. Stopping agent.")
                    agent.stop_generation()
                    break

                # Mirroring to terminal handled internally by Agent.emit_event
                # to avoid double-logging during streaming.
                yield f"data: {json.dumps(event)}\n\n"
        except (httpx.RequestError, RuntimeError, ValueError) as e:
            logger.error("Exception in response_stream: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        # 3. Finish
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


# --- MCP Integration Endpoints ---


@router.get("/mcp/status")
async def mcp_status():
    """List detected MCP servers and tools (Optimized)."""

    # Resolve project root
    project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()
    config_path = project_root / "mcp_servers.json"

    if not config_path.exists():
        return {"servers": [], "status": "not configured"}

    try:
        # 1. Read Config
        with open(config_path, encoding="utf-8") as f:
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
            except (AttributeError, ValueError) as e:
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
    except (
        redis.exceptions.RedisError,
        redis.exceptions.ResponseError,
        AttributeError,
        json.JSONDecodeError,
        FileNotFoundError,
    ) as e:
        return {"status": "error", "message": str(e)}


@router.get("/skills")
async def list_skills_endpoint():
    """List available skills from the library."""
    try:

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
    except (AttributeError, ValueError, KeyError, FileNotFoundError) as e:
        logger.error("Error listing skills: %s", e)
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
                except (
                    WebSocketDisconnect,
                    ConnectionError,
                    RuntimeError,
                    AttributeError,
                ):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        log_buffer.unsubscribe(on_log)
