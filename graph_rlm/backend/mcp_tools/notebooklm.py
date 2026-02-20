"""
Auto-generated wrapper for notebooklm MCP server.

This module provides Python function wrappers for all tools
exposed by the notebooklm server.

Do not edit manually.
"""

from typing import Any


def ask_question(question: str | Any = None, session_id: str | None = None, notebook_id: str | None = None, notebook_url: str | None = None, show_browser: bool | None = None, browser_options: dict[str, Any] | None = None, **kwargs) -> Any:
    """# Conversational Research Partner (NotebookLM • Gemini 2.5 • Session RAG)

**Active Notebook:** Enterprise MCP & Docker Orchestration Guide
**Content:** A comprehensive collection of 30+ documents covering MCP Gateways, Docker container lifecycles, enterprise security hardening, and deployment strategies (IBM ContextForge, Peta MCP).
**Topics:** MCP Gateways, Docker Orchestration, Enterprise Security, Zero-Trust Architecture, Kubernetes Deployment

> Auth tip: If login is required, use the prompt 'notebooklm.auth-setup' and then verify with the 'get_health' tool. If authentication later fails (e.g., expired cookies), use the prompt 'notebooklm.auth-repair'.

## What This Tool Is
- Full conversational research with Gemini (LLM) grounded on your notebook sources
- Session-based: each follow-up uses prior context for deeper, more precise answers
- Source-cited responses designed to minimize hallucinations

## When To Use
  - Architecting enterprise AI agent infrastructure
  - Implementing secure MCP gateways
  - Managing containerized server lifecycles

## Rules (Important)
- Always prefer continuing an existing session for the same task
- If you start a new thread, create a new session and keep its session_id
- Ask clarifying questions before implementing; do not guess missing details
- If multiple notebooks could apply, propose the top 1–2 and ask which to use
- If task context changes, ask to reset the session or switch notebooks
- If authentication fails, use the prompts 'notebooklm.auth-repair' (or 'notebooklm.auth-setup') and verify with 'get_health'
- After every NotebookLM answer: pause, compare with the user's goal, and only respond if you are 100% sure the information is complete. Otherwise, plan the next NotebookLM question in the same session.

## Session Flow (Recommended)
```javascript
// 1) Start broad (no session_id → creates one)
ask_question({ question: "Give me an overview of [topic]" })
// ← Save: result.session_id

// 2) Go specific (same session)
ask_question({ question: "Key APIs/methods?", session_id })

// 3) Cover pitfalls (same session)
ask_question({ question: "Common edge cases + gotchas?", session_id })

// 4) Ask for production example (same session)
ask_question({ question: "Show a production-ready example", session_id })
```

## Automatic Multi-Pass Strategy (Host-driven)
- Simple prompts return once-and-done answers.
- For complex prompts, the host should issue follow-up calls:
  1. Implementation plan (APIs, dependencies, configuration, authentication).
  2. Pitfalls, gaps, validation steps, missing prerequisites.
- Keep the same session_id for all follow-ups, review NotebookLM's answer, and ask more questions until the problem is fully resolved.
- Before replying to the user, double-check: do you truly have everything? If not, queue another ask_question immediately.

## 🔥 REAL EXAMPLE

Task: "Implement error handling in n8n workflow"

Bad (shallow):
```
Q: "How do I handle errors in n8n?"
A: [basic answer]
→ Implement → Probably missing edge cases!
```

Good (deep):
```
Q1: "What are n8n's error handling mechanisms?" (session created)
A1: [Overview of error handling]

Q2: "What's the recommended pattern for API errors?" (same session)
A2: [Specific patterns, uses context from Q1]

Q3: "How do I handle retry logic and timeouts?" (same session)
A3: [Detailed approach, builds on Q1+Q2]

Q4: "Show me a production example with all these patterns" (same session)
A4: [Complete example with full context]

→ NOW implement with confidence!
```
    
## Notebook Selection
- Default: active notebook (enterprise-mcp-docker-orchestr)
- Or set notebook_id to use a library notebook
- Or set notebook_url for ad-hoc notebooks (not in library)
- If ambiguous which notebook fits, ASK the user which to use

    Args:
        question: The question to ask NotebookLM
        session_id: Optional session ID for contextual conversations. If omitted, a new session is created.
        notebook_id: Optional notebook ID from your library. If omitted, uses the active notebook. Use list_notebooks to see available notebooks.
        notebook_url: Optional notebook URL (overrides notebook_id). Use this for ad-hoc queries to notebooks not in your library.
        show_browser: Show browser window for debugging (simple version). For advanced control (typing speed, stealth, etc.), use browser_options instead.
        browser_options: Optional browser behavior settings. Claude can control everything: visibility, typing speed, stealth mode, timeouts. Useful for debugging or fine-tuning.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if question is not None:
        mcp_args["question"] = question
    if session_id is not None:
        mcp_args["session_id"] = session_id
    if notebook_id is not None:
        mcp_args["notebook_id"] = notebook_id
    if notebook_url is not None:
        mcp_args["notebook_url"] = notebook_url
    if show_browser is not None:
        mcp_args["show_browser"] = show_browser
    if browser_options is not None:
        mcp_args["browser_options"] = browser_options

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="ask_question",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def add_notebook(url: str | Any = None, name: str | Any = None, description: str | Any = None, topics: list[str] | Any = None, content_types: list[str] | None = None, use_cases: list[str] | None = None, tags: list[str] | None = None, **kwargs) -> Any:
    """PERMISSION REQUIRED — Only when user explicitly asks to add a notebook.

## Conversation Workflow (Mandatory)
When the user says: "I have a NotebookLM with X"

1) Ask URL: "What is the NotebookLM URL?"
2) Ask content: "What knowledge is inside?" (1–2 sentences)
3) Ask topics: "Which topics does it cover?" (3–5)
4) Ask use cases: "When should we consult it?"
5) Propose metadata and confirm:
   - Name: [suggested]
   - Description: [from user]
   - Topics: [list]
   - Use cases: [list]
   "Add it to your library now?"
6) Only after explicit "Yes" → call this tool

## Rules
- Do not add without user permission
- Do not guess metadata — ask concisely
- Confirm summary before calling the tool

## Example
User: "I have a notebook with n8n docs"
You: Ask URL → content → topics → use cases; propose summary
User: "Yes"
You: Call add_notebook

## How to Get a NotebookLM Share Link

Visit https://notebooklm.google/ → Login (free: 100 notebooks, 50 sources each, 500k words, 50 daily queries)
1) Click "+ New" (top right) → Upload sources (docs, knowledge)
2) Click "Share" (top right) → Select "Anyone with the link"
3) Click "Copy link" (bottom left) → Give this link to Claude

(Upgraded: Google AI Pro/Ultra gives 5x higher limits)

    Args:
        url: The NotebookLM notebook URL
        name: Display name for the notebook (e.g., 'n8n Documentation')
        description: What knowledge/content is in this notebook
        topics: Topics covered in this notebook
        content_types: Types of content (e.g., ['documentation', 'examples', 'best practices'])
        use_cases: When should Claude use this notebook (e.g., ['Implementing n8n workflows'])
        tags: Optional tags for organization

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if url is not None:
        mcp_args["url"] = url
    if name is not None:
        mcp_args["name"] = name
    if description is not None:
        mcp_args["description"] = description
    if topics is not None:
        mcp_args["topics"] = topics
    if content_types is not None:
        mcp_args["content_types"] = content_types
    if use_cases is not None:
        mcp_args["use_cases"] = use_cases
    if tags is not None:
        mcp_args["tags"] = tags

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="add_notebook",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def list_notebooks(**kwargs) -> Any:
    """List all library notebooks with metadata (name, topics, use cases, URL). Use this to present options, then ask which notebook to use for the task.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="list_notebooks",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def get_notebook(id: str | Any = None, **kwargs) -> Any:
    """Get detailed information about a specific notebook by ID

    Args:
        id: The notebook ID

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="get_notebook",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def select_notebook(id: str | Any = None, **kwargs) -> Any:
    """Set a notebook as the active default (used when ask_question has no notebook_id).

## When To Use
- User switches context: "Let's work on React now"
- User asks explicitly to activate a notebook
- Obvious task change requires another notebook

## Auto-Switching
- Safe to auto-switch if the context is clear and you announce it:
  "Switching to React notebook for this task..."
- If ambiguous, ask: "Switch to [notebook] for this task?"

## Example
User: "Now let's build the React frontend"
You: "Switching to React notebook..." (call select_notebook)

    Args:
        id: The notebook ID to activate

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="select_notebook",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def update_notebook(id: str | Any = None, name: str | None = None, description: str | None = None, topics: list[str] | None = None, content_types: list[str] | None = None, use_cases: list[str] | None = None, tags: list[str] | None = None, url: str | None = None, **kwargs) -> Any:
    """Update notebook metadata based on user intent.

## Pattern
1) Identify target notebook and fields (topics, description, use_cases, tags, url)
2) Propose the exact change back to the user
3) After explicit confirmation, call this tool

## Examples
- User: "React notebook also covers Next.js 14"
  You: "Add 'Next.js 14' to topics for React?"
  User: "Yes" → call update_notebook

- User: "Include error handling in n8n description"
  You: "Update the n8n description to mention error handling?"
  User: "Yes" → call update_notebook

Tip: You may update multiple fields at once if requested.

    Args:
        id: The notebook ID to update
        name: New display name
        description: New description
        topics: New topics list
        content_types: New content types
        use_cases: New use cases
        tags: New tags
        url: New notebook URL

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id
    if name is not None:
        mcp_args["name"] = name
    if description is not None:
        mcp_args["description"] = description
    if topics is not None:
        mcp_args["topics"] = topics
    if content_types is not None:
        mcp_args["content_types"] = content_types
    if use_cases is not None:
        mcp_args["use_cases"] = use_cases
    if tags is not None:
        mcp_args["tags"] = tags
    if url is not None:
        mcp_args["url"] = url

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="update_notebook",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def remove_notebook(id: str | Any = None, **kwargs) -> Any:
    """Dangerous — requires explicit user confirmation.

## Confirmation Workflow
1) User requests removal ("Remove the React notebook")
2) Look up full name to confirm
3) Ask: "Remove '[notebook_name]' from your library? (Does not delete the actual NotebookLM notebook)"
4) Only on explicit "Yes" → call remove_notebook

Never remove without permission or based on assumptions.

Example:
User: "Delete the old React notebook"
You: "Remove 'React Best Practices' from your library?"
User: "Yes" → call remove_notebook

    Args:
        id: The notebook ID to remove

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="remove_notebook",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def search_notebooks(query: str | Any = None, **kwargs) -> Any:
    """Search library by query (name, description, topics, tags). Use to propose relevant notebooks for the task and then ask which to use.

    Args:
        query: Search query

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if query is not None:
        mcp_args["query"] = query

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="search_notebooks",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def get_library_stats(**kwargs) -> Any:
    """Get statistics about your notebook library (total notebooks, usage, etc.)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="get_library_stats",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def list_sessions(**kwargs) -> Any:
    """List all active sessions with stats (age, message count, last activity). Use to continue the most relevant session instead of starting from scratch.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="list_sessions",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def close_session(session_id: str | Any = None, **kwargs) -> Any:
    """Close a specific session by session ID. Ask before closing if the user might still need it.

    Args:
        session_id: The session ID to close

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if session_id is not None:
        mcp_args["session_id"] = session_id

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="close_session",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def reset_session(session_id: str | Any = None, **kwargs) -> Any:
    """Reset a session's chat history (keep same session ID). Use for a clean slate when the task changes; ask the user before resetting.

    Args:
        session_id: The session ID to reset

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if session_id is not None:
        mcp_args["session_id"] = session_id

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="reset_session",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def get_health(**kwargs) -> Any:
    """Get server health status including authentication state, active sessions, and configuration. Use this to verify the server is ready before starting research workflows.

If authenticated=false and having persistent issues:
Consider running cleanup_data(preserve_library=true) + setup_auth for fresh start with clean browser session.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="get_health",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def setup_auth(show_browser: bool | None = None, browser_options: dict[str, Any] | None = None, **kwargs) -> Any:
    """Google authentication for NotebookLM access - opens a browser window for manual login to your Google account. Returns immediately after opening the browser. You have up to 10 minutes to complete the login. Use 'get_health' tool afterwards to verify authentication was saved successfully. Use this for first-time authentication or when auto-login credentials are not available. For switching accounts or rate-limit workarounds, use 're_auth' tool instead.

TROUBLESHOOTING for persistent auth issues:
If setup_auth fails or you encounter browser/session issues:
1. Ask user to close ALL Chrome/Chromium instances
2. Run cleanup_data(confirm=true, preserve_library=true) to clean old data
3. Run setup_auth again for fresh start
This helps resolve conflicts from old browser sessions and installation data.

    Args:
        show_browser: Show browser window (simple version). Default: true for setup. For advanced control, use browser_options instead.
        browser_options: Optional browser settings. Control visibility, timeouts, and stealth behavior.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if show_browser is not None:
        mcp_args["show_browser"] = show_browser
    if browser_options is not None:
        mcp_args["browser_options"] = browser_options

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="setup_auth",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def re_auth(show_browser: bool | None = None, browser_options: dict[str, Any] | None = None, **kwargs) -> Any:
    """Switch to a different Google account or re-authenticate. Use this when:
- NotebookLM rate limit is reached (50 queries/day for free accounts)
- You want to switch to a different Google account
- Authentication is broken and needs a fresh start

This will:
1. Close all active browser sessions
2. Delete all saved authentication data (cookies, Chrome profile)
3. Open browser for fresh Google login

After completion, use 'get_health' to verify authentication.

TROUBLESHOOTING for persistent auth issues:
If re_auth fails repeatedly:
1. Ask user to close ALL Chrome/Chromium instances
2. Run cleanup_data(confirm=false, preserve_library=true) to preview old files
3. Run cleanup_data(confirm=true, preserve_library=true) to clean everything except library
4. Run re_auth again for completely fresh start
This removes old installation data and browser sessions that can cause conflicts.

    Args:
        show_browser: Show browser window (simple version). Default: true for re-auth. For advanced control, use browser_options instead.
        browser_options: Optional browser settings. Control visibility, timeouts, and stealth behavior.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if show_browser is not None:
        mcp_args["show_browser"] = show_browser
    if browser_options is not None:
        mcp_args["browser_options"] = browser_options

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="re_auth",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def cleanup_data(confirm: bool | Any = None, preserve_library: bool | None = None, **kwargs) -> Any:
    """ULTRATHINK Deep Cleanup - Scans entire system for ALL NotebookLM MCP data files across 8 categories. Always runs in deep mode, shows categorized preview before deletion.

⚠️ CRITICAL: Close ALL Chrome/Chromium instances BEFORE running this tool! Open browsers can prevent cleanup and cause issues.

Categories scanned:
1. Legacy Installation (notebooklm-mcp-nodejs) - Old paths with -nodejs suffix
2. Current Installation (notebooklm-mcp) - Active data, browser profiles, library
3. NPM/NPX Cache - Cached installations from npx
4. Claude CLI MCP Logs - MCP server logs from Claude CLI
5. Temporary Backups - Backup directories in system temp
6. Claude Projects Cache - Project-specific cache (optional)
7. Editor Logs (Cursor/VSCode) - MCP logs from code editors (optional)
8. Trash Files - Deleted notebooklm files in system trash (optional)

Works cross-platform (Linux, Windows, macOS). Safe by design: shows detailed preview before deletion, requires explicit confirmation.

LIBRARY PRESERVATION: Set preserve_library=true to keep your notebook library.json file while cleaning everything else.

RECOMMENDED WORKFLOW for fresh start:
1. Ask user to close ALL Chrome/Chromium instances
2. Run cleanup_data(confirm=false, preserve_library=true) to preview
3. Run cleanup_data(confirm=true, preserve_library=true) to execute
4. Run setup_auth or re_auth for fresh browser session

Use cases: Clean reinstall, troubleshooting auth issues, removing all traces before uninstall, cleaning old browser sessions and installation data.

    Args:
        confirm: Confirmation flag. Tool shows preview first, then user confirms deletion. Set to true only after user has reviewed the preview and explicitly confirmed.
        preserve_library: Preserve library.json file during cleanup. Default: false. Set to true to keep your notebook library while deleting everything else (browser data, caches, logs).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if confirm is not None:
        mcp_args["confirm"] = confirm
    if preserve_library is not None:
        mcp_args["preserve_library"] = preserve_library

    async def _async_call():
        return await call_mcp_tool(
            server_name="notebooklm",
            tool_name="cleanup_data",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['ask_question', 'add_notebook', 'list_notebooks', 'get_notebook', 'select_notebook', 'update_notebook', 'remove_notebook', 'search_notebooks', 'get_library_stats', 'list_sessions', 'close_session', 'reset_session', 'get_health', 'setup_auth', 're_auth', 'cleanup_data']
