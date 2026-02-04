from typing import List, Dict, Any, Optional
from .database import client


class ContextIndex:
    """
    Constructs a 'Scratchpad' of active contexts (Thoughts/REPLs)
    to prevent context rot in the unified RLM graph.
    Delegates data fetching to the GraphRepository.
    """

    def __init__(self):
        # We access client.repo dynamically to ensure init
        pass

    def get_context_scratchpad(self, root_session_id: str) -> str:
        """
        Query Graph for a topological summary of active contexts.
        Provides a condensed 'Sheaf' to prevent context rot.
        """
        try:
            # Get structured data from repo
            data = client.repo.get_context_scratchpad_data(root_session_id, limit=10)

            if not data:
                return "No active session history."

            lines = ["## Active Session Index (The Sheaf)"]
            lines.append(
                "You are in a Recursive Logic Machine. Memory is a Global Thought Graph."
            )
            lines.append(
                "Below are the most active REPL sub-sessions in this workspace:"
            )

            for row in data:
                sid = row.get("sid", "unknown")
                count = row.get("count", 0)
                prompt = row.get("prompt", "")

                short_sid = str(sid)
                short_prompt = str(prompt).replace("\n", " ")
                lines.append(f"- REPL [{short_sid}]: {short_prompt} ({count} thoughts)")

            lines.append("\n**Semantic Recall Implementation**:")
            lines.append("- The graph is indexed for VECTOR SEARCH.")
            lines.append(
                "- If you are missing details or searching for specific information, DO NOT SCAN ALL NODES."
            )
            lines.append(
                "- Use `rlm.recall(query)` to find semantically relevant thoughts globally."
            )
            lines.append("- Use `graph_search(query)` for structural exploration.")

            return "\n".join(lines)

        except Exception as e:
            return f"Error building Session Index: {e}"

    def get_active_scratchpad_data(self, root_session_id: str) -> list:
        """
        Returns structured data for the UI Scratchpad.
        """
        try:
            return client.repo.get_context_scratchpad_data(root_session_id, limit=20)
        except Exception:
            return []

    def get_current_running_thought(self, root_session_id: str) -> dict | None:
        """Returns the single thought currently being processed (status='running')."""
        try:
            return client.repo.get_current_running_thought(root_session_id)
        except Exception:
            return None

    def get_session_thoughts(self, session_id: str) -> list:
        """Returns all Thought nodes for a given session, ordered chronologically.

        CRITICAL: Only returns thoughts that belong to THIS session chain.
        - If session_id IS a root session: get all thoughts with that root_session_id
        - If session_id is a child: get thoughts with matching session_id only

        This prevents mixing thoughts from unrelated sessions.
        """
        try:
            # We delegate the check to the repository?
            # Or implement the check here using repo?
            # Repo implementation of get_session_thoughts takes is_root param.
            # We need to determine is_root here.

            # Check if this session is a root session
            # We can use a simple check via repo?
            # Repo doesn't have "is_root_session" method but we can query counts.
            # We can use get_session_trace to see if it returns anything for root_session_id=session_id?
            # Or query directly.
            # Let's add a helper to repo or just try.

            # Since get_session_trace returns thoughts where root_session_id = arg,
            # If we call it and get results, it IS a root session (or has children).
            # But wait, get_session_trace is specific to root.

            # Let's just use get_session_thoughts logic from repo, but we need to know is_root.
            # We can assume it is NOT root if we don't know, or try to infer.
            # Original code did a count query.
            # Let's implement that check using repo primitives?
            # Or just assume the repo handles it? No, repo needs `is_root` flag.

            # I can rely on `get_session_trace` if session_id == root_session_id.
            # But session_id passed here is just a string.

            # HACK: If session_id has thoughts where root_session_id == session_id, it is a root.
            # We can get one thought from this session and check its root_session_id?
            # `get_context_frontier` returns thoughts by session_id.

            frontier = client.repo.get_context_frontier(session_id, limit=1)
            is_root = False
            if frontier:
                thought = frontier[0]
                # If root_session_id equals session_id, it's a root session thought
                if thought.get("root_session_id") == session_id:
                     is_root = True
                # Or if the session_id passed matches?

            # Original logic:
            # check_q = "MATCH (n:Thought) WHERE n.root_session_id = $sid RETURN count(n)"
            # If count > 0, it is a root session (it is the root for some thoughts).

            # NetworkX repo doesn't support generic count query.
            # But we can use `get_session_trace(session_id)` -> if it returns list, it is a root session!
            trace = client.repo.get_session_trace(session_id)
            if trace:
                is_root = True

            return client.repo.get_session_thoughts(session_id, is_root)

        except Exception:
            return []


context_index = ContextIndex()
