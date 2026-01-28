from .db import GraphClient, db


class ContextIndex:
    """
    Constructs a 'Scratchpad' of active contexts (Thoughts/REPLs)
    to prevent context rot in the unified RLM graph.
    """

    def __init__(self):
        self.db: GraphClient = db

    def get_context_scratchpad(self, root_session_id: str) -> str:
        """
        Query FalkorDB for a hierarchical summary of the 'Sheaf' (Thoughts/Sessions).
        Returns a compact index to prevent context rot.
        """
        try:
            # Match all thoughts to build a hierarchical index
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id OR n.session_id = $root_id
            WITH n.session_id as sid, n
            ORDER BY n.created_at ASC
            WITH sid, count(n) as thought_count, collect(n.prompt) as prompts
            RETURN sid, thought_count, prompts[0] as initial_prompt
            """
            res = self.db.query(q, {"root_id": root_session_id})

            if not res:
                return "No active session history."

            lines = ["## Active Session Index (The Sheaf)"]
            lines.append(
                "You are in a Recursive Logic Machine (RLM). History is managed symbolically."
            )

            for row in res:
                # Handle row formats
                if isinstance(row, dict):
                    sid = row.get("sid", "unknown")
                    count = row.get("thought_count", 0)
                    prompt = row.get("initial_prompt", "")
                else:
                    sid = row[0]
                    count = row[1]
                    prompt = row[2]

                short_sid = str(sid)[:8]
                short_prompt = str(prompt)[:100].replace("\n", " ")
                lines.append(
                    f"- REPL [{short_sid}]: {short_prompt}... ({count} thoughts)"
                )

            lines.append("\n**How to access context**:")
            lines.append(
                "1. **Trace**: Check `active_repls` in your REPL for sub-session IDs."
            )
            lines.append(
                "2. **Recall**: Use `rlm.recall(query)` to fetch full details from memory."
            )
            lines.append(
                "3. **Inspect**: Use `graph_search(query)` for topological semantic search."
            )

            return "\n".join(lines)

        except Exception as e:
            return f"Error building Session Index: {e}"


context_index = ContextIndex()
