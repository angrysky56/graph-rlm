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
        Query FalkorDB for the 'Sheaf' of thoughts in this conversation tree.
        Returns a formatted string for the System Prompt.
        """
        try:
            # Match all thoughts connected to this root (or are the root)
            # Return ID, Prompt (Summary), Status, and REPL ID (if persisted)
            # We assume thoughts have a 'repl_id' property if they executed code.
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id OR n.session_id = $root_id
            RETURN n
            ORDER BY n.created_at ASC
            """
            res = self.db.query(q, {"root_id": root_session_id})

            if not res:
                return "No active context history."

            lines = ["## Active Context Index (Sheaf Scratchpad)"]

            for row in res:
                node = row[0] if isinstance(row, list) else row.get("n")
                if not node:
                    continue

                props = node.properties if hasattr(node, "properties") else node
                if not isinstance(props, dict):
                    continue

                tid = props.get("id", "unknown")[:8]
                prompt = props.get("prompt", "")[:4096].replace("\n", " ")
                status = props.get("status", "pending")

                # Check for REPL ID (stored properties or inferred)
                # If we don't store repl_id explicitly yet, we might miss it.
                # Assuming we will/do store it.
                repl_info = ""
                if "repl_id" in props:
                    repl_info = f" | REPL: {props['repl_id'][:8]}"

                result_peek = ""
                if "result" in props:
                    r_text = str(props["result"])
                    if len(r_text) > 4096:
                        r_text = r_text[:4096] + "..."
                    result_peek = f" -> {r_text}"

                lines.append(
                    f"- [{tid}] {prompt}... ({status}{repl_info}){result_peek}"
                )

            lines.append(
                "\nUse `rlm.recall('query')` to fetch full details of any node."
            )
            return "\n".join(lines)

        except Exception as e:
            return f"Error building Context Index: {e}"


context_index = ContextIndex()
