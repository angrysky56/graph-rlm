"""
Context Indexer for Graph-RLM.
Maintains a topological summary of active contexts to prevent context rot.
"""

from .db import GraphClient, db
from .logger import get_logger

logger = get_logger("graph_rlm.context_index")


class ContextIndex:
    """
    Constructs a 'Scratchpad' of active contexts (Thoughts/REPLs)
    to prevent context rot in the unified RLM graph.
    """

    def __init__(self):
        self.db: GraphClient = db

    def get_context_scratchpad(self, root_session_id: str) -> str:
        """
        Query FalkorDB for a topological summary of active contexts.
        Provides a condensed 'Sheaf' to prevent context rot.
        """
        try:
            # Match recent session summaries and the overall structure
            q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $root_id OR n.session_id = $root_id
            WITH n.session_id as sid, n, n.created_at as ts
            ORDER BY ts ASC
            WITH sid, count(n) as thought_count,
                 collect(n.prompt)[0] as initial_prompt,
                 max(ts) as last_activity
            RETURN sid, thought_count, initial_prompt, last_activity
            ORDER BY last_activity DESC
            LIMIT 10
            """
            res = self.db.query(q, {"root_id": root_session_id})

            if not res:
                return "No active session history."

            lines = ["## Active Session Index (The Sheaf)"]
            lines.append(
                "You are in a Recursive Logic Machine. Memory is a Global Thought Graph."
            )
            lines.append(
                "Below are the most active REPL sub-sessions in this workspace:"
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
            # Same query as get_context_scratchpad but returns raw list
            q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $root_id OR n.session_id = $root_id)
            WITH n.session_id as sid, n, n.created_at as ts
            ORDER BY ts ASC
            WITH sid, count(n) as thought_count,
                 collect(n.prompt)[0] as initial_prompt,
                 max(ts) as last_activity
            RETURN sid, thought_count, initial_prompt, last_activity
            ORDER BY last_activity DESC
            LIMIT 20
            """
            res = self.db.query(q, {"root_id": root_session_id})

            data = []
            if res:
                for row in res:
                    if isinstance(row, dict):
                        data.append(
                            {
                                "sid": row.get("sid"),
                                "count": row.get("thought_count"),
                                "prompt": row.get("initial_prompt"),
                                "last_activity": row.get("last_activity"),
                            }
                        )
                    else:
                        data.append(
                            {
                                "sid": row[0],
                                "count": row[1],
                                "prompt": row[2],
                                "last_activity": row[3],
                            }
                        )
            return data
        except Exception as e:
            logger.error("Error in get_active_scratchpad_data: %s", e)
            return []

    def get_current_running_thought(self, root_session_id: str) -> dict | None:
        """Returns the single thought currently being processed (status='running')."""
        try:
            q = """
            MATCH (n:Thought)
            WHERE (n.root_session_id = $root_id OR n.session_id = $root_id)
              AND n.status = 'running'
            RETURN n.id as id, n.prompt as prompt, n.status as status, n.created_at as created_at
            ORDER BY n.created_at DESC LIMIT 1
            """
            res = self.db.query(q, {"root_id": root_session_id})
            if res:
                row = res[0]
                if isinstance(row, dict):
                    return row
                return {
                    "id": row[0],
                    "prompt": row[1],
                    "status": row[2],
                    "created_at": row[3],
                }
            return None
        except Exception as e:
            logger.error("Error in get_current_running_thought: %s", e)
            return None

    def get_session_thoughts(self, session_id: str) -> list:
        """Returns all Thought nodes for a given session, ordered chronologically.

        CRITICAL: Only returns thoughts that belong to THIS session chain.
        - If session_id IS a root session: get all thoughts with that root_session_id
        - If session_id is a child: get thoughts with matching session_id only

        This prevents mixing thoughts from unrelated sessions.
        """
        try:
            # First, determine if this session_id is a root or child
            # by checking if any thought has this as root_session_id
            check_q = """
            MATCH (n:Thought)
            WHERE n.root_session_id = $sid
            RETURN count(n) as cnt
            """
            check_res = self.db.query(check_q, {"sid": session_id})
            is_root = False
            if check_res:
                row = check_res[0]
                cnt = (
                    row.get("cnt", 0)
                    if isinstance(row, dict)
                    else (row[0] if row else 0)
                )
                is_root = cnt > 0

            if is_root:
                # This IS a root session - get all thoughts in this chain
                q = """
                MATCH (n:Thought)
                WHERE n.root_session_id = $sid
                RETURN n.id as id,
                       n.prompt as prompt,
                       n.status as status,
                       n.result as result,
                       n.created_at as created_at,
                       n.execution_summary as execution_summary,
                       n.next_action as next_action,
                       n.dreamer_analysis as dreamer_analysis,
                       n.final_response as final_response,
                       n.repl_id as repl_id,
                       n.session_id as session_id,
                       n.turn_id as turn_id,
                       n.step_id as step_id,
                       n.code_hash as code_hash
                ORDER BY n.created_at ASC
                """
            else:
                # This is likely a new/current session - get only its direct thoughts
                q = """
                MATCH (n:Thought)
                WHERE n.session_id = $sid
                RETURN n.id as id,
                       n.prompt as prompt,
                       n.status as status,
                       n.result as result,
                       n.created_at as created_at,
                       n.execution_summary as execution_summary,
                       n.next_action as next_action,
                       n.dreamer_analysis as dreamer_analysis,
                       n.final_response as final_response,
                       n.repl_id as repl_id,
                       n.session_id as session_id,
                       n.turn_id as turn_id,
                       n.step_id as step_id,
                       n.code_hash as code_hash
                ORDER BY n.created_at ASC
                """

            res = self.db.query(q, {"sid": session_id})
            data = []
            for row in res:
                if row is None:
                    continue
                if isinstance(row, dict):
                    data.append(row)
                else:
                    # Handle tuple format
                    data.append(
                        {
                            "id": row[0] if len(row) > 0 else None,
                            "prompt": row[1] if len(row) > 1 else "",
                            "status": row[2] if len(row) > 2 else "unknown",
                            "result": row[3] if len(row) > 3 else None,
                            "created_at": row[4] if len(row) > 4 else None,
                            "execution_summary": row[5] if len(row) > 5 else None,
                            "next_action": row[6] if len(row) > 6 else None,
                            "dreamer_analysis": row[7] if len(row) > 7 else None,
                            "final_response": row[8] if len(row) > 8 else None,
                            "repl_id": row[9] if len(row) > 9 else None,
                            "session_id": row[10] if len(row) > 10 else None,
                            "turn_id": row[11] if len(row) > 11 else None,
                            "step_id": row[12] if len(row) > 12 else None,
                            "code_hash": row[13] if len(row) > 13 else None,
                        }
                    )
            return data
        except Exception as e:
            logger.error("Error in get_session_thoughts: %s", e)
            return []


context_index = ContextIndex()
