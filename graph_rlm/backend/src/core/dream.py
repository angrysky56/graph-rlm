import uuid
from typing import Any, Dict

from .db import GraphClient, db
from .llm import llm
from .logger import get_logger
from .sheaf import sheaf

logger = get_logger("graph_rlm.dreamer")


class Dreamer:
    """
    The 'Sleep' Phase of the Graph-RLM architecture.
    Consolidates high-entropy (Surprise) events into 'Wisdom' (Insights).
    """

    def __init__(self):
        self.db: GraphClient = db
        self.llm = llm

    def dream_cycle(self) -> Dict[str, Any]:
        """
        Main Sleep Cycle:
        1. Query 'Surprise' (High Energy Edges / Failed Tests).
        2. Consolidate into 'Insight'.
        3. Write Insight back to Graph.
        """
        logger.info("🛌 Initiating Dream Cycle (Sleep Phase)...")

        # 1. Gather Surprise (High Energy Edges)
        # We reuse the Sheaf Monitor's energy calculation which now includes Test Failures.
        surprise_events = sheaf.compute_sheaf_surprise_score(limit=10)

        if not surprise_events:
            logger.info("No high-surprise events found. Sleep was peaceful.")
            return {"status": "peaceful", "insights": []}

        logger.info(f"Found {len(surprise_events)} high-surprise events.")

        # 2. Formulate the Dream Prompt
        events_desc = []
        for event in surprise_events:
            # Fetch prompt content for context
            src_node = self._get_node_scan(event["source"])
            tgt_node = self._get_node_scan(event["target"])

            status_str = "FAILED" if event.get("status") == "failed" else "Unknown"
            events_desc.append(
                f"- Edge: {event['source']} -> {event['target']}\n"
                f"  Surprise Score: {event['surprise_score']:.2f}\n"
                f"  Status: {status_str}\n"
                f"  Parent Thought: {src_node.get('prompt', 'Unknown')[:100]}...\n"
                f"  Child Action: {tgt_node.get('prompt', 'Unknown')[:100]}...\n"
                f"  Result: {tgt_node.get('result', 'Unknown')[:200]}..."
            )

        dream_prompt = (
            "You are the 'Dreamer' component of an AI system.\n"
            "Your job is to analyze 'Surprise Events' (failures, contradictions, inconsistencies) "
            "from the recent wake cycle and consolidate them into a coherent 'Insight' or 'Rule'.\n\n"
            "Here are the High-Surprise Events:\n" + "\n".join(events_desc) + "\n\n"
            "Instructions:\n"
            "1. Identify the common pattern of failure (e.g., 'Library X always fails with error Y').\n"
            "2. Formulate a 'Guardrail Rule' to prevent this in the future.\n"
            "3. Return specific actionable advice for the Agent.\n"
        )

        # 3. Generate Insight
        try:
            insight_text = self.llm.generate(
                prompt=dream_prompt,
                system="You are a Meta-Cognitive Analysis Engine. Be concise and prescriptive.",
                stream=False,
            )
        except Exception as e:
            logger.error(f"Dream failed during generation: {e}")
            return {"status": "error", "message": str(e)}

        # 4. Consolidate (Write Rule/Insight)
        logger.info(f"Dream Insight Generated: {insight_text[:100]}...")

        insight_id = str(uuid.uuid4())
        self._save_insight(insight_id, insight_text)

        # Optionally, create CONFLICTS_WITH edges if we can identify them,
        # but for now, the Insight node serves as the anchor.

        return {
            "status": "lucid",
            "events_processed": len(surprise_events),
            "insight": insight_text,
            "id": insight_id,
        }

    def _get_node_scan(self, node_id: str) -> Dict[str, Any]:
        """Helper to get node props for context."""
        try:
            res = self.db.query("MATCH (n:Thought {id: $id}) RETURN n", {"id": node_id})
            if res and res[0]:
                node = res[0]
                if isinstance(node, list):
                    node = node[0]

                # Check properties
                if hasattr(node, "properties"):
                    return node.properties
                if isinstance(node, dict):
                    return node
            return {}
        except Exception:
            return {}

    def _save_insight(self, insight_id: str, content: str):
        """Save the insight as a permanent 'Rule' or 'Wisdom' node AND to rules.md."""
        # 1. Save to Graph
        cypher = """
        CREATE (i:Insight {
            id: $id,
            content: $content,
            created_at: timestamp(),
            type: 'dream_consolidation'
        })
        """
        self.db.query(cypher, {"id": insight_id, "content": content})

        # 2. Append to rules.md (Marge's Rules)
        try:
            from pathlib import Path

            backend_root = Path(__file__).parent.parent.parent
            rules_path = backend_root / "rules.md"

            # Create if not exists
            if not rules_path.exists():
                rules_path.write_text("# System Guardrails (Marge's Rules)\n\n")

            with open(rules_path, "a") as f:
                f.write(
                    f"\n\n### Insight {insight_id[:8]} (Auto-Generated)\n{content}\n"
                )

            logger.info(f"Insight appended to {rules_path}")

        except Exception as e:
            logger.error(f"Failed to update rules.md: {e}")


dreamer = Dreamer()
