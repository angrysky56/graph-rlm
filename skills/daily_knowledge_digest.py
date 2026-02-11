"""
Daily Knowledge Digest Skill.

Summarizes knowledge across specified topics by querying ChatDAG and
formatting the results into a markdown report.
"""

import datetime
import logging
from typing import List, Optional

from graph_rlm.backend.mcp_tools import call_tool

logger = logging.getLogger("graph_rlm.skills.daily_knowledge_digest")


async def daily_knowledge_digest(
    topics: Optional[List[str]] = None, top_k: int = 3
) -> str:
    """
    Generate a digest of knowledge across specified topics.

    Args:
        topics: List of topics to include. Defaults to generic AI/Memory topics.
        top_k: Number of top results per topic to fetch.

    Returns:
        Formatted markdown digest summarizing key findings.
    """
    if topics is None:
        topics = ["AI agents", "coordination patterns", "semantic memory"]

    digest_parts = ["# 📚 Daily Knowledge Digest\n"]
    digest_parts.append(f"*Generated at: {datetime.datetime.now().isoformat()}*\n")

    for topic in topics:
        digest_parts.append(f"\n## 🔍 {topic.title()}\n")

        try:
            result = await call_tool(
                "chatdag",
                "search_knowledge",
                {"query": topic, "k": top_k},
            )

            if result:
                # Result parsing
                content_text = ""
                if isinstance(result, list):
                    for item in result:
                        if hasattr(item, "text"):
                            content_text += item.text + "\n"
                        else:
                            content_text += str(item) + "\n"
                else:
                    content_text = str(result)

                lines = content_text.split("\n")
                count = 0
                for line in lines:
                    line = line.strip()
                    if line:
                        # Use blockquote for content to distinguish it
                        # Truncate lines that are too long
                        safe_line = line[:200] + ("..." if len(line) > 200 else "")
                        digest_parts.append(f"> {safe_line}")
                        digest_parts.append("")  # Spacing
                        count += 1
                        if count >= 3:  # Limit entries per topic
                            break
            else:
                digest_parts.append("- *No relevant knowledge found.*\n")

        except RuntimeError as e:
            logger.error("Search error for topic '%s': %s", topic, e)
            digest_parts.append(f"- *Search error: {e}*\n")
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected search error for topic '%s': %s", topic, e)
            digest_parts.append("- *Technical failure during search.*\n")

    digest_parts.append("\n---\n*Digest powered by Graph RLM + ChatDAG*")

    return "\n".join(digest_parts)
