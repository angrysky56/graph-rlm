async def daily_knowledge_digest(topics: list[str] = None, top_k: int = 3) -> str:
    """
    Generate a digest of knowledge across specified topics.

    Args:
        topics: List of topics to include. Defaults to ["AI", "Coordination", "Memory"].
        top_k: Number of top results per topic.

    Returns:
        Formatted markdown digest summarizing key learnings.
    """
    import datetime

    from mcp_coordinator.coordinator_client import call_mcp_tool

    if topics is None:
        topics = ["AI agents", "coordination patterns", "semantic memory"]

    digest_parts = ["# 📚 Daily Knowledge Digest\n"]
    digest_parts.append(f"*Generated at: {datetime.datetime.now().isoformat()}*\n")

    for topic in topics:
        digest_parts.append(f"\n## 🔍 {topic.title()}\n")

        try:
            result = await call_mcp_tool(
                server_name="chatdag",
                tool_name="search_knowledge",
                arguments={"query": topic, "k": top_k},
            )

            if result:
                # Result from call_mcp_tool can be a list of Content objects
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
                        digest_parts.append(f"> {line[:200]}")
                        digest_parts.append("")  # Spacing
                        count += 1
                        if count >= 3:  # Limit entries per topic
                            break
            else:
                digest_parts.append("- *No relevant knowledge found.*\n")

        except Exception as e:
            digest_parts.append(f"- *Search error: {e}*\n")

    digest_parts.append("\n---\n*Digest powered by MCP Coordinator + ChatDAG*")

    return "\n".join(digest_parts)
