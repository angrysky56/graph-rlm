"""
Coordinator Enhanced Task Skill.

Uses ChatDAG as a memory layer for coordinator tasks, enabling retrieval
of prior context and persistence of results.
"""

import json
import logging
import uuid
from typing import Any

from graph_rlm.backend.mcp_tools.chatdag import feed_data, search_knowledge
from graph_rlm.backend.src.core.agent import agent

logger = logging.getLogger("graph_rlm.skills.coordinator_enhanced_task")


async def coordinator_enhanced_task(task_description: str) -> Any:
    """
    Use ChatDAG as memory layer for coordinator tasks.

    This skill:
    1. Searches ChatDAG for similar prior tasks/knowledge.
    2. Runs the coordinator agent with that context.
    3. Saves the execution result back to ChatDAG.

    Args:
        task_description: The task to perform.

    Returns:
        The result of the task execution.
    """

    # 1. Check for prior work
    print(f"🔍 Searching ChatDAG for context on: {task_description}")
    try:
        prior_work = await search_knowledge(
            query=f"similar tasks: {task_description}", k=10
        )
    except RuntimeError as e:
        logger.warning("Failed to search ChatDAG: %s", e)
        prior_work = []

    # 2. Execute with context
    context_str = f"Context from memory:\n{prior_work}\n\nTask: {task_description}"

    print("🤖 Running agent task with context...")

    # Note: query_sync is an async function in the RLM Agent implementation.
    # No need for asyncio.to_thread.
    result = await agent.query_sync(
        prompt=context_str,
        session_id=f"coordinator_task_{uuid.uuid4()}",
    )

    # 3. Store execution trace
    print("💾 Storing result to ChatDAG...")
    try:
        # Format result safely
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        await feed_data(
            content=f"Task: {task_description}\nResult: {result_str}",
            source_id=f"coordinator/execution/{uuid.uuid4()}",
            metadata={
                "type": "execution_trace",
                "priority": "high",
                "domain": "engineering",
            },
        )
        print("✅ Storage successful")
    except RuntimeError as e:
        logger.error("Failed to feed data to ChatDAG: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error during storage: %s", e)

    return result
