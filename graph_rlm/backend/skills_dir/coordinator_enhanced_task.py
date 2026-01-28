
import uuid
from typing import Any, Dict

# Import ChatDAG tools
# Assuming 'chatdag' is the server name
from graph_rlm.backend.mcp_tools.chatdag import search_knowledge, feed_data

# Import Coordinator tools
from graph_rlm.backend.src.core.agent import agent
import asyncio

async def coordinator_enhanced_task(task_description: str) -> Any:
    """
    Use ChatDAG as memory layer for coordinator tasks.

    This skill:
    1. Searches ChatDAG for similar prior tasks/knowledge
    2. Runs the coordinator agent with that context
    3. Saves the execution result back to ChatDAG

    Args:
        task_description: The task to perform.

    Returns:
        The result of the task execution.
    """

    # 1. Check for prior work
    print(f"🔍 Searching ChatDAG for context on: {task_description}")
    try:
        prior_work = await search_knowledge(
            query=f"similar tasks: {task_description}",
            k=10
        )
    except Exception as e:
        print(f"⚠️ Warning: Failed to search ChatDAG: {e}")
        prior_work = []

    # 2. Execute with context
    context_str = f"Context from memory:\n{prior_work}\n\nTask: {task_description}"

    print(f"🤖 Running agent task with context...")
    # Using default model to ensure compatibility
    result = await asyncio.to_thread(
        agent.query_sync,
        prompt=context_str,
        session_id=f"coordinator_task_{uuid.uuid4()}"
    )

    # 3. Store execution trace
    print(f"💾 Storing result to ChatDAG...")
    try:
        # Format result safely
        if isinstance(result, dict) or isinstance(result, list):
            import json
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        await feed_data(
            content=f"Task: {task_description}\nResult: {result_str}",
            source_id=f"coordinator/execution/{uuid.uuid4()}",
            metadata={
                "type": "api_response",
                "priority": "high",
                "domain": "engineering"
            }
        )
        print("✅ Storage successful")
    except Exception as e:
        print(f"⚠️ Warning: Failed to feed data to ChatDAG: {e}")

    return result
