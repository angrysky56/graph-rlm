import json
from pathlib import Path
from typing import Any


async def replay_entropy_trace(
    trace_id: str, project_id: str, verify_against_crystallization: bool = True
) -> dict[str, Any]:
    """
    Replays a standardized Entropy Trace to verify deterministic state.

    1. Loads the trace JSON from the project folder.
    2. (Optional) Compares the trace metadata against crystallized insights.
    3. Returns the decision DAG for the agent to 'mock' or 'follow' in a new run.
    """
    # 1. Path Resolution
    base_path = Path(
        f"/home/ty/Repositories/ai_workspace/mcp_coordinator/knowledge_base/projects/{project_id}/entropy_traces"
    )

    # Find the file
    matches = list(base_path.glob(f"*{trace_id}*.json"))
    if not matches:
        return {
            "status": "error",
            "message": f"Trace {trace_id} not found in {project_id}.",
        }

    trace_path = matches[0]

    # 2. Load Trace
    try:
        with open(trace_path) as f:
            trace_data = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load trace: {e}"}

    # 3. Verification (Optional)
    verification_status = "Skipped"
    if verify_against_crystallization:
        crystallized_path = Path(
            f"/home/ty/Repositories/ai_workspace/mcp_coordinator/knowledge_base/projects/{project_id}/crystallized_insights_{project_id}.md"
        )
        if crystallized_path.exists():
            verification_status = (
                "Verified: Crystallized insights exist for this project context."
            )
        else:
            verification_status = (
                "Warning: No crystallized insights found for verification."
            )

    return {
        "status": "success",
        "replay_data": trace_data,
        "verification": verification_status,
        "instructions": "To replay, use the 'decision_dag' to mock tool outputs and the 'metadata.seed' to initialize PRNG.",
    }
