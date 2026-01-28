import json
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

async def save_entropy_trace(
    name: str,
    project_id: str,
    decision_nodes: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a standardized Entropy Trace for deterministic workflow replay.
    
    Args:
        name: Name of the trace (e.g., 'successful_api_integration')
        project_id: The ID of the project this trace belongs to
        decision_nodes: List of nodes containing 'entropy_source', 'entropy_value', and 'decision_impact'
        metadata: Optional dict for model params, seeds, etc.
        description: Human-readable description of what this trace achieved
        
    Returns:
        Dict containing the file path and status
    """
    timestamp = int(time.time())
    trace_id = f"trace_{timestamp}_{name}"
    
    # Define the standardized schema
    trace_data = {
        "trace_id": trace_id,
        "name": name,
        "description": description or "No description provided",
        "timestamp": timestamp,
        "project_id": project_id,
        "metadata": metadata or {},
        "decision_dag": decision_nodes
    }
    
    # Ensure directory exists in the knowledge base
    base_path = Path("/home/ty/Repositories/ai_workspace/mcp_coordinator/knowledge_base/projects") / project_id / "entropy_traces"
    os.makedirs(base_path, exist_ok=True)
    
    file_path = base_path / f"{trace_id}.json"
    
    with open(file_path, "w") as f:
        json.dump(trace_data, f, indent=2)
        
    return {
        "status": "success",
        "trace_id": trace_id,
        "file_path": str(file_path),
        "message": f"Entropy trace '{name}' saved successfully."
    }
