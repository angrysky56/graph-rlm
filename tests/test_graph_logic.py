
import json
from unittest.mock import MagicMock


# Define a mock process_node that mimics the new logic in endpoints.py
def process_node(entity):
    if entity is None:
        return None

    props = entity if isinstance(entity, dict) else entity.properties
    node_id = props.get("id")
    if not node_id:
        return None

    res = props.get("result", "")
    if not res or res == "None":
        res = props.get("execution_summary", "")

    return {
        "id": node_id,
        "prompt": props.get("prompt", ""),
        "result": res,
        "status": props.get("status", "pending"),
    }

def test_logic():
    nodes = {}

    # Simulate Row 1: NodeA (source) -> NodeB (target)
    # Target NodeB is seen here with minimal info (Cypher OPTIONAL MATCH target m often has props, but our logic was biased)
    # Actually, in the old logic, t_props was extracted but only saved if t_id not in nodes.
    # The real issue was that row 2 might have NodeB as source, but NodeB was already in 'nodes' as a target.

    row1 = {
        "n": {"id": "NodeA", "prompt": "PromptA", "result": "ResultA"},
        "r": "DECOMPOSES_INTO",
        "m": {"id": "NodeB", "prompt": "PromptB", "result": "ResultB"}
    }

    # Process Row 1
    s_data = process_node(row1["n"])
    if s_data["id"] not in nodes:
        nodes[s_data["id"]] = s_data

    t_data = process_node(row1["m"])
    if t_data["id"] not in nodes:
        nodes[t_data["id"]] = t_data
    else:
        nodes[t_data["id"]].update({k: v for k, v in t_data.items() if v is not None})

    print(f"After Row 1, NodeB prompt: '{nodes['NodeB'].get('prompt')}'")

    # Simulate Row 2: NodeB as Source (Matched as 'n' in Cypher)
    row2 = {
        "n": {"id": "NodeB", "prompt": "PromptB", "result": "ResultB"},
        "r": None,
        "m": None
    }

    s_data2 = process_node(row2["n"])
    s_id2 = s_data2["id"]
    if s_id2 not in nodes:
        nodes[s_id2] = s_data2
    else:
        # NEW LOGIC: Always update
        nodes[s_id2].update({k: v for k, v in s_data2.items() if v is not None})

    print(f"After Row 2, NodeB prompt: '{nodes['NodeB'].get('prompt')}'")

    # Test result/execution_summary fallback
    node_c = {"id": "NodeC", "prompt": "PromptC", "execution_summary": "Extracted Summary"}
    data_c = process_node(node_c)
    print(f"NodeC result (from execution_summary): '{data_c.get('result')}'")

if __name__ == "__main__":
    test_logic()
