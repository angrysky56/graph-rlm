import time
import random
import uuid

def seed_graph_data(repo):
    """
    Populates the repository with a rich, complex dataset to demonstrate UI capabilities.
    Scenario: Designing a Self-Healing Neural Interface.
    """
    print("🌱 Seeding Graph Data...")

    root_id = "session_alpha_001"

    # 1. Root Node
    repo.create_thought_node({
        "id": "root_thought",
        "prompt": "Designing a Self-Healing Neural Interface for Autonomous Agents",
        "status": "success",
        "priority": "high",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "PROJECT ROOT: Neural Interface",
        "execution_summary": "Initiated architectural review. Splitting into 3 sub-domains."
    })

    # 2. Sub-Domains (Parallel Chains)
    domains = [
        ("domain_cag", "Constraint Augmented Generation (CAG) Layer", "active"),
        ("domain_dream", "Dreamer / Sleep Phase Logic", "running"),
        ("domain_repe", "Representation Engineering (RepE) Safety", "failed")
    ]

    domain_nodes = []
    for d_id, d_prompt, d_status in domains:
        node_id = f"node_{d_id}"
        repo.create_thought_node({
            "id": node_id,
            "prompt": d_prompt,
            "status": d_status,
            "priority": "high",
            "session_id": root_id,
            "root_session_id": root_id,
            "label": d_prompt
        }, parent_id="root_thought")
        domain_nodes.append(node_id)

    # 3. Flesh out CAG (Successful Chain)
    cag_root = domain_nodes[0]
    prev = cag_root
    for i in range(5):
        tid = f"cag_step_{i}"
        repo.create_thought_node({
            "id": tid,
            "prompt": f"Ingesting Document: cyber_security_protocols_v{i}.pdf",
            "status": "success",
            "priority": "medium",
            "session_id": root_id,
            "root_session_id": root_id,
            "label": f"CAG Ingestion Step {i+1}",
            "result": "Extracted 12 invariants."
        }, parent_id=prev)
        prev = tid

    # 4. Flesh out Dreamer (Active/Running Chain)
    dream_root = domain_nodes[1]

    # Branch A: Sleep Cycle
    tid_a = "dream_sleep_cycle"
    repo.create_thought_node({
        "id": tid_a,
        "prompt": "Initiating NREM Sleep Cycle...",
        "status": "running",
        "priority": "high",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "Sleep Cycle (Active)"
    }, parent_id=dream_root)

    # Branch B: Hallucination Check
    tid_b = "dream_hallucination"
    repo.create_thought_node({
        "id": tid_b,
        "prompt": "Scanning for Hallucinations in recent traces",
        "status": "pending",
        "priority": "medium",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "Hallucination Scanner"
    }, parent_id=dream_root)

    # 5. Flesh out RepE (Failed Chain with Reflexion)
    repe_root = domain_nodes[2]

    # Failed Node
    fail_id = "repe_fail_01"
    repo.create_thought_node({
        "id": fail_id,
        "prompt": "Extracting 'Deception' Vector from Embedding Layer 12",
        "status": "failed",
        "priority": "high",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "Vector Extraction FAILED",
        "result": "Error: Dimension Mismatch (4096 vs 3072)"
    }, parent_id=repe_root)

    # Reflexion Node (Correction)
    reflex_id = "repe_reflexion"
    repo.create_thought_node({
        "id": reflex_id,
        "prompt": "SYSTEM INTERVENTION: Reflexion Triggered. Adjusting dimensions.",
        "status": "reflexion",
        "priority": "high",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "💡 REFLEXION: Fix Dimensions"
    }, parent_id=fail_id)

    # Retry Node
    retry_id = "repe_retry"
    repo.create_thought_node({
        "id": retry_id,
        "prompt": "Retrying Vector Extraction with dim=3072",
        "status": "running",
        "priority": "medium",
        "session_id": root_id,
        "root_session_id": root_id,
        "label": "Retry Extraction"
    }, parent_id=reflex_id)

    # 6. Scatter some random nodes to fill space (Simulation of background thoughts)
    for i in range(15):
        pid = random.choice(domain_nodes + [cag_root, dream_root, repe_root])
        tid = f"bg_thought_{i}"
        status = random.choice(["success", "pending", "pending", "success"])
        repo.create_thought_node({
            "id": tid,
            "prompt": f"Background processing task {i}...",
            "status": status,
            "priority": "low",
            "session_id": root_id,
            "root_session_id": root_id,
            "label": f"Bg Task {i}"
        }, parent_id=pid)

    print("✅ Seed complete.")
