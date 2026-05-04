import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../graph_rlm/backend/src")))

from core.db import db
from core.sheaf import sheaf
from core.thimac_memory import ThimacMemory, ThimacOperation, ThimacLevel, ThimacEvent

async def run_test():
    session_id = "test_stress_session"
    print("🧹 Cleaning up old test data...")
    db.query("MATCH (n:Thought) WHERE n.session_id = $sid DETACH DELETE n", {"sid": session_id})

    print("🌱 Simulating noisy graph...")
    # Inject 10 successful nodes
    for i in range(10):
        db.create_thought_node(
            thought_id=f"good_{i}",
            prompt=f"Good thought {i}",
            session_id=session_id,
            status="success",
            sheaf_score=0.1
        )
    
    # Inject 5 failed nodes (Noise)
    for i in range(5):
        db.create_thought_node(
            thought_id=f"bad_{i}",
            prompt=f"Bad thought {i}",
            session_id=session_id,
            status="failed",
            sheaf_score=0.2
        )
        
    # Inject 5 high anomaly nodes (Noise)
    for i in range(5):
        db.create_thought_node(
            thought_id=f"anomaly_{i}",
            prompt=f"Anomaly thought {i}",
            session_id=session_id,
            status="success",
            sheaf_score=0.8
        )
        
    print("🔍 Calculating Topological Stress...")
    stress = sheaf.calculate_topological_stress(session_id)
    print(f"Stress Ratio: {stress:.2f} (Expected: ~0.50 since 10/20 nodes are noisy)")
    
    print("✂️ Triggering Auto-Pruning...")
    pruned = db.force_consolidate_noisy_branches(session_id)
    print(f"Pruned Nodes Count: {pruned} (Expected: 10)")
    
    print("🔄 Recalculating Stress...")
    new_stress = sheaf.calculate_topological_stress(session_id)
    print(f"New Stress Ratio: {new_stress:.2f} (Expected: 0.00)")
    
    print("🧠 Testing ThimacMemory compression...")
    mem = ThimacMemory()
    for i in range(40):
        mem.subsistence.append(ThimacEvent(
            thought_id=f"t_{i}",
            operation=ThimacOperation.PROCESS,
            level=ThimacLevel.SUBSISTENCE,
            status="success"
        ))
    
    print(f"Before Stress Adapt: {len(mem.subsistence)} subsistence nodes")
    mem.adapt_to_stress(stress) # use old high stress
    print(f"After High Stress Adapt: {len(mem.subsistence)} subsistence nodes (Should drop PROCESS nodes)")
    
    print("✅ Test Complete.")

if __name__ == "__main__":
    asyncio.run(run_test())
