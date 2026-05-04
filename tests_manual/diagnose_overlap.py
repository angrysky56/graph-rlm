import numpy as np

from graph_rlm.backend.src.core.db import db


def diagnose():
    print("--- Graph Semantic Overlap Diagnostic ---")

    # Get all thoughts
    res = db.query("MATCH (n:Thought) RETURN n.id, n.prompt, n.result, n.embedding ORDER BY n.created_at DESC LIMIT 10")

    if not res:
        print("No thoughts found in DB.")
        return

    print(f"Found {len(res)} recent thoughts.")

    thoughts = []
    for r in res:
        # FalkorDB vectors might need conversion
        emb = r.get('n.embedding')
        if emb:
            thoughts.append({
                'id': r.get('n.id'),
                'prompt': r.get('n.prompt', '')[:50],
                'vec': np.array(emb)
            })

    if len(thoughts) < 2:
        print("Not enough thoughts to compare.")
        return

    latest = thoughts[0]
    print(f"\nLatest Thought: {latest['id']} ({latest['prompt']}...)")

    for other in thoughts[1:]:
        # Cosine Similarity
        norm_l = np.linalg.norm(latest['vec'])
        norm_o = np.linalg.norm(other['vec'])

        if norm_l > 0 and norm_o > 0:
            sim = np.dot(latest['vec'], other['vec']) / (norm_l * norm_o)
            print(f"  Similarity with {other['id']} ({other['prompt']}...): {sim:.4f}")
            if sim > 0.92:
                print("  ⚠️ REPETITION DETECTED!")

if __name__ == "__main__":
    diagnose()
