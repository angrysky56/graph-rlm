import asyncio
import time

import numpy as np

from graph_rlm.backend.src.core.db import db


def test_db_insert():
    tid = f"test_id_{time.time()}"
    shaky = np.float64(0.555)

    print("Inserting node...")
    db.create_thought_node(
        thought_id=tid,
        prompt="Test prompt",
        session_id="test_session",
        repe_shakiness=shaky,
        status="success"
    )

    print("Querying node...")
    q = "MATCH (n:Thought {id: $id}) RETURN n.id, n.repe_shakiness"
    rows = db.query(q, {"id": tid})
    print(rows)

if __name__ == "__main__":
    test_db_insert()
