import sys
import os
import unittest
import time

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db import db
from core.llm import llm

class TestVectors(unittest.TestCase):
    def test_embedding_generation(self):
        vec = llm.get_embedding("Test prompt")
        self.assertIsInstance(vec, list)
        self.assertTrue(len(vec) > 0)

    def test_vector_storage_and_search(self):
        print("\n--- Starting Manual Vector Test ---")
        # 1. Clean slate
        try:
            db.query("MATCH (n:Thought) DETACH DELETE n")
            db.query("DROP VECTOR INDEX FOR (t:Thought) ON (t.embedding)")
        except:
            pass

        # 2. Manual Index Creation
        db.query("CREATE VECTOR INDEX FOR (t:Thought) ON (t.embedding) OPTIONS {dimension:768, similarityFunction:'cosine'}")

        # Poll specific for Thought index
        print("Waiting for index 'Thought'...")
        for _ in range(20):
            ready = False
            try:
                res = db.query("CALL db.indexes() YIELD label, status RETURN label, status")
                for row in res.result_set:
                    if row[0] == 'Thought' and row[1] == 'OPERATIONAL':
                        ready = True
                        break
            except:
                pass
            if ready: break
            time.sleep(0.5)

        # 3. Manual Insert
        vec = llm.get_embedding("manual test")
        db.query("CREATE (t:Thought {id: 'manual-1', embedding: vecf32($vec)})", {"vec": vec})

        time.sleep(2) # Extra buffer

        # 4. Manual Search
        res = db.query("CALL db.idx.vector.queryNodes('Thought', 'embedding', 5, vecf32($vec)) YIELD node, score RETURN node, score", {"vec": vec})
        print(f"Manual Search Results: {len(res.result_set)}")
        if len(res.result_set) > 0:
            print(f"Top Score: {res.result_set[0][1]}")

        self.assertTrue(len(res.result_set) > 0)
        print("Manual verification passed!")

if __name__ == '__main__':
    unittest.main()
