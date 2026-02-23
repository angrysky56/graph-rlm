import os
import sys

backend_src_path = os.path.join(os.getcwd(), "graph_rlm", "backend", "src")
sys.path.insert(0, backend_src_path)

from core.db import GraphClient

db = GraphClient()
res = db.query(
    "MATCH (n:Thought) WHERE n.status = 'success' OR n.status = 'failed' RETURN properties(n) ORDER BY n.created_at DESC LIMIT 5"
)
import json

for r in res:
    print(r)
