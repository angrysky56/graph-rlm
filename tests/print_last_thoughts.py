import asyncio
import os
import sys
import traceback

sys.path.append(os.path.join(os.getcwd(), "graph_rlm", "backend", "src"))


async def main():
    try:
        from core.db import GraphClient

        db = GraphClient()
        res = db.query(
            "MATCH (n:Thought) RETURN n.session_id AS sid, n.step_id AS step, n.prompt AS prompt ORDER BY n.timestamp DESC LIMIT 20"
        )
        if hasattr(res, "result_set"):
            res = res.result_set
        for r in reversed(list(res)):
            prompt = str(dict(r).get("prompt", "None"))
            prompt_100 = prompt[:100].replace("\n", "\\n")
            step = str(dict(r).get("step", "N/A"))
            print(f"Step {step}: {prompt_100}")
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
