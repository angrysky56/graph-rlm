import asyncio
import os
import sys

backend_src_path = os.path.join(os.getcwd(), 'graph_rlm', 'backend', 'src')
sys.path.insert(0, backend_src_path)

from core.llm import llm


async def main():
    print("Testing Embedding...")
    vec = await llm.get_embedding("Hello world")
    if vec:
        print(f"Success! Length: {len(vec)}")
        print(vec[:5])
    else:
        print(f"Failed! Received: {vec}")

asyncio.run(main())
