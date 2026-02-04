import asyncio

from graph_rlm.backend.src.core.dream import dreamer


async def reproduce():
    print("Triggering Dream Cycle...")
    # Mocking necessary state if needed, but let's just call it
    try:
        res = await dreamer.dream_cycle()
        print(f"Dream cycle finished: {res}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce())
