
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from graph_rlm.backend.src.core.llm import LLMService


async def debug_llm():
    service = LLMService()
    print(f"Provider: {service.provider}")
    print(f"Config: {service.config}")

    print("\n--- Listing Models ---")
    try:
        models = service.list_models()
        print(f"Total Models Found: {len(models)}")
        # Look for the current model
        current = service.config.get('model')
        found = any(m['id'] == current for m in models)
        print(f"Current Model '{current}' Found: {found}")

        if not found:
            print("Suggesting alternatives...")
            alternatives = [m['id'] for m in models if 'gemini' in m['id'].lower()]
            print(f"Gemini Alts: {alternatives[:10]}")
    except Exception as e:
        print(f"List Models Failed: {e}")

    print("\n--- Testing Generation ---")
    try:
        res = await service.generate("Hello, are you there?", system="Be brief.")
        print(f"Response: '{res}'")
        if not res:
            print("!!! EMPTY RESPONSE DETECTED !!!")
    except Exception as e:
        print(f"Generation Failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_llm())
