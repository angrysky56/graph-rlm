import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from graph_rlm.backend.src.core.llm import llm


def check_embedding_dim():
    print("Checking embedding dimension for current model...")
    text = "Hello world"
    vec = llm.get_embedding(text)
    if vec:
        print(f"Success! Vector length: {len(vec)}")
    else:
        print("Failed to get embedding.")

if __name__ == "__main__":
    check_embedding_dim()
