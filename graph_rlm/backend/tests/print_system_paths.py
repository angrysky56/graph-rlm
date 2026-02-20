"""
Verification script for Graph-RLM system paths.
"""

import sys
from pathlib import Path

# Add repo root to path to support absolute imports like 'graph_rlm.backend...'
repo_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add src to path to support local imports like 'core.prompts' if script is run directly
src_path = repo_root / "graph_rlm" / "backend" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.prompts import get_system_paths


def main():
    print("--- Graph-RLM System Path Verification ---")
    paths = get_system_paths()
    for name, path in paths.items():
        print(f"{name:15}: {path}")
        if not path.exists():
            print("  [!] WARNING: Path does not exist!")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
