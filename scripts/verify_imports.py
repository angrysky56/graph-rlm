import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

print("Testing Agent Import...")
try:
    from graph_rlm.backend.src.core.agent import agent
    print("✅ Agent imported successfully.")
except Exception as e:
    print(f"❌ Agent import failed: {e}")

print("Testing Database Import...")
try:
    from graph_rlm.backend.src.core.database import db
    print("✅ Database imported successfully.")
except Exception as e:
    print(f"❌ Database import failed: {e}")

print("Testing Shim Import...")
try:
    from graph_rlm.backend.src.core.db import db as shim_db
    print("✅ Shim imported successfully.")
except Exception as e:
    print(f"❌ Shim import failed: {e}")
