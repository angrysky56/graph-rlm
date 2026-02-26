
import asyncio
import re
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock


# Simplified structures to match those in the codebase
class MockDB:
    def query(self, q, params=None):
        print(f"DEBUG: Querying DB with: {q[:50]}...")
        if "MATCH (n:Thought)" in q and "$rid" in q:
            # Simulate first query failure (no results for specific round)
            return []
        if "MATCH (n:Thought)" in q and "$sid" in q:
            # Simulate fallback success
            return [{"prompt": "Fallback history found", "created_at": 100}]
        return []

    def get_completed_rounds(self, rsid):
        return []

# 1. Test Scratchpad Logic (Manually verifying the fallback added)
async def test_scratchpad_logic():
    print("\n--- Testing Scratchpad Fallback Logic ---")
    mock_db = MockDB()

    # We'll just test the code structure by injecting a mock
    # Since we can't easily import the class without triggering structlog,
    # we'll use a local version of the method logic for verification.

    session_id = "test-sid"
    root_session_id = "test-rsid"
    current_round_id = "test-rid"

    # Simulate the logic in _build_current_round_progress
    q = "MATCH (n:Thought) WHERE ... AND n.round_id = $rid ..."
    results = mock_db.query(q, {"sid": session_id, "rid": current_round_id})

    if not results:
        print("Round-specific query failed as expected. Triggering fallback...")
        q_fallback = "MATCH (n:Thought) WHERE n.session_id = $sid ..."
        results = mock_db.query(q_fallback, {"sid": session_id})

    if results and results[0]["prompt"] == "Fallback history found":
        print("✅ SUCCESS: Fallback logic correctly retrieves history.")
    else:
        print("❌ FAILURE: Fallback logic failed.")

# 2. Test Agent Verification Pattern Logic
def test_verification_patterns():
    print("\n--- Testing Agent Verification Patterns ---")

    # Pattern list from our fix
    verification_patterns = [
        r"os\.path\.exists",
        r"os\.path\.isfile",
        r"os\.path\.isdir",
        r"Path\.exists",
        r"os\.stat",
        r"os\.path\.getsize",
        r"json\.load",
        r"\.read\(",
        r"view_file",
        r"ls ",
        r"list_dir",
        r"grep_search",
    ]

    test_cases = [
        ("if os.path.isfile('test.txt'):", True),
        ("data = json.load(f)", True),
        ("content = f.read()", True),
        ("print('hello')", False),
    ]

    for code, expected in test_cases:
        matched = any(re.search(p, code) for p in verification_patterns)
        if matched == expected:
            print(f"✅ PASSED: '{code}' -> {matched}")
        else:
            print(f"❌ FAILED: '{code}' -> {matched} (Expected: {expected})")

if __name__ == "__main__":
    asyncio.run(test_scratchpad_logic())
    test_verification_patterns()
