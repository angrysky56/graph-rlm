
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "graph_rlm/backend/src"))

from core.guardrails import GuardrailError, validate_thought_node


def test_unified_guardrails():
    # Test Syntax Error
    print("Testing Syntax Error Guardrail...")
    bad_code_prompt = "Let's run some code:\n```python\nprint('Hello' \n```"
    try:
        validate_thought_node(
            thought_id="test_syntax",
            prompt=bad_code_prompt,
            parent_id=None,
            session_id="test",
            root_session_id="test"
        )
        print("FAIL: Syntax error should have been blocked.")
    except GuardrailError as e:
        print(f"PASS: Caught expected error: {e}")

    # Test Async Error
    print("\nTesting Async Error Guardrail...")
    bad_async_prompt = "Ending now:\n```python\nrlm.done(final_answer='done')\n```"
    try:
        validate_thought_node(
            thought_id="test_async",
            prompt=bad_async_prompt,
            parent_id=None,
            session_id="test",
            root_session_id="test"
        )
        print("FAIL: Missing await should have been blocked.")
    except GuardrailError as e:
        print(f"PASS: Caught expected error: {e}")

    # Test Valid Code
    print("\nTesting Valid Code...")
    valid_prompt = "Calculating:\n```python\nawait rlm.done(final_answer='42')\n```"
    try:
        validate_thought_node(
            thought_id="test_valid",
            prompt=valid_prompt,
            parent_id=None,
            session_id="test",
            root_session_id="test"
        )
        print("PASS: Valid code accepted.")
    except GuardrailError as e:
        print(f"FAIL: Valid code rejected unexpectedly: {e}")

if __name__ == "__main__":
    test_unified_guardrails()
