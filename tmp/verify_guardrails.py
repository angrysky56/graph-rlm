import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.core.guardrails import (
    EmpiricalGuard,
    ValidationError,
    extract_python_code,
    validate_thought_node,
)


def test_code_extraction():
    print("Testing code extraction...")
    text = "Here is some code:\n```python\nprint('hello')\n```\nAnd another block:\n```python\nawait rlm.done()\n```"
    extracted = extract_python_code(text)
    print(f"Extracted:\n{extracted}")
    assert "print('hello')" in extracted
    assert "await rlm.done()" in extracted
    assert "# --- RLM BLOCK SEPARATOR ---" in extracted

    truncated = "Streaming code...\n```python\nprint('incomplete'"
    extracted_truncated = extract_python_code(truncated)
    print(f"Extracted Truncated:\n{extracted_truncated}")
    assert "print('incomplete'" in extracted_truncated
    print("Code extraction tests passed!")


def test_syntax_validation():
    print("Testing syntax validation...")
    valid_code = "print('valid')"
    EmpiricalGuard.validate_syntax(valid_code)  # Should not raise

    invalid_code = "print('invalid'))"
    try:
        EmpiricalGuard.validate_syntax(invalid_code)
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        print(f"Caught expected ValidationError: {e}")
        assert "Syntax Error" in str(e)
    print("Syntax validation tests passed!")


def test_rlm_signature_validation():
    print("Testing RLM signature validation...")
    valid_code = "await rlm.done()"
    EmpiricalGuard.validate_rlm_signatures(valid_code)  # Should not raise

    invalid_code = "rlm.done()"
    try:
        EmpiricalGuard.validate_rlm_signatures(invalid_code)
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        print(f"Caught expected ValidationError: {e}")
        assert "Protocol Error" in str(e)
    print("RLM signature validation tests passed!")


def test_thought_node_validation():
    print("Testing thought node validation...")
    try:
        validate_thought_node(
            thought_id="test-1",
            prompt="",
            parent_id=None,
            session_id="session-1",
            root_session_id="root-1",
        )
        raise AssertionError("Should have raised ValidationError for empty prompt")
    except ValidationError as e:
        print(f"Caught expected ValidationError for empty prompt: {e}")
        assert "Empty" in str(e) or "Null-Context" in str(e)
    print("Thought node validation tests passed!")


if __name__ == "__main__":
    try:
        test_code_extraction()
        test_syntax_validation()
        test_rlm_signature_validation()
        test_thought_node_validation()
        print("\n✅ ALL GUARDRAIL TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
