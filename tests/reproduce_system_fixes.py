
import re
from pathlib import Path


def test_regex_fixes():
    print("--- Testing Regex Fixes ---")

    # Old regex was: r"\{[a-zA-Z0-9_]+\}"
    # New regex is: r"\[(?:TODO|INSERT|FILL|MISSING).*?\]"
    placeholder_regex = r"\[(?:TODO|INSERT|FILL|MISSING).*?\]"

    test_cases = [
        ("{f_string_var}", False),
        ("[TODO]", True),
        ("[INSERT_CODE]", True),
        ("[FILL_HERE]", True),
        ("[MISSING_VALUE]", True),
        ("Just some text", False),
        ("{\"key\": \"val\"}", False),
    ]

    all_passed = True
    for text, expected in test_cases:
        match = bool(re.findall(placeholder_regex, text, re.IGNORECASE))
        if match == expected:
            print(f"✅ Pass: '{text}' -> {match}")
        else:
            print(f"❌ Fail: '{text}' -> {match} (Expected {expected})")
            all_passed = False

    # TODO markers check: m in thought_trace for m in ["[TODO]", "TODO:", "FIXME"]
    # Ellipsis should no longer trigger it.
    todo_markers = ["[TODO]", "TODO:", "FIXME"]

    test_todos = [
        ("...", False),
        ("FIXME: logic here", True),
        ("TODO: fix this", True),
        ("[TODO]", True),
        ("Wait...", False),
    ]

    for text, expected in test_todos:
        match = any(m in text for m in todo_markers)
        if match == expected:
            print(f"✅ Pass TODO: '{text}' -> {match}")
        else:
            print(f"❌ Fail TODO: '{text}' -> {match} (Expected {expected})")
            all_passed = False

    return all_passed

def test_summary_sanitization():
    print("\n--- Testing Summary Sanitization ---")

    # Simulate result/prompt
    text_with_code = "Here is the code:\n```python\nprint('hello')\n```\nAnd more text."

    # Sub matches: re.sub(r"```[\s\S]*?```", "", result).strip()
    sanitized = re.sub(r"```[\s\S]*?```", "", text_with_code).strip()

    expected = "Here is the code:\n\nAnd more text."
    if sanitized == expected:
        print(f"✅ Pass Sanitization: Matches expected output (code block removed).")
    else:
        print(f"❌ Fail Sanitization: '{sanitized}' (Expected '{expected}')")
        return False
    return True

if __name__ == "__main__":
    p1 = test_regex_fixes()
    p2 = test_summary_sanitization()

    if p1 and p2:
        print("\n🎉 ALL LOCAL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED.")
        exit(1)
