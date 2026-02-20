import re


def test_regex_fixes():
    print("--- Testing Regex Fixes ---")
    placeholder_regex = r"\[(?:TODO|INSERT|FILL|MISSING).*?\]"

    test_cases = [
        ("{f_string_var}", False),
        ("[TODO]", True),
        ("[INSERT_CODE]", True),
        ("[FILL_HERE]", True),
        ("[MISSING_VALUE]", True),
        ("Just some text", False),
        ('{"key": "val"}', False),
    ]

    all_passed = True
    for text, expected in test_cases:
        match = bool(re.findall(placeholder_regex, text, re.IGNORECASE))
        if match == expected:
            print(f"✅ Pass: '{text}' -> {match}")
        else:
            print(f"❌ Fail: '{text}' -> {match} (Expected {expected})")
            all_passed = False

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


def test_camel_case_regex():
    print("\n--- Testing CamelCase Regex ---")
    regex = r"\b([A-Z][a-z]+[A-Z][a-zA-Z0-9_]*)\b"

    test_cases = [
        ("CamelCase", True),
        ("PascalCase", True),
        ("Simple", False),
        ("SCREAMING", False),
        ("snake_case", False),
        ("GraphRLM", True),
        ("The", False),
        ("If", False),
        ("When", False),
    ]

    all_passed = True
    for text, expected in test_cases:
        match = bool(re.search(regex, text))
        if match == expected:
            print(f"✅ Pass: '{text}' -> {match}")
        else:
            print(f"❌ Fail: '{text}' -> {match} (Expected {expected})")
            all_passed = False
    return all_passed


def test_path_compressor():
    print("\n--- Testing Path Compressor ---")

    def compress(summary):
        if len(summary) > 80 and "/" in summary:
            parts = summary.split("/")
            if len(parts) > 3:
                return f"{parts[0]}/.../{parts[-2]}/{parts[-1]}"
        return summary[:120].replace("\n", " ")

    long_path = "/home/ty/Repositories/ai_workspace/graph-rlm/knowledge_base/outputs/deep_research_agents_core_findings.md"
    compressed = compress(long_path)
    expected = "/.../outputs/deep_research_agents_core_findings.md"

    if compressed == expected:
        print(f"✅ Pass Compressor: {compressed}")
    else:
        print(f"❌ Fail Compressor: {compressed} (Expected {expected})")
        return False
    return True


def test_thimac_failure_extraction():
    print("\n--- Testing Thimac Failure Extraction ---")

    def extract(status, prompt, result):
        if status in ["failed", "error", "rejected"]:
            combined = prompt + " " + result
            err_match = re.search(
                r"(Error:.*?|Exception:.*?|DREAMER REJECTION:.*?)(?=\n|$)", combined
            )
            if err_match:
                return err_match.group(1)[:100]
            return f"[{status.upper()}] Action failed or rejected."
        return "Not a failure"

    case1 = extract("failed", "Action here", "Trace: Error: File not found")
    if case1 == "Error: File not found":
        print(f"✅ Pass Case 1: {case1}")
    else:
        print(f"❌ Fail Case 1: {case1}")
        return False

    case2 = extract(
        "rejected",
        "Code here",
        "DREAMER REJECTION: Trace contradiction detected\nMore details",
    )
    if case2 == "DREAMER REJECTION: Trace contradiction detected":
        print(f"✅ Pass Case 2: {case2}")
    else:
        print(f"❌ Fail Case 2: {case2}")
        return False

    return True


def test_summary_sanitization():
    print("\n--- Testing Summary Sanitization ---")
    text_with_code = "Here is the code:\n```python\nprint('hello')\n```\nAnd more text."
    sanitized = re.sub(r"```[\s\S]*?```", "", text_with_code).strip()

    expected = "Here is the code:\n\nAnd more text."
    if sanitized == expected:
        print(f"✅ Pass Sanitization: Matches expected output.")
    else:
        print(f"❌ Fail Sanitization: '{sanitized}' (Expected '{expected}')")
        return False
    return True


def test_gestalt_formatting():
    print("\n--- Testing Gestalt Formatting ---")

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "-"

    row = {
        "repe_shakiness": -0.12,
        "repe_confluence": 0.05,
        "repe_evasion": -0.21,
        "repe_freedom": 0.33,
    }

    repe_str = f"S:{fmt(row.get('repe_shakiness'))} C:{fmt(row.get('repe_confluence'))} E:{fmt(row.get('repe_evasion'))} F:{fmt(row.get('repe_freedom'))}"
    expected = "S:-0.1 C:0.1 E:-0.2 F:0.3"

    if repe_str == expected:
        print(f"✅ Pass Gestalt: {repe_str}")
        return True
    else:
        print(f"❌ Fail Gestalt: {repe_str} (Expected {expected})")
        return False


if __name__ == "__main__":
    tests = [
        test_regex_fixes(),
        test_summary_sanitization(),
        test_camel_case_regex(),
        test_path_compressor(),
        test_thimac_failure_extraction(),
        test_gestalt_formatting(),
    ]

    if all(tests):
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED.")
        exit(1)
