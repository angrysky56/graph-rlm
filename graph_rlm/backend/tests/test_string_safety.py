import unittest

# Raw string test case
RAW_LATEX = r"\equiv \neg \forall \exists"
NORMAL_STRING = "\\equiv \\neg \\forall \\exists"


class TestStringSafety(unittest.TestCase):
    def test_raw_string_latex(self):
        # This should execute without SyntaxWarning if handled correctly by the parser
        self.assertEqual(RAW_LATEX, NORMAL_STRING)
        print(f"Raw LaTeX: {RAW_LATEX}")
        print(f"Normal String: {NORMAL_STRING}")

    def test_regex_safety(self):
        # Test backslashes in regex-like strings
        raw_regex = r"\d+\.\d+"
        normal_regex = "\\d+\\.\\d+"
        self.assertEqual(raw_regex, normal_regex)
        print(f"Raw Regex: {raw_regex}")


if __name__ == "__main__":
    print("Testing string safety and escape sequences...")
    unittest.main()
