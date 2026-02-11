"""
This module provides a validator for ensuring string integrity within RLM reports
and code blocks, specifically checking for unescaped backslashes in non-raw strings.
"""

import ast
from typing import List


def raw_string_integrity_validator(source_code: str) -> bool:
    """
    Scans source code to ensure any string containing backslashes is either
    marked as a raw string or properly escaped.

    Args:
        source_code: The Python source code or report content to validate.

    Returns:
        bool: True if all backslashes are in raw strings or escaped, False otherwise.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # If the code cannot be parsed, it is invalid by default
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # We look for backslashes in the evaluated string value
            if "\\" in node.value:
                # Find the segment of source text corresponding to this string
                # We check the original source to see if it was prefixed with 'r'
                # Note: lineno is 1-indexed
                lines = source_code.splitlines()
                if node.lineno is not None and node.lineno <= len(lines):
                    line_segment = lines[node.lineno - 1]
                    # Check if the string literal in source starts with 'r' or 'R'
                    # This is a heuristic check for the raw prefix in the source
                    # A more robust check involves tokenizing, but this covers standard cases.
                    if not any(prefix in line_segment.lower()
                               for prefix in ["r'", 'r"', "f'", 'f"']):
                        # If a backslash exists but no raw prefix and it wasn't
                        # caught by valid escaping during parsing, it's a violation
                        return False

    return True
