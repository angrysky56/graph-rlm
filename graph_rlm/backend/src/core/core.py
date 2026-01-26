"""
The Python REPL implementation that maintains state between executions.
Ported from local-repl-mcp with minimal changes.
"""

import ast
import io
import logging
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional, Tuple, cast

logger = logging.getLogger("graph_rlm.repl.core")


class StreamingOutput(io.TextIOBase):
    """
    A custom writer that buffers line by line (or chunk by chunk)
    and invokes a callback immediately.
    """

    def __init__(self, callback):
        self.callback = callback
        self.buffer = io.StringIO()

    def write(self, s):
        # Write to internal buffer for final return value
        self.buffer.write(s)
        # Invoke callback immediately for streaming
        if self.callback:
            try:
                self.callback(s)
            except Exception as e:
                # Don't let callback errors break execution
                logger.error(f"Streaming callback error: {e}")
        return len(s)

    def getvalue(self):
        return self.buffer.getvalue()


def execute(self, code: str, output_callback=None) -> Tuple[str, str, Any]:
    """
    Execute Python code in the REPL and return stdout, stderr, and the result.

    Args:
        code: The Python code to execute
        output_callback: Optional callable(str) -> None for streaming stdout.

    Returns:
        Tuple of (stdout, stderr, result)
    """
    # Use StreamingOutput for capturing output
    stdout_capture = StreamingOutput(output_callback)
    stderr_capture = io.StringIO()
    result = None

    # Make sure code is a string to avoid issues
    if not isinstance(code, str):
        return ("", "Error: Code must be a string", None)

    # Skip empty code
    if not code.strip():
        return ("", "", None)

    try:
        # Redirect stdout and stderr to our capture objects
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):

            # Parse the code into an AST
            try:
                tree = ast.parse(code)
            except SyntaxError:
                # If it's a syntax error, run it to get the standard traceback
                # trunk-ignore(bandit/B102)
                exec(code, self.namespace, self.namespace)
                return stdout_capture.getvalue(), stderr_capture.getvalue(), None

            # Check if the last node is an expression
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # It's an expression! We want to capture its value.
                last_node = cast(ast.Expr, tree.body[-1])
                body_nodes = tree.body[:-1]

                # Execute previous statements if any
                if body_nodes:
                    module = ast.Module(body=body_nodes, type_ignores=[])
                    # trunk-ignore(bandit/B102)
                    exec(
                        compile(module, filename="<string>", mode="exec"),
                        self.namespace,
                        self.namespace,
                    )

                # Evaluate the last expression
                expr = ast.Expression(body=last_node.value)
                # trunk-ignore(bandit/B307)
                result = eval(
                    compile(expr, filename="<string>", mode="eval"),
                    self.namespace,
                    self.namespace,
                )

                # If the result is not None, print it to stdout so it streams
                # But typically REPLs only return it.
                # If we want it to stream, we might need to print it?
                # Standard behavior: return it in result tuple.
            else:
                # Last node is not an expression (e.g. assignment, function def)
                # Just exec the whole thing
                # trunk-ignore(bandit/B102)
                exec(code, self.namespace, self.namespace)

    except Exception:
        # Catch any exceptions and add to stderr
        err = traceback.format_exc()
        stderr_capture.write(err)
        logger.error(f"REPL {self.repl_id} Execution Error: {err}")

    # Return the captured output and result
    return (stdout_capture.getvalue(), stderr_capture.getvalue(), result)


class PythonREPL:
    """
    A stateful Python REPL implementation that maintains separate environment for each instance.
    """

    def __init__(self, repl_id: Optional[str] = None):
        self.repl_id = repl_id or str(uuid.uuid4())
        logger.debug(f"Initializing REPL {self.repl_id}")
        # Initialize a single namespace for environment
        # This is crucial for recursive functions to work properly
        self.namespace: Dict[str, Any] = {"__builtins__": __builtins__}

        # Inject standard libraries for convenience
        try:
            import json
            import math
            import os
            import random
            import re
            import sys
            import time

            self.namespace.update(
                {
                    "os": os,
                    "sys": sys,
                    "json": json,
                    "time": time,
                    "math": math,
                    "re": re,
                    "random": random,
                }
            )
        except ImportError:
            pass

    execute = execute
