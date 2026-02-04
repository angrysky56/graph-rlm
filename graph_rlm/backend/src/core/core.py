"""
The Python REPL implementation that maintains state between executions.
Ported from local-repl-mcp with minimal changes.
"""

import ast
import io
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional, Tuple, cast

from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.repl.core")


class StreamingOutput(io.StringIO):
    """
    A custom writer that buffers line by line (or chunk by chunk)
    and invokes a callback immediately.
    """

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def write(self, s):
        # Write to internal buffer for final return value
        super().write(s)
        # Invoke callback immediately for streaming
        if self.callback:
            try:
                self.callback(s)
            except Exception as e:
                # Don't let callback errors break execution
                logger.error(f"Streaming callback error: {e}")
        return len(s)


class PythonREPL:
    """
    A stateful Python REPL implementation that maintains separate environment for each instance.
    """

    def __init__(self, repl_id: Optional[str] = None):
        # NOTE: nest_asyncio removed - incompatible with uvloop
        # execute() is async and always awaited, no nested loops needed.
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

    async def execute(
        self, code: str, output_callback=None, silent: bool = False
    ) -> Tuple[str, str, Any, bool]:
        """
        Execute Python code in the REPL and return stdout, stderr, result, and success status.
        Supports top-level await by wrapping code in an async function.

        Args:
            code: The Python code to execute
            output_callback: Optional callable(str) -> None for streaming stdout.
            silent: If True, suppresses trace logging (useful for internal logic like Axioms).

        Returns:
            Tuple of (stdout, stderr, result, exception_occurred)
        """
        import asyncio

        from .config import settings

        # Use StreamingOutput for capturing output
        stdout_capture = StreamingOutput(output_callback)
        stderr_capture = io.StringIO()
        result = None
        exception_occurred = False

        # Make sure code is a string to avoid issues
        if not isinstance(code, str):
            return ("", "Error: Code must be a string", None, True)

        # Skip empty code
        if not code.strip():
            return ("", "", None, False)

        if not silent:
            trace_action("REPL", "EXECUTE", result=code, tag="REPL")

        try:
            # Redirect stdout and stderr to our capture objects
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):

                # Parse the code into an AST
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    # Catch syntax error and return it cleanly via stderr
                    err = traceback.format_exc()
                    stderr_capture.write(err)
                    return (
                        stdout_capture.getvalue(),
                        stderr_capture.getvalue(),
                        None,
                        True,
                    )

                # Optimization: To support top-level await, we wrap the whole script in an async function if needed.
                # However, for simple expressions we prefer the eval() route for better result capture.

                # 1. Check if the block contains any 'await' calls
                has_await = any(isinstance(node, ast.Await) for node in ast.walk(tree))

                if has_await:
                    # WRAPPER PATTERN: Rewrite AST to be an async function
                    # This allows top-level await.
                    # async def __repl_async_wrapper__():
                    #     ... (original code) ...
                    #     return (last expression value)

                    # 1. Modify the last node to be a return if it's an expression
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_node = cast(ast.Expr, tree.body[-1])
                        tree.body[-1] = ast.Return(value=last_node.value)

                    # 2. Create the async function wrapper
                    wrapper_name = "__repl_async_wrapper__"
                    func_def = ast.AsyncFunctionDef(
                        name=wrapper_name,
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=tree.body,
                        decorator_list=[],
                        returns=None,
                        type_comment=None,
                    )

                    # 3. Create a new module with the function
                    new_module = ast.Module(body=[func_def], type_ignores=[])
                    ast.fix_missing_locations(new_module)

                    # 4. Exec the wrapper definition
                    # trunk-ignore(bandit/B102)
                    exec(
                        compile(new_module, filename="<string>", mode="exec"),
                        self.namespace,
                    )

                    # 5. Run the wrapper with potential timeout
                    timeout = settings.REPL_TIMEOUT
                    wrapper_coro = self.namespace[wrapper_name]()
                    try:
                        result = await asyncio.wait_for(wrapper_coro, timeout=timeout)
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"REPL Execution timed out after {timeout} seconds."
                        ) from None
                    finally:
                        # Clean up wrapper
                        self.namespace.pop(wrapper_name, None)

                elif tree.body and isinstance(tree.body[-1], ast.Expr):
                    # Standard Sync Expression Handling
                    last_node = cast(ast.Expr, tree.body[-1])
                    body_nodes = tree.body[:-1]

                    # Execute previous statements if any
                    if body_nodes:
                        module = ast.Module(body=body_nodes, type_ignores=[])
                        # trunk-ignore(bandit/B102)
                        exec(
                            compile(module, filename="<string>", mode="exec"),
                            self.namespace,
                        )

                    # Evaluate the last expression by wrapping in a return statement
                    return_stmt = ast.Return(value=last_node.value)
                    wrapper_func = ast.FunctionDef(
                        name="__eval_wrapper__",
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=[return_stmt],
                        decorator_list=[],
                        returns=None,
                        type_comment=None,
                    )
                    eval_module = ast.Module(body=[wrapper_func], type_ignores=[])
                    ast.fix_missing_locations(eval_module)
                    # trunk-ignore(bandit/B102)
                    exec(
                        compile(eval_module, filename="<string>", mode="exec"),
                        self.namespace,
                    )
                    result = self.namespace["__eval_wrapper__"]()
                    self.namespace.pop("__eval_wrapper__", None)
                else:
                    # Standard Sync Statement Handling
                    # trunk-ignore(bandit/B102)
                    exec(code, self.namespace)

        except Exception:
            # Catch any exceptions and add to stderr
            err = traceback.format_exc()
            stderr_capture.write(err)
            logger.error(f"REPL {self.repl_id} Execution Error: {err}")
            exception_occurred = True

        # Return the captured output and result
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # If stderr has content but NO exception was caught, it's likely just warnings or prints to stderr.
        # We rely on exception_occurred for "Hard Failures".

        if not silent:
            if stdout:
                trace_action("REPL", "STDOUT", result=stdout, tag="REPL")
            if stderr:
                # If exception occurred, log as Error, else Warning
                level = "error" if exception_occurred else "warning"
                trace_action("REPL", "STDERR", result=stderr, tag="REPL", level=level)
            if result is not None:
                trace_action("REPL", "RETURN", result=result, tag="REPL")

        return (stdout, stderr, result, exception_occurred)
