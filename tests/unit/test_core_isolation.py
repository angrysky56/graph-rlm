import pytest

from graph_rlm.backend.src.core.core import PythonREPL


@pytest.mark.asyncio
async def test_dreamer_mock_injection():
    """
    Reproduces the Dreamer/Sheaf usage pattern:
    1. Instantiate PythonREPL
    2. Inject mocks (e.g. UniversalAsyncMock) into namespace
    3. Execute code that uses those mocks
    """
    repl = PythonREPL(repl_id="test_dreamer_isolation")

    # Define the Mock class (similar to Dreamer)
    class UniversalAsyncMock:
        def __getattr__(self, name):
            return UniversalAsyncMock()

        def __call__(self, *args, **kwargs):
            async def _dummy():
                return "MOCK_RESULT"

            return _dummy()

        def __repr__(self):
            return "<UniversalAsyncMock>"

    # Inject into namespace
    repl.namespace.update(
        {
            "rlm": UniversalAsyncMock(),
            "mcp": UniversalAsyncMock(),
            "context_var": "test_value",
        }
    )

    # Code to execute
    code = """
import asyncio
result = await rlm.query("hello")
print(f"Result: {result}")
print(f"Context: {context_var}")
"""

    stdout, stderr, result, is_failed = await repl.execute(code)

    assert not is_failed, f"Execution failed with stderr: {stderr}"
    assert "Result: MOCK_RESULT" in stdout
    assert "Context: test_value" in stdout


@pytest.mark.asyncio
async def test_simple_math_execution():
    """Verifies basic execution still works."""
    repl = PythonREPL()
    stdout, stderr, result, is_failed = await repl.execute("print(2 + 2)")
    assert not is_failed
    assert "4" in stdout
