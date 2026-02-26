import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add backend src to path
project_root = Path(__file__).parent.parent.parent.parent.resolve()
backend_src = project_root / "graph_rlm" / "backend" / "src"
sys.path.insert(0, str(backend_src))

from core.stealth import human_type, random_sleep, realistic_click, scroll_humanly


class TestStealthUtilities(unittest.IsolatedAsyncioTestCase):

    async def test_random_sleep(self):
        start = time.time()
        await random_sleep(100, 200)
        end = time.time()
        duration_ms = (end - start) * 1000
        print(f"random_sleep(100, 200) took {duration_ms:.2f}ms")
        self.assertGreaterEqual(duration_ms, 90)  # Buffer for event loop
        self.assertLessEqual(duration_ms, 300)

    async def test_human_type_simulation(self):
        # Mock Playwright Page
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.click = AsyncMock()

        async def mocked_type(*args, **kwargs):
            # Simulate the delay parameter in playwright.type
            delay = kwargs.get("delay", 0)
            await asyncio.sleep(delay / 1000.0)

        mock_page.type.side_effect = mocked_type

        start = time.time()
        await human_type(mock_page, "#input", "Hello", delay_min=50, delay_max=100)
        end = time.time()

        duration = end - start
        print(f"human_type('Hello') took {duration:.2f}s")

        # 5 chars * 50ms min = 0.25s min
        self.assertGreaterEqual(duration, 0.2)
        # Verify calls
        self.assertEqual(mock_page.type.call_count, 5)

    async def test_realistic_click_simulation(self):
        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        mock_page.query_selector = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.mouse = MagicMock()
        mock_page.mouse.move = AsyncMock()

        # Mock element bounding box
        mock_element = AsyncMock()
        mock_element.bounding_box = AsyncMock(
            return_value={"x": 10, "y": 10, "width": 100, "height": 50}
        )
        mock_page.query_selector.return_value = mock_element

        await realistic_click(mock_page, "#button")

        mock_page.mouse.move.assert_called_once()
        mock_page.click.assert_called_once_with("#button")

    async def test_skill_harness_injection(self):
        from mcp_integration.skill_harness import execute_skill_internal

        # Create a temporary skill file that uses the injected 'stealth'
        # BACKEND_ROOT in skill_harness is .../graph_rlm/backend
        backend_root = project_root / "graph_rlm" / "backend"
        skill_dir = backend_root / "skills" / "test-stealth-injection"
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        skill_code = """
import asyncio

async def test_stealth_injection():
    # 'stealth' should be injected automatically
    if not hasattr(stealth, "random_sleep"):
        return {"error": "stealth not injected"}

    await stealth.random_sleep(10, 20)
    return {"status": "success", "delay_func": str(type(stealth.random_sleep))}
"""
        script_path = scripts_dir / "test_stealth_injection.py"
        script_path.write_text(skill_code)

        try:
            # We mock the internal run flag handled by execute_skill_internal
            result = await execute_skill_internal("test-stealth-injection", {})
            print(f"Injection test result: {result}")
            self.assertEqual(result.get("status"), "success")
        finally:
            # Cleanup
            import shutil

            shutil.rmtree(skill_dir)


if __name__ == "__main__":
    unittest.main()
