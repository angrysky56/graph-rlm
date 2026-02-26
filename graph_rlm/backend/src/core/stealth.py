"""
Stealth utilities for human-like browser interactions.
Provides agential grace to avoid bot detection and rate-limiting.
"""

import asyncio
import logging
import random
from typing import Any, Optional

logger = logging.getLogger("graph_rlm.core.stealth")


async def random_sleep(min_ms: int = 500, max_ms: int = 2000):
    """
    Perform an asynchronous sleep for a random duration between min_ms and max_ms.
    """
    # trunk-ignore(bandit/B311)
    duration = random.uniform(min_ms / 1000.0, max_ms / 1000.0)
    await asyncio.sleep(duration)


async def human_type(
    page: Any, selector: str, text: str, delay_min: int = 50, delay_max: int = 150
):
    """
    Type text into a selector one character at a time with randomized delays.

    Args:
        page: The playwright page object.
        selector: The CSS/playwright selector for the target element.
        text: The text to type.
        delay_min: Minimum delay between characters in ms.
        delay_max: Maximum delay between characters in ms.
    """
    # Wait for element to be visible and stable
    await page.wait_for_selector(selector)
    await page.click(selector)

    for char in text:
        # trunk-ignore(bandit/B311)
        await page.type(selector, char, delay=random.uniform(delay_min, delay_max))
        # Occasional longer pause for 'thinking'
        # trunk-ignore(bandit/B311)
        if random.random() < 0.1:
            # trunk-ignore(bandit/B311)
            await asyncio.sleep(random.uniform(0.2, 0.5))


async def realistic_click(page: Any, selector: str):
    """
    Perform a click with randomized mouse movement and pre-click delay.
    """
    await page.wait_for_selector(selector)
    element = await page.query_selector(selector)
    if not element:
        return

    box = await element.bounding_box()
    if box:
        # Move mouse to a random point within the element
        # trunk-ignore(bandit/B311)
        target_x = box["x"] + random.uniform(2, box["width"] - 2)
        # trunk-ignore(bandit/B311)
        target_y = box["y"] + random.uniform(2, box["height"] - 2)
        # trunk-ignore(bandit/B311)
        await page.mouse.move(target_x, target_y, steps=random.randint(5, 15))

    await random_sleep(200, 500)
    await page.click(selector)
    await random_sleep(100, 300)


async def scroll_humanly(page: Any, distance: Optional[int] = None):
    """
    Scroll the page in small chunks to simulate a human reading.
    """
    if distance is None:
        # trunk-ignore(bandit/B311)
        distance = random.randint(300, 700)

    # trunk-ignore(bandit/B311)
    steps = random.randint(3, 7)
    step_dist = distance // steps

    for _ in range(steps):
        # trunk-ignore(bandit/B311)
        await page.mouse.wheel(0, step_dist + random.randint(-50, 50))
        await random_sleep(200, 600)
