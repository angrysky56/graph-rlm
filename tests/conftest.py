"""Pytest configuration and fixtures for Graph-RLM tests.

Provides async fixtures for testing, including event_loop and mock_registry
for proper async test lifecycle management.
"""

import pytest
import asyncio
from tests.mocking.mocks import MockRegistry

# Import mock fixtures from mocking modules
from tests.mocking.falkordb import (
    mock_falkordb,
    mock_registry_with_falkordb,
    async_mock_falkordb,
)
from tests.mocking.llm import (
    mock_llm_service,
    mock_registry_with_llm,
    async_mock_llm_service,
)
from tests.mocking.external import (
    mock_http_client,
    mock_registry_with_external,
    async_mock_http_client,
)


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """Create an instance of the default event loop for the test session.

    This fixture ensures proper async event loop lifecycle across all tests.
    The session scope means this is created once per test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_registry() -> MockRegistry:
    """Provide a fresh mock registry for each test.

    This fixture creates a new MockRegistry for each test function and
    resets it after the test completes. This ensures test isolation
    and prevents mock state leakage between tests.
    """
    registry = MockRegistry()
    yield registry
    # Reset the registry after each test to ensure clean state
    registry.reset()
