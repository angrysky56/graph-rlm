"""FalkorDB mock fixtures for Graph-RLM tests.

Provides pytest fixtures for creating and registering FalkorDB client mocks
with proper session-based query interface support.
"""

from unittest.mock import MagicMock, AsyncMock
from typing import Any, Optional

import pytest

from tests.mocking.mocks import MockRegistry


@pytest.fixture
def mock_falkordb() -> MagicMock:
    """Create a mock FalkorDB client with session interface.

    Returns:
        MagicMock configured to represent a FalkorDB client with session
        that supports both sync and async query operations.
    """
    mock_client = MagicMock()
    mock_client.session = MagicMock()

    # Setup sync query interface
    mock_client.session.query = MagicMock(return_value=[])

    # Setup async query interface using AsyncMock
    mock_client.session.query_async = AsyncMock(return_value=[])

    # Close method
    mock_client.close = MagicMock()

    # Connection management
    mock_client.connect = MagicMock()
    mock_client.is_connected = MagicMock(return_value=True)

    return mock_client


@pytest.fixture
def mock_registry_with_falkordb(
    mock_registry: MockRegistry, mock_falkordb: MagicMock
) -> MockRegistry:
    """Provide a mock registry with FalkorDB client pre-registered.

    Registers the FalkorDB mock under the 'falkordb' key in the registry,
    making it accessible via mock_registry.falkordb.

    Args:
        mock_registry: The base mock registry fixture
        mock_falkordb: The FalkorDB client mock fixture

    Returns:
        The mock registry with FalkorDB registered
    """
    mock_registry.register("falkordb", mock_falkordb)
    return mock_registry


@pytest.fixture
async def async_mock_falkordb() -> MagicMock:
    """Create a FalkorDB mock optimized for async test scenarios.

    This fixture is useful for tests that specifically need to test
    async interaction patterns with the FalkorDB client.

    Returns:
        MagicMock with async methods properly configured
    """
    mock_client = MagicMock()
    mock_client.session = MagicMock()

    # Async session methods
    mock_client.session.execute = AsyncMock(return_value=MagicMock())
    mock_client.session.run = AsyncMock(return_value=MagicMock())
    mock_client.session.query_async = AsyncMock(return_value=MagicMock())

    # Connection lifecycle
    mock_client.connect = AsyncMock()
    mock_client.close = AsyncMock()
    mock_client.is_connected = AsyncMock(return_value=True)

    return mock_client


def configure_falkordb_mock(
    mock_client: MagicMock, query_result: Optional[list[Any]] = None
) -> MagicMock:
    """Configure a FalkorDB mock with custom query behavior.

    Args:
        mock_client: The FalkorDB mock client to configure
        query_result: Optional custom result for query operations

    Returns:
        The configured mock client
    """
    if query_result is not None:
        mock_client.session.query.return_value = query_result
        mock_client.session.query_async.return_value = query_result

    return mock_client
