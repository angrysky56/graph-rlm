"""Mock registry for centralized mock management in Graph-RLM tests.

Provides MockRegistry class for managing FalkorDB, LLM service, and external API mocks
with proper registration, retrieval, and reset capabilities.
"""

from unittest.mock import MagicMock, AsyncMock
from typing import Any, Optional


class MockRegistry:
    """Centralized mock management for FalkorDB, LLM services, and external APIs.

    This class provides a single source of truth for test mocks, ensuring
    consistent mock behavior across all tests and proper cleanup between runs.
    """

    def __init__(self) -> None:
        """Initialize an empty mock registry."""
        self._mocks: dict[str, Any] = {}
        self._reset_history: list[str] = []

    def register(self, name: str, mock: Any) -> None:
        """Register a mock by name.

        Args:
            name: Unique identifier for the mock
            mock: The mock object to register
        """
        self._mocks[name] = mock
        self._reset_history.append(f"Registered: {name}")

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a mock by name.

        Args:
            name: The unique identifier for the mock

        Returns:
            The registered mock object, or None if not found
        """
        return self._mocks.get(name)

    def reset(self) -> None:
        """Reset all mocks and clear history.

        Calls reset() on any mock that has this method, then clears
        the internal registry and reset history.
        """
        for mock in self._mocks.values():
            if hasattr(mock, "reset") and callable(mock.reset):
                mock.reset()
        self._reset_history.clear()
        self._mocks.clear()

    @property
    def falkordb(self) -> Optional[Any]:
        """Get FalkorDB mock.

        Returns:
            The FalkorDB mock registered under 'falkordb', or None
        """
        return self._mocks.get("falkordb")

    @property
    def llm(self) -> Optional[Any]:
        """Get LLM service mock.

        Returns:
            The LLM service mock registered under 'llm', or None
        """
        return self._mocks.get("llm")

    @property
    def external(self) -> Optional[Any]:
        """Get external API mock.

        Returns:
            The external API mock registered under 'external', or None
        """
        return self._mocks.get("external")


def create_falkordb_mock() -> MagicMock:
    """Create a FalkorDB client mock.

    Returns:
        A MagicMock configured to represent a FalkorDB client
    """
    mock_client = MagicMock()
    mock_client.session = MagicMock()
    mock_client.session.query = MagicMock(return_value=[])
    mock_client.close = MagicMock()
    return mock_client


def create_llm_mock(response: str = "Mocked LLM response") -> MagicMock:
    """Create an LLM service mock.

    Args:
        response: The response the mock should return

    Returns:
        A MagicMock configured to represent an LLM service
    """
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=MagicMock(content=response))
    mock_model.abatch = AsyncMock(
        return_value=[MagicMock(content=f"Response {i + 1}") for i in range(2)]
    )
    return mock_model


def create_external_api_mock(base_url: str = "https://api.example.com") -> MagicMock:
    """Create an external API mock.

    Args:
        base_url: The base URL for the API mock

    Returns:
        A MagicMock configured to represent an external API client
    """
    mock_client = MagicMock()
    mock_client.get = MagicMock()
    mock_client.post = MagicMock()
    mock_client.put = MagicMock()
    mock_client.delete = MagicMock()
    mock_client.base_url = base_url
    return mock_client
