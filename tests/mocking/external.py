"""External API mock fixtures for Graph-RLM tests.

Provides pytest fixtures for mocking HTTP/API clients with support for
various HTTP request patterns (GET, POST, PUT, DELETE) and async operations.
"""

from unittest.mock import MagicMock, AsyncMock
from typing import Any, Optional, Callable

import pytest

from tests.mocking.mocks import MockRegistry


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create a mock HTTP client with standard request methods.

    Returns:
        MagicMock configured to represent an HTTP client with
        async and sync request methods.
    """
    mock_client = MagicMock()

    # HTTP methods returning AsyncMock for async usage
    mock_client.get = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": "get response"}),
            text="get response",
        )
    )
    mock_client.post = AsyncMock(
        return_value=MagicMock(
            status_code=201,
            json=MagicMock(return_value={"data": "post response"}),
            text="post response",
        )
    )
    mock_client.put = AsyncMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": "put response"}),
            text="put response",
        )
    )
    mock_client.delete = AsyncMock(
        return_value=MagicMock(
            status_code=204,
            json=MagicMock(return_value={}),
            text="",
        )
    )

    # Sync versions for compatibility
    mock_client.get_sync = MagicMock(
        return_value=MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": "get response"}),
        )
    )
    mock_client.post_sync = MagicMock(
        return_value=MagicMock(
            status_code=201,
            json=MagicMock(return_value={"data": "post response"}),
        )
    )

    # Client configuration
    mock_client.base_url = "https://api.example.com"
    mock_client.timeout = 30
    mock_client.headers = MagicMock()

    return mock_client


@pytest.fixture
def mock_registry_with_external(
    mock_registry: MockRegistry, mock_http_client: MagicMock
) -> MockRegistry:
    """Provide a mock registry with HTTP client pre-registered.

    Registers the HTTP client mock under the 'external' key in the registry,
    making it accessible via mock_registry.external.

    Args:
        mock_registry: The base mock registry fixture
        mock_http_client: The HTTP client mock fixture

    Returns:
        The mock registry with HTTP client registered
    """
    mock_registry.register("external", mock_http_client)
    return mock_registry


@pytest.fixture
def mock_http_client_with_handler() -> MagicMock:
    """Create an HTTP mock with configurable response handler.

    Returns:
        MagicMock with configurable response methods
    """
    mock_client = MagicMock()

    async def handle_request(method: str, url: str, **kwargs) -> MagicMock:
        """Default request handler that returns a generic success response."""
        return MagicMock(
            status_code=200,
            json=MagicMock(return_value={"method": method, "url": url}),
        )

    async def handle_get(url: str, **kwargs) -> MagicMock:
        return await handle_request("GET", url, **kwargs)

    async def handle_post(url: str, json: dict | None = None, **kwargs) -> MagicMock:
        return await handle_request("POST", url, json=json, **kwargs)

    async def handle_put(url: str, json: dict | None = None, **kwargs) -> MagicMock:
        return await handle_request("PUT", url, json=json, **kwargs)

    async def handle_delete(url: str, **kwargs) -> MagicMock:
        return await handle_request("DELETE", url, **kwargs)

    mock_client.get = AsyncMock(side_effect=handle_get)
    mock_client.post = AsyncMock(side_effect=handle_post)
    mock_client.put = AsyncMock(side_effect=handle_put)
    mock_client.delete = AsyncMock(side_effect=handle_delete)

    return mock_client
    """Create an HTTP mock with configurable response handler.

    Args:
        request_handler: Callable that takes method and params, returns (status_code, response_dict)

    Returns:
        MagicMock with response methods configured via handler
    """
    mock_client = MagicMock()

    async def handle_get(url: str, **kwargs) -> MagicMock:
        status, data = request_handler("GET", {"url": url, **kwargs})
        return MagicMock(status_code=status, json=MagicMock(return_value=data))

    async def handle_post(url: str, json: Optional[dict] = None, **kwargs) -> MagicMock:
        status, data = request_handler("POST", {"url": url, "json": json, **kwargs})
        return MagicMock(status_code=status, json=MagicMock(return_value=data))

    async def handle_put(url: str, json: Optional[dict] = None, **kwargs) -> MagicMock:
        status, data = request_handler("PUT", {"url": url, "json": json, **kwargs})
        return MagicMock(status_code=status, json=MagicMock(return_value=data))

    async def handle_delete(url: str, **kwargs) -> MagicMock:
        status, data = request_handler("DELETE", {"url": url, **kwargs})
        return MagicMock(status_code=status, json=MagicMock(return_value=data))

    mock_client.get = AsyncMock(side_effect=handle_get)
    mock_client.post = AsyncMock(side_effect=handle_post)
    mock_client.put = AsyncMock(side_effect=handle_put)
    mock_client.delete = AsyncMock(side_effect=handle_delete)

    return mock_client


@pytest.fixture
def mock_aiohttp_client() -> MagicMock:
    """Create an aiohttp-style mock client for async HTTP operations.

    Returns:
        MagicMock mimicking aiohttp ClientSession interface
    """
    mock_client = MagicMock()

    # aiohttp-style async methods
    mock_client._request = AsyncMock(
        return_value=MagicMock(
            status=200,
            json=AsyncMock(return_value={"data": "response"}),
            text=AsyncMock(return_value="response"),
        )
    )

    # Convenience methods
    mock_client.get = MagicMock()
    mock_client.post = MagicMock()
    mock_client.put = MagicMock()
    mock_client.delete = MagicMock()

    # Context manager support
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


def configure_http_mock(
    mock_client: MagicMock,
    method: str,
    response_data: Optional[dict] = None,
    status_code: int = 200,
) -> MagicMock:
    """Configure a specific HTTP method on an HTTP mock.

    Args:
        mock_client: The HTTP client mock to configure
        method: HTTP method to configure ('get', 'post', 'put', 'delete')
        response_data: Response data to return
        status_code: HTTP status code to return

    Returns:
        The configured mock client
    """
    response_mock = MagicMock(
        status_code=status_code,
        json=MagicMock(return_value=response_data or {}),
    )

    method_map = {
        "get": mock_client.get,
        "post": mock_client.post,
        "put": mock_client.put,
        "delete": mock_client.delete,
    }

    method_func = method_map.get(method.lower())
    if method_func:
        method_func.return_value = response_mock

    return mock_client


@pytest.fixture
async def async_mock_http_client() -> MagicMock:
    """Create an HTTP client mock optimized for async tests.

    Returns:
        MagicMock with full async HTTP support
    """
    mock_client = MagicMock()

    # Async methods
    mock_client.request = AsyncMock(
        return_value=MagicMock(
            status=200,
            json=AsyncMock(return_value={"async": "response"}),
        )
    )
    mock_client.get = AsyncMock(
        return_value=MagicMock(
            status=200,
            json=AsyncMock(return_value={"method": "GET", "async": True}),
        )
    )
    mock_client.post = AsyncMock(
        return_value=MagicMock(
            status=201,
            json=AsyncMock(return_value={"method": "POST", "async": True}),
        )
    )

    # Async context manager
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client
