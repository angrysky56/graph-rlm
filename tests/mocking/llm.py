"""LLM service mock fixtures for Graph-RLM tests.

Provides pytest fixtures for mocking LLM services, particularly LangChain
ChatOpenAI interfaces with support for async invoke and batch operations.
"""

from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, Optional

import pytest

from tests.mocking.mocks import MockRegistry


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Create a mock LLM service with ChatOpenAI interface.

    Returns:
        MagicMock configured to represent an LLM service with
        async invoke and batch methods.
    """
    mock_model = MagicMock()

    # Async methods
    mock_model.ainvoke = AsyncMock(
        return_value=MagicMock(content="Mocked LLM response")
    )
    mock_model.abatch = AsyncMock(
        return_value=[
            MagicMock(content="Response 1"),
            MagicMock(content="Response 2"),
        ]
    )

    # Sync methods (for compatibility)
    mock_model.invoke = MagicMock(return_value=MagicMock(content="Mocked LLM response"))
    mock_model.batch = MagicMock(
        return_value=[
            MagicMock(content="Response 1"),
            MagicMock(content="Response 2"),
        ]
    )

    # Chat-specific interface
    mock_model._generate = MagicMock(return_value=MagicMock(generations=[[]]))
    mock_model.predict = MagicMock(return_value="Mocked prediction")
    mock_model.predict_messages = MagicMock(
        return_value=MagicMock(content="Mocked message")
    )

    # Model configuration (common properties)
    mock_model.model_name = "gpt-4"
    mock_model.temperature = 0.7
    mock_model.max_tokens = 2048
    mock_model.api_key = "sk-test-key"

    return mock_model


@pytest.fixture
def mock_registry_with_llm(
    mock_registry: MockRegistry, mock_llm_service: MagicMock
) -> MockRegistry:
    """Provide a mock registry with LLM service pre-registered.

    Registers the LLM service mock under the 'llm' key in the registry,
    making it accessible via mock_registry.llm.

    Args:
        mock_registry: The base mock registry fixture
        mock_llm_service: The LLM service mock fixture

    Returns:
        The mock registry with LLM service registered
    """
    mock_registry.register("llm", mock_llm_service)
    return mock_registry


@pytest.fixture
def mock_llm_service_with_responses() -> MagicMock:
    """Create an LLM service mock with sequential response support.

    Returns:
        MagicMock configured with sequential response support
    """
    mock_model = MagicMock()

    # Default responses for sequential calls
    default_responses = ["Response 1", "Response 2"]

    # Track call count for sequential responses
    call_count = 0

    def get_response(input_data: Any) -> MagicMock:
        nonlocal call_count
        response_index = min(call_count, len(default_responses) - 1)
        call_count += 1
        return MagicMock(content=default_responses[response_index])

    mock_model.ainvoke = AsyncMock(side_effect=get_response)
    mock_model.invoke = MagicMock(side_effect=get_response)

    return mock_model


@pytest.fixture
def langchain_patch() -> MagicMock:
    """Provide a patch context manager for LangChain imports.

    Usage:
        with langchain_patch():
            from langchain_openai import ChatOpenAI
            # ChatOpenAI is patched to return mock_llm_service

    Returns:
        MagicMock configured as a patch context manager
    """
    mock_llm = MagicMock()
    return patch("langchain_openai.ChatOpenAI", return_value=mock_llm)


def configure_llm_mock(
    mock_model: MagicMock,
    response: Optional[str] = None,
    responses: Optional[list[str]] = None,
) -> MagicMock:
    """Configure an LLM mock with specific response behavior.

    Args:
        mock_model: The LLM mock to configure
        response: Single response to return (overrides responses)
        responses: List of responses for sequential calls

    Returns:
        The configured mock model
    """
    if response is not None:
        mock_model.ainvoke.return_value = MagicMock(content=response)
        mock_model.invoke.return_value = MagicMock(content=response)

    if responses is not None:
        mock_model.ainvoke.side_effect = [MagicMock(content=r) for r in responses]
        mock_model.invoke.side_effect = [MagicMock(content=r) for r in responses]

    return mock_model


@pytest.fixture
async def async_mock_llm_service() -> MagicMock:
    """Create an LLM service mock optimized for async tests.

    Returns:
        MagicMock with enhanced async configuration
    """
    mock_model = MagicMock()

    # Enhanced async support
    mock_model.ainvoke = AsyncMock(return_value=MagicMock(content="Async response"))
    mock_model.abatch = AsyncMock(
        return_value=[MagicMock(content=f"Batch response {i}") for i in range(3)]
    )

    # Async stream support
    mock_model._astream = AsyncMock(
        return_value=iter(
            [
                MagicMock(chunk="Chunk 1"),
                MagicMock(chunk="Chunk 2"),
            ]
        )
    )

    return mock_model
