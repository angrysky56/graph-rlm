"""Integration tests for circuit breaker protection in agent core.

Tests that verify:
- Protected LLM calls work correctly
- CircuitOpenError is raised when circuit is open
- Graceful degradation works as expected
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from graph_rlm.backend.src.core.services.circuit import (
    protected_llm_generate,
    protected_llm_with_fallback,
)
from graph_rlm.backend.src.core.circuit import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    llm_circuit,
    generate_correlation_id,
)


class TestProtectedLLMGenerate:
    """Tests for protected_llm_generate function."""

    @pytest.mark.asyncio
    async def test_protected_call_success(self):
        """Test successful LLM call through circuit breaker."""
        mock_generate = AsyncMock(return_value="Test response")

        with patch(
            "graph_rlm.backend.src.core.services.circuit.get_llm_service"
        ) as mock_get:
            mock_service = MagicMock()
            mock_service.generate = mock_generate
            mock_get.return_value = mock_service

            result = await protected_llm_generate(
                prompt="test prompt",
                system="test system",
                correlation_id="test-correlation",
            )

            assert result == "Test response"
            mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_open_raises_error(self):
        """Test that CircuitOpenError is raised when circuit is open."""
        # Force circuit to open state
        llm_circuit._state = CircuitState.OPEN
        llm_circuit._last_failure_time = None

        with pytest.raises(CircuitOpenError) as exc_info:
            await protected_llm_generate(
                prompt="test prompt", correlation_id="test-correlation"
            )

        assert exc_info.value.correlation_id == "test-correlation"

        # Reset circuit state
        llm_circuit._state = CircuitState.CLOSED


class TestProtectedLLMWithFallback:
    """Tests for protected_llm_with_fallback function."""

    @pytest.mark.asyncio
    async def test_fallback_on_circuit_open(self):
        """Test that fallback is used when circuit is open."""
        # Force circuit to open state
        llm_circuit._state = CircuitState.OPEN
        llm_circuit._last_failure_time = None

        result, was_fallback = await protected_llm_with_fallback(
            prompt="test prompt", fallback_message="Service unavailable"
        )

        assert was_fallback is True
        assert result == "Service unavailable"

        # Reset circuit state
        llm_circuit._state = CircuitState.CLOSED


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics."""

    def test_llm_circuit_metrics(self):
        """Test that LLM circuit provides metrics."""
        metrics = llm_circuit.get_metrics()

        assert "total_calls" in metrics
        assert "successful_calls" in metrics
        assert "failed_calls" in metrics
        assert "rejected_calls" in metrics
        assert "state" in metrics
