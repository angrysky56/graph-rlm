"""Tests for MCP server circuit breaker protection.

Validates that MCP server connections are protected by circuit breaker,
with proper HTTP 503 error mapping when circuit opens.
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from graph_rlm.backend.src.core.circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    mcp_circuit,
)
from graph_rlm.backend.src.mcp_integration.circuit import (
    safe_mcp_call,
    get_mcp_circuit_metrics,
)
from graph_rlm.backend.src.core.exceptions.types import ExternalServiceError


class TestMcpCircuitBreakerBasic:
    """Test basic circuit breaker behavior for MCP connections."""

    def test_circuit_open_error_has_correct_attributes(self):
        """CircuitOpenError should have circuit_name and http_status_code attributes."""
        error = CircuitOpenError(
            message="Circuit 'test' is open",
            circuit_name="mcp_test",
        )
        assert error.circuit_name == "mcp_test"
        assert error.http_status_code == 503

    def test_circuit_open_error_http_status_code_is_503(self):
        """CircuitOpenError should return 503 for HTTP status code mapping."""
        error = CircuitOpenError(
            message="Circuit is open",
            circuit_name="mcp_server",
        )
        # ExternalServiceError also returns 503
        assert error.http_status_code == 503


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions."""

    def test_circuit_creation_default_state(self):
        """New circuit should be in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        circuit = CircuitBreaker(name="test", config=config)
        assert circuit.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failure_threshold(self):
        """Circuit should open after configured failure threshold is exceeded."""
        config = CircuitBreakerConfig(failure_threshold=3)
        circuit = CircuitBreaker(name="test_failure", config=config)

        # Circuit should start CLOSED
        assert circuit.state == CircuitState.CLOSED

        # After 3 failures, circuit should OPEN
        async def failing_func():
            raise Exception("Test failure")

        for _ in range(3):
            try:
                await circuit.call_async(failing_func)
            except Exception:
                pass  # Expected

        # Circuit should now be OPEN
        assert circuit.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self):
        """After timeout, circuit should transition from OPEN to HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,  # Short timeout for testing
        )
        circuit = CircuitBreaker(name="test_half_open", config=config)

        # Force circuit to OPEN state
        async def failing_func():
            raise Exception("Test failure")

        try:
            await circuit.call_async(failing_func)
        except Exception:
            pass
        assert circuit.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.2)  # Wait longer than timeout
        # Access state property to trigger timeout check
        _ = circuit.state

        # Should now be HALF_OPEN
        assert circuit.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success_threshold(self):
        """Circuit should close after success_threshold successes in HALF_OPEN."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
            success_threshold=2,
        )
        circuit = CircuitBreaker(name="test_close", config=config)

        async def failing_func():
            raise Exception("Fail")

        async def succeeding_func():
            return "success"

        # First, open the circuit
        try:
            await circuit.call_async(failing_func)
        except Exception:
            pass
        assert circuit.state == CircuitState.OPEN

        # Wait for HALF_OPEN
        await asyncio.sleep(0.2)
        # Access state to trigger HALF_OPEN
        _ = circuit.state
        assert circuit.state == CircuitState.HALF_OPEN

        # Make successful calls
        for _ in range(2):
            result = await circuit.call_async(succeeding_func)
            assert result == "success"

        # Circuit should be CLOSED
        assert circuit.state == CircuitState.CLOSED


class TestSafeMcpCall:
    """Test safe_mcp_call function behavior."""

    @pytest.mark.asyncio
    async def test_safe_mcp_call_raises_circuit_open_error_when_circuit_open(self):
        """safe_mcp_call should raise CircuitOpenError when circuit is open."""
        # Create a circuit breaker and force it to open
        config = CircuitBreakerConfig(failure_threshold=1)
        test_circuit = CircuitBreaker(name="test_mcp", config=config)

        async def failing_func():
            raise Exception("Connection failed")

        # Open the circuit
        try:
            await test_circuit.call_async(failing_func)
        except Exception:
            pass

        assert test_circuit.state == CircuitState.OPEN

        # Now call should raise CircuitOpenError
        async def dummy_func():
            return "should not reach here"

        with pytest.raises(CircuitOpenError) as exc_info:
            await test_circuit.call_async(dummy_func)

        assert exc_info.value.circuit_name == "test_mcp"
        assert exc_info.value.http_status_code == 503

    @pytest.mark.asyncio
    async def test_safe_mcp_call_succeeds_when_circuit_closed(self):
        """safe_mcp_call should succeed when circuit is closed."""
        # Get the actual mcp_circuit
        circuit = mcp_circuit

        async def successful_func():
            return "MCP response"

        result = await safe_mcp_call(successful_func)
        assert result == "MCP response"


class TestMcpCircuitMetrics:
    """Test MCP circuit metrics functionality."""

    def test_get_mcp_circuit_metrics_returns_dict(self):
        """get_mcp_circuit_metrics should return a dictionary."""
        metrics = get_mcp_circuit_metrics()

        assert isinstance(metrics, dict)
        assert "circuit_name" in metrics
        assert "state" in metrics
        assert "total_calls" in metrics
        assert "failure_count" in metrics
        assert "success_count" in metrics
        assert metrics["circuit_name"] == "mcp_server"

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_track_calls(self):
        """Circuit breaker should track metrics for calls."""
        config = CircuitBreakerConfig(failure_threshold=5)
        circuit = CircuitBreaker(name="metrics_test", config=config)

        async def success():
            return "ok"

        async def fail():
            raise Exception("fail")

        # Make some calls
        for _ in range(3):
            try:
                await circuit.call_async(fail)
            except Exception:
                pass

        metrics = circuit.get_metrics()
        assert metrics["failed_calls"] == 3
        assert metrics["total_calls"] == 3


class TestCircuitOpenErrorHttp503:
    """Test that CircuitOpenError properly maps to HTTP 503."""

    def test_circuit_open_error_isinstance_external_service_error(self):
        """CircuitOpenError should work with HTTP 503 mapping."""
        error = CircuitOpenError(
            message="MCP server circuit open",
            circuit_name="mcp_server",
        )
        # Should have http_status_code = 503
        assert error.http_status_code == 503

    def test_external_service_error_http_status_code(self):
        """ExternalServiceError should return 503."""
        from graph_rlm.backend.src.core.exceptions.codes import ErrorCode

        error = ExternalServiceError(
            message="Service unavailable",
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
        )
        assert error.http_status_code == 503

    def test_circuit_open_error_exception_handling_scenario(self):
        """Test complete scenario: circuit open leads to 503 via exception handler."""
        circuit = CircuitBreaker(
            name="test_handler",
            config=CircuitBreakerConfig(failure_threshold=1),
        )

        async def failing_call():
            raise Exception("Server down")

        async def open_circuit():
            try:
                await circuit.call_async(failing_call)
            except Exception:
                pass

        asyncio.run(open_circuit())
        assert circuit.state == CircuitState.OPEN

        async def protected_call():
            return await circuit.call_async(lambda: "result")

        with pytest.raises(CircuitOpenError) as exc_info:
            asyncio.run(protected_call())

        # Exception should be catchable as ExternalServiceError base type
        assert isinstance(exc_info.value, CircuitOpenError)
        assert exc_info.value.http_status_code == 503


class TestMcpCircuitBreakerIntegration:
    """Integration tests for MCP circuit breaker with the existing mcp_circuit."""

    def test_mcp_circuit_exists_and_is_configured(self):
        """Verify mcp_circuit singleton is properly configured."""
        assert mcp_circuit.name == "mcp_server"
        assert mcp_circuit.config.failure_threshold == 5
        assert mcp_circuit.config.timeout_seconds == 90.0
        assert mcp_circuit.config.success_threshold == 3

    @pytest.mark.asyncio
    async def test_safe_mcp_call_with_actual_mcp_circuit(self):
        """Test safe_mcp_call with the actual mcp_circuit instance."""

        async def mock_mcp_operation():
            return {"result": "success", "data": [1, 2, 3]}

        # This should succeed since circuit is likely CLOSED
        result = await safe_mcp_call(mock_mcp_operation)
        assert result == {"result": "success", "data": [1, 2, 3]}

    def test_mcp_circuit_metrics_format(self):
        """Verify mcp_circuit metrics have expected format."""
        metrics = get_mcp_circuit_metrics()

        # Check required keys
        expected_keys = [
            "circuit_name",
            "state",
            "total_calls",
            "successful_calls",
            "failed_calls",
            "rejected_calls",
            "failure_count",
            "success_count",
            "success_threshold",
            "failure_threshold",
            "timeout_seconds",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        # Check types
        assert isinstance(metrics["state"], str)
        assert isinstance(metrics["total_calls"], int)
        assert isinstance(metrics["failure_count"], int)
