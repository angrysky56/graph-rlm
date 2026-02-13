"""Circuit breaker pattern implementation for Graph-RLM.

Provides async-aware circuit breaker protection against cascading failures
from external services (LLM, MCP). Integrates with Phase 1 exception
hierarchy and structured logging.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional

import structlog

from .exceptions.base import BaseGraphRLMError
from .exceptions.codes import ErrorCode

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states.

    Attributes:
        CLOSED: Normal operation, failures are counted toward threshold.
        OPEN: Circuit is open, calls are rejected immediately.
        HALF_OPEN: Testing recovery, limited calls allowed.
    """

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        timeout_seconds: Time in OPEN state before transitioning to HALF_OPEN.
        success_threshold: Number of successes in HALF_OPEN to close circuit.
    """

    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 3


class CircuitOpenError(BaseGraphRLMError):
    """Raised when circuit breaker is open and calls are rejected.

    Extends BaseGraphRLMError to provide structured error handling
    with correlation ID support and proper error codes.
    """

    def __init__(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        circuit_name: Optional[str] = None,
    ) -> None:
        """Initialize CircuitOpenError.

        Args:
            message: Human-readable error message describing the rejection.
            correlation_id: Request correlation ID for tracing.
            circuit_name: Name of the circuit that is open.
        """
        # Store circuit_name as instance attribute for proper access
        self.circuit_name = circuit_name
        super().__init__(
            message=message,
            error_code=ErrorCode.CORE_CIRCUIT_OPEN,
            correlation_id=correlation_id,
            circuit_name=circuit_name,
        )

    @property
    def http_status_code(self) -> int:
        """Return 503 for circuit breaker open errors."""
        return 503


# Correlation ID propagation utilities
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Get or create correlation ID for current execution context.

    Returns:
        Existing correlation ID from context or None if not set.
    """
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns:
        A short UUID-based correlation ID.
    """
    import uuid

    return str(uuid.uuid4())[:8]


def set_correlation_id(correlation_id: str) -> contextvars.Token:
    """Set correlation ID in current execution context.

    Args:
        correlation_id: The correlation ID to set.

    Returns:
        Token that can be used to reset the correlation ID.
    """
    return correlation_id_var.set(correlation_id)


def reset_correlation_id(token: contextvars.Token) -> None:
    """Reset correlation ID to previous value.

    Args:
        Token from set_correlation_id call.
    """
    correlation_id_var.reset(token)


class CircuitBreaker:
    """Async-aware circuit breaker with state machine.

    Provides protection against cascading failures by monitoring
    external service calls and opening the circuit when failures
    exceed a threshold.

    Features:
    - Async-aware with asyncio.Lock for thread-safe state transitions
    - Three-state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    - Structured logging via structlog
    - Correlation ID propagation
    - Metrics tracking for observability
    - Integration with Phase 1 exception hierarchy
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        logger_instance: Optional[structlog.stdlib.BoundLogger] = None,
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker.
            config: Circuit breaker configuration. Defaults to standard config.
            logger_instance: Logger instance for structured logging.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._logger = logger_instance or logger

        # State management
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[datetime] = None

        # Metrics tracking
        self._total_calls: int = 0
        self._successful_calls: int = 0
        self._failed_calls: int = 0
        self._rejected_calls: int = 0
        self._state_transitions: list[dict[str, Any]] = []

        # Thread-safe state transitions
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state with automatic OPEN -> HALF_OPEN transition.

        Returns:
            Current state, potentially transitioned from OPEN if timeout expired.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = datetime.now(timezone.utc) - self._last_failure_time
                if elapsed >= timedelta(seconds=self.config.timeout_seconds):
                    self._transition_to(CircuitState.HALF_OPEN)

        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count in CLOSED/HALF_OPEN state."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count in HALF_OPEN state."""
        return self._success_count

    def _transition_to(
        self, new_state: CircuitState, reason: Optional[str] = None
    ) -> None:
        """Transition to a new state with logging.

        Args:
            new_state: State to transition to.
            reason: Optional reason for the transition.
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state

        # Record state transition for metrics
        self._state_transitions.append(
            {
                "from_state": old_state.name,
                "to_state": new_state.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason
                or f"State transition: {old_state.name} -> {new_state.name}",
                "failure_count": self._failure_count,
                "success_count": self._success_count,
            }
        )

        # Log the transition
        log_args = {
            "circuit": self.name,
            "old_state": old_state.name,
            "new_state": new_state.name,
            "correlation_id": get_correlation_id(),
        }

        if reason:
            log_args["reason"] = reason

        if new_state == CircuitState.OPEN:
            self._logger.warning("circuit_opened", **log_args)
        elif new_state == CircuitState.HALF_OPEN:
            self._logger.info("circuit_half_open", **log_args)
        elif new_state == CircuitState.CLOSED:
            self._logger.info("circuit_closed", **log_args)

    async def call_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Result from the async function.

        Raises:
            CircuitOpenError: If circuit is open and call is rejected.
            Exception: If the function raises an exception.
        """
        async with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                self._rejected_calls += 1
                correlation_id = get_correlation_id()
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open, operation rejected",
                    correlation_id=correlation_id,
                    circuit_name=self.name,
                )

        correlation_id = get_correlation_id()
        self._logger.info(
            "circuit_call_start",
            circuit=self.name,
            correlation_id=correlation_id,
        )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            self._successful_calls += 1
            self._logger.info(
                "circuit_call_success",
                circuit=self.name,
                correlation_id=correlation_id,
            )
            return result

        except Exception as exc:
            self._failed_calls += 1
            await self._on_failure(exc)
            self._logger.error(
                "circuit_call_failure",
                circuit=self.name,
                error=str(exc),
                error_type=type(exc).__name__,
                correlation_id=correlation_id,
            )
            raise

    async def _on_success(self) -> None:
        """Handle successful call - update state accordingly."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._failure_count = 0
                    self._success_count = 0
                    self._transition_to(
                        CircuitState.CLOSED, "Success threshold reached"
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in CLOSED state
                # This allows recovery from brief failure streaks
                self._failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call - update state accordingly."""
        async with self._lock:
            self._last_failure_time = datetime.now(timezone.utc)

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"Failure threshold reached ({self._failure_count}/{self.config.failure_threshold})",
                    )
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN opens the circuit again
                self._success_count = 0
                self._transition_to(
                    CircuitState.OPEN,
                    f"Failure during HALF_OPEN state: {type(error).__name__}",
                )

    def get_metrics(self) -> dict[str, Any]:
        """Get current circuit metrics for observability.

        Returns:
            Dictionary containing all metrics.
        """
        return {
            "circuit_name": self.name,
            "state": self._state.name,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "rejected_calls": self._rejected_calls,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "success_threshold": self.config.success_threshold,
            "failure_threshold": self.config.failure_threshold,
            "timeout_seconds": self.config.timeout_seconds,
        }

    def get_state_transition_history(self) -> list[dict[str, Any]]:
        """Get history of state transitions.

        Returns:
            List of state transition records.
        """
        return self._state_transitions.copy()

    def reset_metrics(self) -> None:
        """Reset all metrics and state to initial values."""
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._rejected_calls = 0
        self._state_transitions = []

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, "
            f"state={self._state.name}, "
            f"failures={self._failure_count}/{self.config.failure_threshold})"
        )


# Service circuit instances - exported for service integration
llm_circuit = CircuitBreaker(
    name="llm_service",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=60.0,
        success_threshold=2,
    ),
)

# MCP server circuit - more lenient thresholds for connection-based services
mcp_circuit = CircuitBreaker(
    name="mcp_server",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=90.0,
        success_threshold=3,
    ),
)
