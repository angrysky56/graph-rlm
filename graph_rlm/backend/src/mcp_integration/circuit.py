"""Circuit-breaker-protected MCP integration wrapper.

Provides a wrapper around MCP server operations that integrates with the
circuit breaker pattern for resilience against MCP connection failures.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import structlog

from graph_rlm.backend.src.core.circuit import (
    CircuitOpenError,
    get_correlation_id,
    generate_correlation_id,
    mcp_circuit,
    set_correlation_id,
    reset_correlation_id,
)

logger = structlog.get_logger(__name__)


async def safe_mcp_call(
    func: Callable[..., Any],
    *args: Any,
    correlation_id: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Execute MCP operation through circuit breaker with correlation tracking.

    Wraps any MCP server function with circuit breaker protection.
    Handles CircuitOpenError gracefully with proper logging.

    Args:
        func: Async MCP function to execute.
        *args: Positional arguments for the function.
        correlation_id: Correlation ID for tracing. Generated if not provided.
        **kwargs: Keyword arguments for the function.

    Returns:
        MCP operation result.

    Raises:
        CircuitOpenError: If circuit is open and call is rejected.
        Exception: If MCP operation fails.
    """
    # Set or generate correlation ID
    if correlation_id is None:
        correlation_id = get_correlation_id() or generate_correlation_id()

    token = set_correlation_id(correlation_id)

    try:
        logger.info(
            "mcp_call_start",
            correlation_id=correlation_id,
            function=func.__name__ if hasattr(func, "__name__") else str(func),
        )

        # Execute through circuit breaker
        result = await mcp_circuit.call_async(func, *args, **kwargs)

        logger.info(
            "mcp_call_success",
            correlation_id=correlation_id,
            function=func.__name__ if hasattr(func, "__name__") else str(func),
        )

        return result

    except CircuitOpenError:
        logger.warning(
            "mcp_call_circuit_open",
            correlation_id=correlation_id,
            circuit=mcp_circuit.name,
        )
        raise

    finally:
        reset_correlation_id(token)


async def safe_mcp_connection(
    connect_func: Callable[..., Any],
    disconnect_func: Optional[Callable[..., Any]] = None,
    correlation_id: Optional[str] = None,
) -> Any:
    """Execute MCP connection through circuit breaker with proper lifecycle.

    Handles connection establishment with circuit breaker protection.
    Ensures proper cleanup even if connection fails.

    Args:
        connect_func: Async function to establish MCP connection.
        disconnect_func: Optional async function to disconnect.
        correlation_id: Correlation ID for tracing.

    Returns:
        Connection result.

    Raises:
        CircuitOpenError: If circuit is open.
        Exception: If connection fails.
    """
    if correlation_id is None:
        correlation_id = get_correlation_id() or generate_correlation_id()

    token = set_correlation_id(correlation_id)

    try:
        logger.info(
            "mcp_connection_start",
            correlation_id=correlation_id,
        )

        # Establish connection through circuit breaker
        result = await mcp_circuit.call_async(connect_func)

        logger.info(
            "mcp_connection_success",
            correlation_id=correlation_id,
        )

        return result

    except CircuitOpenError:
        logger.warning(
            "mcp_connection_circuit_open",
            correlation_id=correlation_id,
        )
        raise

    except Exception as e:
        logger.error(
            "mcp_connection_failed",
            correlation_id=correlation_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise

    finally:
        # Always attempt cleanup if disconnect function provided
        if disconnect_func is not None:
            try:
                await disconnect_func()
                logger.info(
                    "mcp_disconnect_complete",
                    correlation_id=correlation_id,
                )
            except Exception as cleanup_error:
                logger.warning(
                    "mcp_disconnect_failed",
                    correlation_id=correlation_id,
                    error=str(cleanup_error),
                )

        reset_correlation_id(token)


def get_mcp_circuit_metrics() -> dict[str, Any]:
    """Get current MCP circuit metrics for monitoring.

    Returns:
        Dictionary containing circuit metrics.
    """
    return mcp_circuit.get_metrics()


def get_all_circuit_metrics() -> dict[str, Any]:
    """Get metrics for all circuit breakers.

    Returns:
        Dictionary containing metrics for LLM and MCP circuits.
    """
    return {
        "llm": llm_circuit.get_metrics(),
        "mcp": mcp_circuit.get_metrics(),
    }
