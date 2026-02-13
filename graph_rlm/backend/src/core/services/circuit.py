"""Circuit-breaker-protected LLM service wrapper.

Provides a wrapper around the LLM service that integrates with the
circuit breaker pattern for resilience against external AI provider failures.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from ..circuit import (
    CircuitOpenError,
    get_correlation_id,
    generate_correlation_id,
    llm_circuit,
    set_correlation_id,
    reset_correlation_id,
)
from ..llm import LLMService

logger = structlog.get_logger(__name__)

# Singleton LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton.

    Returns:
        Configured LLMService instance.
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def protected_llm_generate(
    prompt: Any,
    system: Optional[str] = None,
    stream: bool = False,
    stop: Optional[list[str]] = None,
    on_usage: Optional[Any] = None,
    model: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Any:
    """Execute LLM query through circuit breaker with correlation tracking.

    Wraps LLMService.generate() with circuit breaker protection.
    Handles CircuitOpenError gracefully with proper logging.

    Args:
        prompt: The prompt or messages to send to the LLM.
        system: Optional system message.
        stream: Whether to stream the response.
        stop: Optional stop sequences.
        on_usage: Optional callback for usage info.
        model: Optional model override.
        correlation_id: Correlation ID for tracing. Generated if not provided.

    Returns:
        LLM response.

    Raises:
        CircuitOpenError: If circuit is open and call is rejected.
        Exception: If LLM service returns an error.
    """
    # Set or generate correlation ID
    if correlation_id is None:
        correlation_id = get_correlation_id() or generate_correlation_id()

    token = set_correlation_id(correlation_id)

    try:
        logger.info(
            "llm_query_start",
            correlation_id=correlation_id,
            has_system=system is not None,
            stream=stream,
        )

        # Execute through circuit breaker
        result = await llm_circuit.call_async(
            get_llm_service().generate,
            prompt,
            system=system,
            stream=stream,
            stop=stop,
            on_usage=on_usage,
            model=model,
        )

        logger.info(
            "llm_query_success",
            correlation_id=correlation_id,
            response_type=type(result).__name__,
        )

        return result

    except CircuitOpenError:
        logger.warning(
            "llm_query_circuit_open",
            correlation_id=correlation_id,
            circuit=llm_circuit.name,
        )
        raise

    finally:
        reset_correlation_id(token)


async def protected_llm_with_fallback(
    prompt: Any,
    fallback_message: str = "AI service temporarily unavailable",
    **kwargs: Any,
) -> tuple[Any, bool]:
    """Execute LLM query with fallback on circuit open.

    Provides graceful degradation when LLM service is unavailable.

    Args:
        prompt: The prompt or messages to send to the LLM.
        fallback_message: Message to return if circuit is open.
        **kwargs: Additional arguments for protected_llm_generate.

    Returns:
        Tuple of (result, was_fallback) where was_fallback indicates
        if the fallback was used.
    """
    try:
        result = await protected_llm_generate(prompt, **kwargs)
        return result, False

    except CircuitOpenError:
        logger.warning(
            "llm_fallback_used",
            correlation_id=get_correlation_id(),
            fallback_message=fallback_message,
        )
        return fallback_message, True


def get_llm_circuit_metrics() -> dict[str, Any]:
    """Get current LLM circuit metrics for monitoring.

    Returns:
        Dictionary containing circuit metrics.
    """
    return llm_circuit.get_metrics()
