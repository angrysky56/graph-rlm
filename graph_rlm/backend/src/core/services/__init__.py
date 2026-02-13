"""Circuit-breaker-protected service wrappers."""

from .circuit import (
    protected_llm_generate,
    protected_llm_with_fallback,
    get_llm_circuit_metrics,
    get_llm_service,
)

__all__ = [
    "protected_llm_generate",
    "protected_llm_with_fallback",
    "get_llm_circuit_metrics",
    "get_llm_service",
]
