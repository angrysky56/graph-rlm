"""Unit tests for src.core.exceptions.base module.

Tests for BaseGraphRLMError, GraphRLMExceptionContext, and ErrorCode integration.
"""

from datetime import datetime, timezone
from unittest import mock

import pytest

from graph_rlm.backend.src.core.exceptions.base import (
    BaseGraphRLMError,
    GraphRLMExceptionContext,
)
from graph_rlm.backend.src.core.exceptions.codes import ErrorCode


class TestGraphRLMExceptionContext:
    """Test GraphRLMExceptionContext class."""

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        ctx = GraphRLMExceptionContext(key1="value1", key2="value2")
        assert ctx["key1"] == "value1"
        assert ctx["key2"] == "value2"

    def test_getitem(self):
        """Test __getitem__ method."""
        ctx = GraphRLMExceptionContext(test="value")
        assert ctx["test"] == "value"

    def test_setitem(self):
        """Test __setitem__ method."""
        ctx = GraphRLMExceptionContext()
        ctx["new_key"] = "new_value"
        assert ctx["new_key"] == "new_value"

    def test_contains(self):
        """Test __contains__ method."""
        ctx = GraphRLMExceptionContext(existing="value")
        assert "existing" in ctx
        assert "missing" not in ctx

    def test_iter(self):
        """Test __iter__ method."""
        ctx = GraphRLMExceptionContext(a=1, b=2, c=3)
        keys = list(ctx)
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys

    def test_len(self):
        """Test __len__ method."""
        ctx = GraphRLMExceptionContext(a=1, b=2)
        assert len(ctx) == 2

    def test_repr(self):
        """Test __repr__ method."""
        ctx = GraphRLMExceptionContext(key="value")
        repr_str = repr(ctx)
        assert "key" in repr_str
        assert "value" in repr_str

    def test_to_dict(self):
        """Test to_dict method."""
        ctx = GraphRLMExceptionContext(key="value")
        result = ctx.to_dict()
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy, not the original."""
        ctx = GraphRLMExceptionContext(key="value")
        result = ctx.to_dict()
        result["key"] = "modified"
        assert ctx["key"] == "value"

    def test_merge(self):
        """Test merge method."""
        ctx1 = GraphRLMExceptionContext(a=1, b=2)
        ctx2 = GraphRLMExceptionContext(b=3, c=4)
        merged = ctx1.merge(ctx2)
        assert merged["a"] == 1
        # ctx2 values should override ctx1 values
        assert merged["b"] == 3
        assert merged["c"] == 4

    def test_empty_context(self):
        """Test empty context initialization."""
        ctx = GraphRLMExceptionContext()
        assert len(ctx) == 0
        assert list(ctx) == []


class TestBaseGraphRLMError:
    """Test BaseGraphRLMError class."""

    def test_init_basic(self):
        """Test basic initialization."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.CORE_INTERNAL_ERROR
        assert error.correlation_id is None

    def test_init_with_correlation_id(self):
        """Test initialization with correlation ID."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            correlation_id="corr-123",
        )
        assert error.correlation_id == "corr-123"

    def test_init_with_cause(self):
        """Test initialization with cause exception."""
        original_error = ValueError("Original error")
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            cause=original_error,
        )
        assert error.__cause__ is original_error
        assert error.context["cause_error_type"] == "ValueError"
        assert error.context["cause_message"] == "Original error"

    def test_init_with_context(self):
        """Test initialization with additional context."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            user_id="user-123",
            request_id="req-456",
        )
        assert error.context["user_id"] == "user-123"
        assert error.context["request_id"] == "req-456"

    def test_properties(self):
        """Test property accessors."""
        timestamp_before = datetime.now(timezone.utc)
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        timestamp_after = datetime.now(timezone.utc)

        assert error.message == "Test error"
        assert error.error_code == ErrorCode.CORE_INTERNAL_ERROR
        assert error.correlation_id is None
        assert timestamp_before <= error.timestamp <= timestamp_after
        assert isinstance(error.context, GraphRLMExceptionContext)

    def test_with_correlation_id(self):
        """Test with_correlation_id method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            user_id="user-123",
        )
        new_error = error.with_correlation_id("new-corr-456")

        assert new_error.correlation_id == "new-corr-456"
        assert new_error.message == "Test error"
        assert new_error.error_code == ErrorCode.CORE_INTERNAL_ERROR
        # Context should be merged
        assert new_error.context["user_id"] == "user-123"

    def test_with_correlation_id_preserves_cause(self):
        """Test that with_correlation_id preserves cause."""
        original_error = ValueError("Original")
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            cause=original_error,
        )
        new_error = error.with_correlation_id("corr-123")

        assert new_error.__cause__ is original_error

    def test_with_context(self):
        """Test with_context method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            existing="value",
        )
        new_error = error.with_context(additional="new_value")

        assert new_error.context["existing"] == "value"
        assert new_error.context["additional"] == "new_value"

    def test_with_context_merges(self):
        """Test that with_context merges with existing context."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            key1="value1",
        )
        new_error = error.with_context(key2="value2")

        assert new_error.context["key1"] == "value1"
        assert new_error.context["key2"] == "value2"

    def test_add_context(self):
        """Test add_context method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        error.add_context("new_key", "new_value")
        assert error.context["new_key"] == "new_value"

    def test_add_context_multiple(self):
        """Test adding multiple context entries."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        error.add_context("key1", "value1")
        error.add_context("key2", "value2")
        error.add_context("key3", "value3")

        assert error.context["key1"] == "value1"
        assert error.context["key2"] == "value2"
        assert error.context["key3"] == "value3"

    def test_to_dict(self):
        """Test to_dict serialization."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            correlation_id="corr-123",
            user_id="user-456",
        )
        # Mock the timestamp to be deterministic
        with mock.patch.object(error, "_timestamp", timestamp):
            result = error.to_dict()

        assert result["error_type"] == "BaseGraphRLMError"
        assert result["error_code"] == "CORE_100"
        assert result["message"] == "Test error"
        assert result["correlation_id"] == "corr-123"
        assert result["context"]["user_id"] == "user-456"

    def test_to_dict_with_cause(self):
        """Test to_dict includes cause information."""
        original_error = ValueError("Original error")
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            cause=original_error,
        )

        result = error.to_dict()

        assert result["cause"]["type"] == "ValueError"
        assert result["cause"]["message"] == "Original error"

    def test_to_dict_without_cause(self):
        """Test to_dict when no cause exists."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )

        result = error.to_dict()

        assert result["cause"] is None

    def test_to_json(self):
        """Test to_json method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )

        json_str = error.to_json()

        assert isinstance(json_str, str)
        assert "Test error" in json_str
        assert "CORE_100" in json_str

    def test_to_json_indent(self):
        """Test to_json with indentation."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )

        json_str = error.to_json(indent=2)

        assert isinstance(json_str, str)
        # Indented JSON should contain newlines and spaces
        assert "\n" in json_str

    def test_format_traceback_with_traceback(self):
        """Test format_traceback when traceback exists."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        try:
            raise error
        except BaseGraphRLMError as e:
            traceback_str = e.format_traceback()
            assert isinstance(traceback_str, str)

    def test_format_traceback_without_traceback(self):
        """Test format_traceback when no traceback exists."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )

        traceback_str = error.format_traceback()

        assert traceback_str == ""

    def test_str_method(self):
        """Test __str__ method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )

        str_repr = str(error)

        assert "[CORE_100] Test error" in str_repr

    def test_repr_method(self):
        """Test __repr__ method."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            correlation_id="corr-123",
        )

        repr_str = repr(error)

        assert "BaseGraphRLMError" in repr_str
        assert "Test error" in repr_str
        assert "CORE_100" in repr_str
        assert "corr-123" in repr_str

    def test_reduce_for_pickling(self):
        """Test __reduce__ for pickling support."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            correlation_id="corr-123",
            extra="context",
        )

        result = error.__reduce__()

        assert result[0] == BaseGraphRLMError
        assert result[1] == ("Test error", ErrorCode.CORE_INTERNAL_ERROR)
        assert result[2]["correlation_id"] == "corr-123"
        assert result[2]["context"]["extra"] == "context"

    def test_inheritance_from_exception(self):
        """Test that BaseGraphRLMError inherits from Exception."""
        error = BaseGraphRLMError(
            message="Test error",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        assert isinstance(error, Exception)

    def test_error_code_various_values(self):
        """Test various ErrorCode values."""
        for error_code in ErrorCode:
            error = BaseGraphRLMError(
                message="Test",
                error_code=error_code,
            )
            assert error.error_code == error_code
            assert error.error_code.value == error_code.value


class TestBaseGraphRLMErrorEdgeCases:
    """Test edge cases for BaseGraphRLMError."""

    def test_empty_message(self):
        """Test with empty message."""
        error = BaseGraphRLMError(
            message="",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        assert error.message == ""

    def test_none_correlation_id(self):
        """Test with explicitly None correlation_id."""
        error = BaseGraphRLMError(
            message="Test",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            correlation_id=None,
        )
        assert error.correlation_id is None

    def test_none_cause(self):
        """Test with explicitly None cause."""
        error = BaseGraphRLMError(
            message="Test",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            cause=None,
        )
        assert error.__cause__ is None
        assert "cause_error_type" not in error.context

    def test_chained_exceptions(self):
        """Test exception chaining."""
        exc1 = ValueError("Level 1")
        # BaseGraphRLMError supports cause parameter
        exc3 = BaseGraphRLMError(
            message="Level 3",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            cause=exc1,
        )

        assert exc3.__cause__ is exc1

    def test_nested_context(self):
        """Test with nested context data."""
        error = BaseGraphRLMError(
            message="Test",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
            user={"id": "123", "name": "Test User"},
            items=[1, 2, 3],
        )

        assert error.context["user"]["id"] == "123"
        assert error.context["items"] == [1, 2, 3]

    def test_to_dict_timestamp_format(self):
        """Test that timestamp is properly formatted in to_dict."""
        error = BaseGraphRLMError(
            message="Test",
            error_code=ErrorCode.CORE_INTERNAL_ERROR,
        )
        result = error.to_dict()
        # Timestamp should be ISO format string
        assert "T" in result["timestamp"]
        assert "+00:00" in result["timestamp"] or "Z" in result["timestamp"]


class TestErrorCodeIntegration:
    """Test ErrorCode enum integration with BaseGraphRLMError."""

    def test_all_error_codes(self):
        """Test that all error codes can be used."""
        error_codes = list(ErrorCode)
        assert len(error_codes) > 0

        for code in error_codes:
            error = BaseGraphRLMError(
                message=f"Error {code.name}",
                error_code=code,
            )
            assert error.error_code == code

    def test_error_code_value(self):
        """Test error_code.value returns string."""
        for code in ErrorCode:
            error = BaseGraphRLMError(
                message="Test",
                error_code=code,
            )
            result = error.to_dict()
            assert isinstance(result["error_code"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
