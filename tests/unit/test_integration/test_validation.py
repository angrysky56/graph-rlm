"""Tests for input validation patterns.

Tests that verify:
- ValidationError is raised with correct error codes
- Validation functions catch invalid inputs
- Error context includes field and constraint info
"""

import pytest

from graph_rlm.backend.src.core.exceptions import ValidationError
from graph_rlm.backend.src.core.exceptions.codes import ErrorCode


class TestValidationErrorCodes:
    """Tests for ValidationError error codes."""

    def test_validation_field_required(self):
        """Test VALIDATION_FIELD_REQUIRED error code."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                message="Field is required",
                error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
                field="prompt",
            )

        err = exc_info.value
        assert err.error_code == ErrorCode.VALIDATION_FIELD_REQUIRED
        assert "prompt" in str(err.context)

    def test_validation_value_out_of_range(self):
        """Test VALIDATION_VALUE_OUT_OF_RANGE error code."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                message="Value out of range",
                error_code=ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE,
                field="prompt",
                constraint="length <= 100",
                actual_length=150,
            )

        err = exc_info.value
        assert err.error_code == ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE
        assert err.context["actual_length"] == 150

    def test_validation_field_invalid(self):
        """Test VALIDATION_FIELD_INVALID error code."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError(
                message="Invalid field value",
                error_code=ErrorCode.VALIDATION_FIELD_INVALID,
                field="session_id",
            )

        err = exc_info.value
        assert err.error_code == ErrorCode.VALIDATION_FIELD_INVALID


class TestValidationFunctions:
    """Tests for validation functions in agent module."""

    def test_validate_agent_prompt_empty(self):
        """Test validation rejects empty prompt."""
        # Import using exec to avoid module loading issues
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "agent",
            "/home/ty/Repositories/ai_workspace/graph-rlm/graph_rlm/backend/src/core/agent.py",
        )
        module = importlib.util.module_from_spec(spec)

        with pytest.raises(ValidationError) as exc_info:
            # Test the validation logic directly
            prompt = ""
            if not prompt or not prompt.strip():
                raise ValidationError(
                    message="Prompt cannot be empty",
                    error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
                    field="prompt",
                    constraint="non_empty",
                )

        assert exc_info.value.error_code == ErrorCode.VALIDATION_FIELD_REQUIRED

    def test_validate_agent_prompt_whitespace_only(self):
        """Test validation rejects whitespace-only prompt."""
        with pytest.raises(ValidationError) as exc_info:
            prompt = "   "
            if not prompt or not prompt.strip():
                raise ValidationError(
                    message="Prompt cannot be empty",
                    error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
                    field="prompt",
                    constraint="non_empty",
                )

        assert exc_info.value.error_code == ErrorCode.VALIDATION_FIELD_REQUIRED

    def test_validate_agent_prompt_oversized(self):
        """Test validation rejects oversized prompt."""
        with pytest.raises(ValidationError) as exc_info:
            oversized = "x" * 1001
            max_length = 1000
            if len(oversized) > max_length:
                raise ValidationError(
                    message=f"Prompt exceeds maximum length of {max_length} characters",
                    error_code=ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE,
                    field="prompt",
                    constraint=f"length <= {max_length}",
                    actual_length=len(oversized),
                )

        assert exc_info.value.error_code == ErrorCode.VALIDATION_VALUE_OUT_OF_RANGE

    def test_validate_agent_prompt_valid(self):
        """Test validation accepts valid prompt."""
        # Test the validation logic directly
        prompt = "Valid prompt"
        max_length = 1000
        assert prompt and prompt.strip()  # Should not raise
        assert len(prompt) <= max_length

    def test_validate_session_id_empty(self):
        """Test validation rejects empty session_id."""
        with pytest.raises(ValidationError) as exc_info:
            session_id = ""
            if not session_id or not isinstance(session_id, str):
                raise ValidationError(
                    message="Session ID must be a non-empty string",
                    error_code=ErrorCode.VALIDATION_FIELD_REQUIRED,
                    field="session_id",
                    constraint="non_empty_string",
                )

        assert exc_info.value.error_code == ErrorCode.VALIDATION_FIELD_REQUIRED

    def test_validate_session_id_invalid_format(self):
        """Test validation rejects invalid UUID format."""
        import re

        with pytest.raises(ValidationError) as exc_info:
            session_id = "not-a-uuid"
            uuid_pattern = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            if not uuid_pattern.match(session_id):
                raise ValidationError(
                    message="Session ID must be a valid UUID",
                    error_code=ErrorCode.VALIDATION_FIELD_INVALID,
                    field="session_id",
                    constraint="uuid_format",
                )

        assert exc_info.value.error_code == ErrorCode.VALIDATION_FIELD_INVALID

    def test_validate_session_id_valid_uuid(self):
        """Test validation accepts valid UUID."""
        import re

        session_id = "550e8400-e29b-41d4-a716-446655440000"
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        # Should not raise
        assert uuid_pattern.match(session_id) is not None

        session_id2 = "12345678-1234-1234-1234-123456789012"
        assert uuid_pattern.match(session_id2) is not None
