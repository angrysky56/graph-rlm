import pytest
from graph_rlm.backend.src.core.exceptions.types import (
    CoreError,
    GraphError,
    SkillExecutionError,
    ExternalServiceError,
    ValidationError,
)
from graph_rlm.backend.src.core.exceptions.codes import ErrorCode

class TestCoreError:
    def test_normalization(self):
        # Should normalize to CORE_INTERNAL_ERROR if wrong category provided
        error = CoreError("test", ErrorCode.GRAPH_OPERATION_FAILED)
        assert error.error_code == ErrorCode.CORE_INTERNAL_ERROR
        
        # Should keep CORE code if correct
        error = CoreError("test", ErrorCode.CORE_INTERNAL_ERROR)
        assert error.error_code == ErrorCode.CORE_INTERNAL_ERROR

    def test_with_operation(self):
        error = CoreError("test", ErrorCode.CORE_INTERNAL_ERROR).with_operation("op1")
        assert error.context["operation"] == "op1"

class TestGraphError:
    def test_normalization(self):
        error = GraphError("test", ErrorCode.CORE_INTERNAL_ERROR)
        assert error.error_code == ErrorCode.GRAPH_OPERATION_FAILED

    def test_context_methods(self):
        error = (GraphError("test", ErrorCode.GRAPH_OPERATION_FAILED)
                 .with_graph_operation("read")
                 .with_node_id("node1")
                 .with_edge("s", "t"))
        assert error.context["graph_operation"] == "read"
        assert error.context["node_id"] == "node1"
        assert error.context["source"] == "s"
        assert error.context["target"] == "t"

class TestSkillExecutionError:
    def test_context_methods(self):
        error = (SkillExecutionError("test", ErrorCode.SKILL_EXECUTION_FAILED)
                 .with_skill_name("myskill")
                 .with_skill_input({"a": 1})
                 .with_skill_output("done"))
        assert error.context["skill_name"] == "myskill"
        assert "a" in str(error.context["skill_input"])
        assert error.context["skill_output"] == "done"

class TestExternalServiceError:
    def test_context_methods(self):
        error = (ExternalServiceError("test", ErrorCode.EXTERNAL_SERVICE_ERROR)
                 .with_service_name("svc")
                 .with_endpoint("/end")
                 .with_request("GET", "http://u")
                 .with_response_status(500))
        assert error.context["service"] == "svc"
        assert error.context["endpoint"] == "/end"
        assert error.context["method"] == "GET"
        assert error.context["url"] == "http://u"
        assert error.context["status_code"] == 500
        assert error.http_status_code == 503

class TestValidationError:
    def test_context_methods(self):
        error = (ValidationError("test", ErrorCode.VALIDATION_FIELD_INVALID)
                 .with_field_errors({"f": "err"})
                 .with_field("f2", "v")
                 .with_schema("s")
                 .with_constraint("c"))
        assert error.context["field_errors"] == {"f": "err"}
        assert error.context["field"] == "f2"
        assert error.context["field_value"] == "v"
        assert error.context["schema"] == "s"
        assert error.context["constraint"] == "c"
        assert error.http_status_code == 422
