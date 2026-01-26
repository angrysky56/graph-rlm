"""
Schema-Guided Task Processing Utility

Uses the schema_builder to:
1. Classify incoming tasks against known task schemas
2. Guide tool selection based on task type
3. Track accommodation events as learning signals
4. Integrate with stagnation recovery
"""

import sys
from pathlib import Path

# Add skills_dir to path for skill imports
_kb_path = Path(__file__).parent.parent.parent.parent / "skills_dir"
if str(_kb_path) not in sys.path:
    sys.path.insert(0, str(_kb_path))

# trunk-ignore(ruff/E402)
from .schema_builder import SchemaBuilder


class TaskSchemaProcessor:
    """
    Coordinator utility for schema-guided task processing.

    Uses meta-ontology to classify tasks and track schema evolution.
    """

    def __init__(self):
        self.sb = SchemaBuilder()
        self._initialized = False
        self._task_schemas_created = False

    def initialize(self) -> None:
        """Bootstrap meta-ontology and common task schemas."""
        if self._initialized:
            return

        # Bootstrap the meta-ontology
        self.sb.bootstrap_meta_ontology()

        # Create common task type schemas
        self._create_task_schemas()
        self._initialized = True

    def _create_task_schemas(self) -> None:
        """Create schemas for common coordinator task types."""
        if self._task_schemas_created:
            return

        # Base task schema
        self.sb.create_schema(
            "Task",
            slots={
                "intent": {"type": "str", "weight": 0.9},  # Essential: what to do
                "target": {"type": "str", "weight": 0.7},  # Important: what to do it to
                "constraints": {"type": "list", "weight": 0.3},  # Optional
                "context": {"type": "str", "weight": 0.2},  # Optional background
            },
            defaults={"constraints": [], "context": ""},
        )

        # Specialized task types (inherit from Task)
        self.sb.create_schema(
            "SearchTask",
            parent="Task",
            slots={
                "query": {"type": "str", "weight": 0.9},
                "sources": {"type": "list", "weight": 0.5},
                "limit": {"type": "int", "weight": 0.2},
            },
            defaults={"sources": ["web"], "limit": 10},
        )

        self.sb.create_schema(
            "CodeTask",
            parent="Task",
            slots={
                "language": {"type": "str", "weight": 0.7},
                "operation": {"type": "str", "weight": 0.8},  # create, modify, analyze
                "file_path": {"type": "str", "weight": 0.6},
            },
            defaults={"language": "python"},
        )

        self.sb.create_schema(
            "ReasoningTask",
            parent="Task",
            slots={
                "problem": {"type": "str", "weight": 0.9},
                "depth": {"type": "str", "weight": 0.5},  # shallow, deep, meta
                "methodology": {"type": "str", "weight": 0.4},
            },
            defaults={"depth": "deep", "methodology": "structured"},
        )

        self._task_schemas_created = True

    def classify_task(
        self, task_description: str, extracted_params: dict | None = None
    ) -> dict:
        """
        Classify a task against known schemas.

        Args:
            task_description: Natural language task
            extracted_params: Optional pre-extracted parameters

        Returns:
            Best matching schema, fit score, and recommendations
        """
        self.initialize()

        params = extracted_params or {}
        params["intent"] = task_description[:100]  # Use description as intent

        # Evaluate against each task type
        task_types = ["Task", "SearchTask", "CodeTask", "ReasoningTask"]
        results = []

        for task_type in task_types:
            try:
                fit = self.sb.evaluate_fit(task_type, params)
                results.append(
                    {
                        "task_type": task_type,
                        "fit_score": fit["fit_score"],
                        "status": fit["status"],
                        "missing": fit["missing_slots"],
                        "essential_missing": fit.get("essential_missing", False),
                    }
                )
            except Exception as e:
                print(f"Error evaluating {task_type}: {e}", file=sys.stderr)

        # Sort by fit score
        results.sort(key=lambda x: x["fit_score"], reverse=True)
        best = results[0] if results else None

        return {
            "best_match": best,
            "all_matches": results,
            "recommendation": self._get_recommendation(best) if best else None,
        }

    def _get_recommendation(self, match: dict) -> str:
        """Generate action recommendation based on match."""
        if match["fit_score"] >= 0.8:
            return f"Use {match['task_type']} schema directly"
        elif match["fit_score"] >= 0.5:
            return f"Consider {match['task_type']} but gather: {match['missing']}"
        else:
            return "Low confidence - may need new task schema or clarification"

    def suggest_tools(self, task_type: str) -> list[str]:
        """Suggest tools based on task type schema and dynamic system capabilities."""
        tools = []

        # 1. Inspect MCP Multi-Server Tools
        try:
            import graph_rlm.backend.mcp_tools as mcp_pkg
            ignored = {"list_servers", "call_tool", "run_skill"}
            mcp_tools = [
                t for t in dir(mcp_pkg) if not t.startswith("_") and t not in ignored
            ]
            tools.extend(mcp_tools)
        except ImportError:
            # MCP system might not be active in this context
            pass

        # 2. Fetch Compiled Skills
        try:
            from graph_rlm.backend.src.mcp_integration.skills import get_skills_manager
            mgr = get_skills_manager()
            skills = mgr.list_skills().keys()
            tools.extend(skills)
        except ImportError:
            pass

        return list(set(tools))

    def get_methodology_for_task(
        self, task_type: str, depth: str = "deep"
    ) -> str | None:
        """Get recommended methodology for complex task types.

        Returns the methodology function name to use, or None for simple tasks.
        """
        if task_type == "ReasoningTask" and depth in ("deep", "meta"):
            return "apply_advanced_reasoning"
        return None

    def record_accommodation(self, task_type: str, new_data: dict) -> str:
        """
        Record schema evolution when accommodation occurs.
        Call this when a task doesn't fit existing schemas.
        """
        self.initialize()
        return self.sb.evolve_schema(task_type, new_data)


# Singleton instance for coordinator use
_processor: TaskSchemaProcessor | None = None


def get_task_schema_processor() -> TaskSchemaProcessor:
    """Get or create the task schema processor singleton."""
    global _processor
    if _processor is None:
        _processor = TaskSchemaProcessor()
    return _processor


async def classify_and_route_task(task: str, params: dict | None = None) -> dict:
    """
    Convenience function for task classification.

    Returns classification with suggested tools.
    """
    processor = get_task_schema_processor()
    classification = processor.classify_task(task, params)

    if classification["best_match"]:
        classification["suggested_tools"] = processor.suggest_tools(
            classification["best_match"]["task_type"]
        )

    return classification
