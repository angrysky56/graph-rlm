from unittest.mock import MagicMock, patch

from graph_rlm.backend.src.core.sheaf import SheafMonitor


def test_axiomatic_filtering_failure():
    """
    Reproduces the failure where an unrelated axiom is executed
    because it lacks tags, even when the task has specific domain tags,
    OR when a meta-task is poisoned by keywords.
    """
    sheaf = SheafMonitor()

    # Mock Skill Manager
    mock_mgr = MagicMock()

    # Define a set of skills:
    # 1. A general axiom (no tags) -> Should run
    # 2. A physics axiom (tagged 'physics') -> SHOULD BE FILTERED in meta tasks
    # 3. A coding axiom (tagged 'coding') -> Should run in coding tasks

    mock_skills = {
        "axiom_general_safety": {
            "name": "axiom_general_safety",
            "tags": [],
            "function_name": "validate_safety",
            "code": "def validate_safety(x): return True",
        },
        "axiom_test_physics_f079": {
            "name": "axiom_test_physics_f079",
            "tags": ["physics"],
            "function_name": "validate_physics",
            "code": "def validate_physics(x): return False",
        },
        "axiom_coding_best_practices": {
            "name": "axiom_coding_best_practices",
            "tags": ["coding"],
            "function_name": "validate_coding",
            "code": "def validate_coding(x): return True",
        },
    }

    mock_mgr.list_skills.return_value = mock_skills
    mock_mgr.get_skill.side_effect = lambda name: mock_skills.get(name)

    with patch(
        "graph_rlm.backend.src.mcp_integration.skills.get_skills_manager",
        return_value=mock_mgr,
    ):
        # Scenario 1: Coding task
        task_tags = ["coding"]
        proposed_code = "x = 1"

        with patch("graph_rlm.backend.src.core.sheaf.PythonREPL") as MockREPL:
            instance = MockREPL.return_value
            instance.execute.return_value = ("", "", 1)

            diag = sheaf.check_axiomatic_consistency(proposed_code, task_tags=task_tags)
            axioms_run = diag.get("axioms_run", [])
            print(f"  Axioms selected for 'coding' task: {axioms_run}")

            if "axiom_test_physics_f079" in axioms_run:
                raise AssertionError(
                    "Filtering failure: physics axiom run for coding task"
                )
            print("  ✅ Logic Correct: Physics axiom filtered out of coding task.")

        # Scenario 2: Meta-reasoning task mentioning physics (Keyword Poisoning)
        # We simulate the corrected task_tags from agent.py: ["meta", "math"]
        meta_task_tags = ["meta", "math"]

        with patch("graph_rlm.backend.src.core.sheaf.PythonREPL") as MockREPL:
            instance = MockREPL.return_value
            instance.execute.return_value = ("", "", 1)

            diag_meta = sheaf.check_axiomatic_consistency(
                proposed_code, task_tags=meta_task_tags
            )
            meta_axioms = diag_meta.get("axioms_run", [])
            print(
                f"  Axioms selected for 'meta' task with physics mention: {meta_axioms}"
            )

            if "axiom_test_physics_f079" in meta_axioms:
                raise AssertionError(
                    "Keyword poisoning: physics axiom run for meta task"
                )
            print("  ✅ Logic Correct: Physics axiom filtered out of meta task.")


if __name__ == "__main__":
    print("🧪 Running updated filtering test...")
    try:
        test_axiomatic_filtering_failure()
        print("🎉 SUCCESS: Test PASSED (Keyword poisoning resolved).")
    except AssertionError as e:
        print(f"❌ FAIL: {e}")
    except Exception as e:
        print(f"💥 Test crashed: {e}")
        import traceback

        traceback.print_exc()
