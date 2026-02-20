import sys
from pathlib import Path

# Add the project root to sys.path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from graph_rlm.backend.axioms_dir.axiom_side_effect_grounding_validator import (  # noqa: E402
    side_effect_grounding_validator,
)
from graph_rlm.backend.axioms_dir.axiom_trace_temporal_grounding_validator import (  # noqa: E402
    trace_temporal_grounding_validator,
)
from graph_rlm.backend.axioms_dir.axiom_validate_axiom_grounded_synthesis import (  # noqa: E402
    validate_axiom_grounded_synthesis,
)
from graph_rlm.backend.axioms_dir.axiom_validate_structural_alignment import (  # noqa: E402
    validate_structural_alignment,
)
from graph_rlm.backend.axioms_dir.axiom_verify_trace_proposal_alignment import (  # noqa: E402
    verify_trace_proposal_alignment,
)


def test_trace_temporal_grounding_validator():
    print("Testing trace_temporal_grounding_validator...")
    # This should not raise an error
    trace_temporal_grounding_validator("output", "session", [1])
    print("Passed.")


def test_side_effect_grounding_validator():
    print("Testing side_effect_grounding_validator...")
    # This should not raise an error
    # Note: it might return False because files don't exist, but it shouldn't crash on required_keys
    side_effect_grounding_validator(["non_existent_file"])
    print("Passed.")


def test_validate_axiom_grounded_synthesis():
    print("Testing validate_axiom_grounded_synthesis...")
    # This should not raise an error
    validate_axiom_grounded_synthesis("synthesis", {"node": "data"})
    print("Passed.")


def test_verify_trace_proposal_alignment():
    print("Testing verify_trace_proposal_alignment...")
    # This should not raise an error
    verify_trace_proposal_alignment(
        [{"node_id": "1", "logic": "test"}], {"content": "test"}
    )
    print("Passed.")


def test_validate_structural_alignment():
    print("Testing validate_structural_alignment...")
    # This should not raise an error
    validate_structural_alignment({"concept": "definition"})
    print("Passed.")


if __name__ == "__main__":
    try:
        test_trace_temporal_grounding_validator()
        test_side_effect_grounding_validator()
        test_validate_axiom_grounded_synthesis()
        test_verify_trace_proposal_alignment()
        test_validate_structural_alignment()
        print("\nAll axiom default value tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)
