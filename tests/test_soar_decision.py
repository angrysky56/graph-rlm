"""Tests for SOAR cognitive cycle (decision procedure)."""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_rlm.backend.src.core.meta_agents import (
    MetaAgentController,
    Operator,
    OperatorProposal,
)


def _make_proposal(
    goal: str = "Test goal",
    state: str = "Test state",
    operators: list = None,
) -> OperatorProposal:
    """Helper to build an OperatorProposal."""
    if operators is None:
        operators = []
    return OperatorProposal(
        current_goal=goal,
        current_state=state,
        operators=operators,
    )


def test_detect_impasse_clear_winner():
    """BETTER operator should win with no impasse."""
    ctrl = MetaAgentController()
    proposal = _make_proposal(
        operators=[
            Operator(
                action="Run the test suite",
                tool="REPL",
                preference="BETTER",
                rationale="Tests validate correctness.",
            ),
            Operator(
                action="Read docs for background",
                tool="Search",
                preference="WORSE",
                rationale="Docs may not be relevant.",
            ),
        ]
    )
    result = ctrl.detect_impasse(proposal)
    assert result["impasse_type"] == "NONE"
    assert result["chosen_operator"] is not None
    assert result["chosen_operator"].action == "Run the test suite"


def test_detect_impasse_tie():
    """Two BETTER operators should trigger TIE impasse."""
    ctrl = MetaAgentController()
    proposal = _make_proposal(
        operators=[
            Operator(
                action="Approach A",
                tool="REPL",
                preference="BETTER",
                rationale="Equally valid.",
            ),
            Operator(
                action="Approach B",
                tool="REPL",
                preference="BETTER",
                rationale="Also equally valid.",
            ),
        ]
    )
    result = ctrl.detect_impasse(proposal)
    assert result["impasse_type"] == "TIE"
    assert result["chosen_operator"] is None
    assert "subgoal" in result


def test_detect_impasse_no_knowledge():
    """All WORSE operators should trigger NO_KNOWLEDGE impasse."""
    ctrl = MetaAgentController()
    proposal = _make_proposal(
        operators=[
            Operator(
                action="Guess randomly",
                tool="REPL",
                preference="WORSE",
                rationale="No idea what to do.",
            ),
        ]
    )
    result = ctrl.detect_impasse(proposal)
    assert result["impasse_type"] == "NO_KNOWLEDGE"
    assert result["chosen_operator"] is None


def test_detect_impasse_empty_operators():
    """Empty operator list should trigger NO_KNOWLEDGE."""
    ctrl = MetaAgentController()
    proposal = _make_proposal(operators=[])
    result = ctrl.detect_impasse(proposal)
    assert result["impasse_type"] == "NO_KNOWLEDGE"


def test_detect_impasse_acceptable_triggers_no_knowledge():
    """ACCEPTABLE + WORSE should trigger NO_KNOWLEDGE (desire too low).

    This is correct NAL behavior: if the best option is only 'acceptable',
    the system recognizes insufficient confidence and asks for more info.
    """
    ctrl = MetaAgentController()
    proposal = _make_proposal(
        operators=[
            Operator(
                action="Risky move",
                tool="REPL",
                preference="WORSE",
                rationale="Speculative.",
            ),
            Operator(
                action="Safe approach",
                tool="Search",
                preference="ACCEPTABLE",
                rationale="Moderate confidence.",
            ),
        ]
    )
    result = ctrl.detect_impasse(proposal)
    # NAL desire_value for ACCEPTABLE is 0.545 < 0.55 threshold
    # This correctly triggers NO_KNOWLEDGE — the system should gather more info
    assert result["impasse_type"] == "NO_KNOWLEDGE"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
