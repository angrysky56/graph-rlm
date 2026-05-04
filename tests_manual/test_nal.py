"""Tests for NARS NAL truth-value calculus."""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_rlm.backend.src.core.nal import (
    TruthValue,
    abduction,
    choice,
    conjunction,
    deduction,
    desire_value,
    disjunction,
    evidence_to_confidence,
    induction,
    merge_truth_values,
    negation,
    revision,
    truth_from_raw,
)


def test_truth_value_basics():
    """TruthValue clamps and computes expectation correctly."""
    tv = TruthValue(frequency=0.8, confidence=0.9)
    assert tv.frequency == 0.8
    assert tv.confidence == 0.9
    # expectation = 0.9 * (0.8 - 0.5) + 0.5 = 0.77
    assert abs(tv.expectation - 0.77) < 0.01

    # Clamping
    tv_bad = TruthValue(frequency=1.5, confidence=-0.1)
    assert tv_bad.frequency == 1.0
    assert tv_bad.confidence == 0.0


def test_truth_from_evidence():
    """Create TruthValue from evidence counts."""
    tv = TruthValue.from_evidence(positive=7, total=10)
    assert abs(tv.frequency - 0.7) < 0.01
    # c = 10 / (10 + 1) ≈ 0.909
    assert abs(tv.confidence - 0.909) < 0.01


def test_revision_increases_confidence():
    """Revision should always increase confidence."""
    tv1 = TruthValue(frequency=0.8, confidence=0.5)
    tv2 = TruthValue(frequency=0.7, confidence=0.5)
    result = revision(tv1, tv2)

    # Confidence must increase
    assert result.confidence > tv1.confidence
    assert result.confidence > tv2.confidence

    # Frequency should be between the two inputs
    assert 0.7 <= result.frequency <= 0.8


def test_revision_symmetry():
    """Revision is commutative."""
    tv1 = TruthValue(frequency=0.9, confidence=0.6)
    tv2 = TruthValue(frequency=0.3, confidence=0.4)
    r1 = revision(tv1, tv2)
    r2 = revision(tv2, tv1)
    assert abs(r1.frequency - r2.frequency) < 1e-9
    assert abs(r1.confidence - r2.confidence) < 1e-9


def test_deduction():
    """Deduction: strong premises yield strong conclusion."""
    tv_ab = TruthValue(frequency=0.9, confidence=0.9)
    tv_bc = TruthValue(frequency=0.9, confidence=0.9)
    result = deduction(tv_ab, tv_bc)

    # f = 0.9 * 0.9 = 0.81
    assert abs(result.frequency - 0.81) < 0.01
    # c = 0.9 * 0.9 * 0.9 * 0.9 = 0.6561
    assert abs(result.confidence - 0.6561) < 0.01


def test_abduction_low_confidence():
    """Abduction should yield lower confidence than deduction."""
    tv_ab = TruthValue(frequency=0.8, confidence=0.9)
    tv_cb = TruthValue(frequency=0.7, confidence=0.8)
    ded = deduction(tv_ab, tv_cb)
    abd = abduction(tv_ab, tv_cb)

    # Abduction confidence should be lower
    assert abd.confidence < ded.confidence


def test_induction_symmetry_with_abduction():
    """Induction and abduction use same confidence formula."""
    tv_ab = TruthValue(frequency=0.8, confidence=0.9)
    tv_ac = TruthValue(frequency=0.7, confidence=0.8)
    ind = induction(tv_ab, tv_ac)
    abd = abduction(tv_ab, tv_ac)
    # Same confidence formula, same inputs → same confidence
    assert abs(ind.confidence - abd.confidence) < 1e-9


def test_negation():
    """Negation flips frequency, keeps confidence."""
    tv = TruthValue(frequency=0.8, confidence=0.9)
    neg = negation(tv)
    assert abs(neg.frequency - 0.2) < 0.01
    assert neg.confidence == tv.confidence


def test_conjunction():
    """Conjunction reduces both frequency and confidence."""
    tv1 = TruthValue(frequency=0.8, confidence=0.9)
    tv2 = TruthValue(frequency=0.7, confidence=0.8)
    conj = conjunction(tv1, tv2)
    assert abs(conj.frequency - 0.56) < 0.01
    assert abs(conj.confidence - 0.72) < 0.01


def test_disjunction():
    """Disjunction increases frequency."""
    tv1 = TruthValue(frequency=0.3, confidence=0.9)
    tv2 = TruthValue(frequency=0.4, confidence=0.8)
    disj = disjunction(tv1, tv2)
    # f = 1 - (0.7 * 0.6) = 1 - 0.42 = 0.58
    assert abs(disj.frequency - 0.58) < 0.01


def test_choice_selects_higher_expectation():
    """Choice returns the judgment with higher expectation."""
    tv_high = TruthValue(frequency=0.9, confidence=0.9)
    tv_low = TruthValue(frequency=0.3, confidence=0.5)
    assert choice(tv_high, tv_low) is tv_high
    assert choice(tv_low, tv_high) is tv_high


def test_desire_value():
    """Desire value combines goal and belief."""
    goal = TruthValue(frequency=1.0, confidence=0.9)
    belief = TruthValue(frequency=0.8, confidence=0.7)
    dv = desire_value(goal, belief)
    assert abs(dv.frequency - 0.8) < 0.01
    assert abs(dv.confidence - 0.63) < 0.01


def test_merge_truth_values():
    """Merging multiple judgments increases confidence progressively."""
    judgments = [
        TruthValue(frequency=0.8, confidence=0.3),
        TruthValue(frequency=0.7, confidence=0.3),
        TruthValue(frequency=0.75, confidence=0.3),
    ]
    result = merge_truth_values(judgments)
    # Confidence should be higher than any individual
    assert result.confidence > 0.3


def test_truth_from_raw():
    """Convert raw floats to TruthValue."""
    tv = truth_from_raw(f=0.8, c=0.5)
    assert tv.frequency == 0.8
    assert tv.confidence == 0.5

    # Defaults
    tv_default = truth_from_raw()
    assert tv_default.frequency == 1.0
    assert tv_default.confidence == 0.9


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
