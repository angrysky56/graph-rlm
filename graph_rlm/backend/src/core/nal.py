"""
NARS NAL (Non-Axiomatic Logic) Truth-Value Calculus for Graph-RLM.

Implements the core inference functions from NAL-1 through NAL-6:
- Revision: Combining two judgments about the same statement
- Deduction: A→B, B→C ⊢ A→C
- Abduction: A→B, C→B ⊢ A→C (hypothesis forming)
- Induction: A→B, A→C ⊢ B→C (generalization)
- Expectation: Decision criterion e = (c * (f - 0.5) + 0.5)
- Choice: Selecting between competing judgments

Truth values follow NARS convention:
    ⟨f, c⟩ where f ∈ [0,1] (frequency) and c ∈ [0,1) (confidence)
    f = positive_evidence / total_evidence
    c = total_evidence / (total_evidence + k)  (k = evidential horizon, default 1)

References:
    - Pei Wang, "Non-Axiomatic Reasoning System" (2006)
    - Pei Wang, "Non-Axiomatic Logic: A Model of Intelligent Reasoning" (2013)
    - OpenNARS project: https://github.com/opennars/opennars
"""

from dataclasses import dataclass
from typing import List, Optional

from .logger import get_logger

logger = get_logger("graph_rlm.nal")

# Evidential horizon: controls how quickly confidence saturates.
# k=1 is the standard NARS default.
K_HORIZON: float = 1.0


@dataclass
class TruthValue:
    """A NARS truth value ⟨f, c⟩.

    Attributes:
        frequency: Proportion of positive evidence (0.0 to 1.0).
        confidence: Degree of evidential support (0.0 to <1.0).
    """

    frequency: float
    confidence: float

    def __post_init__(self) -> None:
        """Clamp values to valid ranges."""
        self.frequency = max(0.0, min(1.0, self.frequency))
        self.confidence = max(0.0, min(1.0 - 1e-9, self.confidence))

    @property
    def expectation(self) -> float:
        """Decision-making criterion: e = c * (f - 0.5) + 0.5.

        Used for choosing between competing judgments.
        Range: [0, 1], with 0.5 being neutral.
        """
        return self.confidence * (self.frequency - 0.5) + 0.5

    @property
    def evidence_weight(self) -> float:
        """Convert confidence to evidence weight: w = c / (1 - c)."""
        if self.confidence >= 1.0 - 1e-9:
            return 1e6  # Practical infinity
        return self.confidence / (1.0 - self.confidence)

    @classmethod
    def from_evidence(cls, positive: float, total: float) -> "TruthValue":
        """Create a TruthValue from evidence counts.

        Args:
            positive: Amount of positive evidence.
            total: Total amount of evidence.
        """
        if total <= 0:
            return cls(frequency=0.5, confidence=0.0)
        f = positive / total
        c = total / (total + K_HORIZON)
        return cls(frequency=f, confidence=c)

    def __repr__(self) -> str:
        return f"⟨{self.frequency:.3f}, {self.confidence:.3f}⟩"


# ──────────────────────────────────────────────
# NAL-1: First-Order Inference (Inheritance)
# ──────────────────────────────────────────────


def revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """NAL Revision: Combine two independent judgments about the SAME statement.

    This is the primary truth-maintenance operation.
    Precondition: tv1 and tv2 must be based on independent evidence.

    Formula:
        w1 = c1 / (1 - c1)
        w2 = c2 / (1 - c2)
        f_new = (w1 * f1 + w2 * f2) / (w1 + w2)
        c_new = (w1 + w2) / (w1 + w2 + k)

    Args:
        tv1: First judgment.
        tv2: Second judgment (independent evidence).

    Returns:
        Revised truth value with increased confidence.
    """
    w1 = tv1.evidence_weight
    w2 = tv2.evidence_weight
    w_total = w1 + w2

    if w_total < 1e-9:
        return TruthValue(frequency=0.5, confidence=0.0)

    f_new = (w1 * tv1.frequency + w2 * tv2.frequency) / w_total
    c_new = w_total / (w_total + K_HORIZON)

    return TruthValue(frequency=f_new, confidence=c_new)


def deduction(tv_ab: TruthValue, tv_bc: TruthValue) -> TruthValue:
    """NAL Deduction: A→B, B→C ⊢ A→C.

    The classic syllogistic inference. Strong when both premises
    have high frequency and confidence.

    Formula:
        f = f_ab * f_bc
        c = f_ab * f_bc * c_ab * c_bc

    Args:
        tv_ab: Truth value of A→B.
        tv_bc: Truth value of B→C.

    Returns:
        Truth value of the deduced A→C.
    """
    f = tv_ab.frequency * tv_bc.frequency
    c = tv_ab.frequency * tv_bc.frequency * tv_ab.confidence * tv_bc.confidence
    return TruthValue(frequency=f, confidence=c)


def abduction(tv_ab: TruthValue, tv_cb: TruthValue) -> TruthValue:
    """NAL Abduction: A→B, C→B ⊢ A→C (hypothesis formation).

    'Inference to the best explanation.' Weak but creative.
    Used when two things share a common consequence.

    Formula:
        f = f_cb
        c = f_ab * c_ab * c_cb / (f_ab * c_ab * c_cb + k)

    Args:
        tv_ab: Truth value of A→B.
        tv_cb: Truth value of C→B.

    Returns:
        Truth value of the abduced A→C (typically low confidence).
    """
    f = tv_cb.frequency
    w = tv_ab.frequency * tv_ab.confidence * tv_cb.confidence
    c = w / (w + K_HORIZON)
    return TruthValue(frequency=f, confidence=c)


def induction(tv_ab: TruthValue, tv_ac: TruthValue) -> TruthValue:
    """NAL Induction: A→B, A→C ⊢ B→C (generalization).

    Learns a new relation by observing two consequences of the same cause.
    The dual of abduction.

    Formula:
        f = f_ac
        c = f_ab * c_ab * c_ac / (f_ab * c_ab * c_ac + k)

    Args:
        tv_ab: Truth value of A→B.
        tv_ac: Truth value of A→C.

    Returns:
        Truth value of the induced B→C (typically low confidence).
    """
    f = tv_ac.frequency
    w = tv_ab.frequency * tv_ab.confidence * tv_ac.confidence
    c = w / (w + K_HORIZON)
    return TruthValue(frequency=f, confidence=c)


# ──────────────────────────────────────────────
# NAL-5: Higher-Order Inference
# ──────────────────────────────────────────────


def negation(tv: TruthValue) -> TruthValue:
    """NAL Negation: ¬S.

    Formula: f_neg = 1 - f, c_neg = c

    Args:
        tv: Truth value to negate.

    Returns:
        Negated truth value.
    """
    return TruthValue(frequency=1.0 - tv.frequency, confidence=tv.confidence)


def conjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """NAL Intersection/Conjunction: S1 ∧ S2.

    Formula: f = f1 * f2, c = c1 * c2

    Args:
        tv1: First component truth value.
        tv2: Second component truth value.

    Returns:
        Conjunctive truth value.
    """
    return TruthValue(
        frequency=tv1.frequency * tv2.frequency,
        confidence=tv1.confidence * tv2.confidence,
    )


def disjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """NAL Union/Disjunction: S1 ∨ S2.

    Formula: f = 1 - (1-f1)*(1-f2), c = c1 * c2

    Args:
        tv1: First component truth value.
        tv2: Second component truth value.

    Returns:
        Disjunctive truth value.
    """
    f = 1.0 - (1.0 - tv1.frequency) * (1.0 - tv2.frequency)
    return TruthValue(frequency=f, confidence=tv1.confidence * tv2.confidence)


# ──────────────────────────────────────────────
# Decision & Choice
# ──────────────────────────────────────────────


def choice(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """NAL Choice Rule: Select the judgment with higher expectation.

    Used when two dependent (non-independent) judgments compete.
    Unlike revision, this does NOT combine evidence.

    Args:
        tv1: First judgment.
        tv2: Second judgment.

    Returns:
        The judgment with higher expectation value.
    """
    return tv1 if tv1.expectation >= tv2.expectation else tv2


def desire_value(tv_goal: TruthValue, tv_belief: TruthValue) -> TruthValue:
    """NAL Desire Value: How desirable is an action given a goal?

    d(Goal, Belief) = ⟨f_g * f_b, c_g * c_b⟩

    Used for operator selection in SOAR-style decision procedure.

    Args:
        tv_goal: Truth value of the goal (desirability).
        tv_belief: Truth value of the belief (likelihood of achieving goal).

    Returns:
        Desire value for action selection.
    """
    return TruthValue(
        frequency=tv_goal.frequency * tv_belief.frequency,
        confidence=tv_goal.confidence * tv_belief.confidence,
    )


# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────


def confidence_to_evidence(confidence: float) -> float:
    """Convert confidence to total evidence count.

    w = k * c / (1 - c)

    Args:
        confidence: Confidence value in [0, 1).

    Returns:
        Evidence weight.
    """
    if confidence >= 1.0 - 1e-9:
        return 1e6
    return K_HORIZON * confidence / (1.0 - confidence)


def evidence_to_confidence(evidence: float) -> float:
    """Convert evidence count to confidence.

    c = w / (w + k)

    Args:
        evidence: Total evidence weight.

    Returns:
        Confidence value in [0, 1).
    """
    return evidence / (evidence + K_HORIZON)


def merge_truth_values(judgments: List[TruthValue]) -> TruthValue:
    """Sequentially revise a list of independent judgments.

    Applies revision pairwise across all judgments.

    Args:
        judgments: List of independent truth values to merge.

    Returns:
        Merged truth value, or neutral if empty.
    """
    if not judgments:
        return TruthValue(frequency=0.5, confidence=0.0)

    result = judgments[0]
    for tv in judgments[1:]:
        result = revision(result, tv)
    return result


def truth_from_raw(f: Optional[float] = None, c: Optional[float] = None) -> TruthValue:
    """Create a TruthValue from raw floats with defaults.

    Convenience function for integration with existing code that
    passes nars_f / nars_c as optional floats.

    Args:
        f: Frequency (default 1.0 = fully positive).
        c: Confidence (default 0.9 = high confidence).

    Returns:
        A TruthValue instance.
    """
    return TruthValue(
        frequency=f if f is not None else 1.0,
        confidence=c if c is not None else 0.9,
    )
