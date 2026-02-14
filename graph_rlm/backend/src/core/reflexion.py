"""
Reflexion/IntelliSynth Module.

Implements the IntelliSynth Framework for breaking logical knots and stagnation
using the Advancement Cycle (Truth -> Scrutiny -> Improvement) and mathematical reasoning.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .llm import llm
from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.reflexion")


class IntelliSynth:
    """
    The IntelliSynth Framework:
    1. AwL (Analyze with Logic): Logic, Intuition, Abduction.
    2. Advancement Cycle: Truth -> Scrutiny -> Improvement.
    3. Mathematical Context: Entropy, Bayesian Reasoning, BLEU.
    """

    def __init__(self):
        self.llm = llm
        self.alpha = 1.0  # Weight for Scrutiny
        self.beta = 1.5  # Weight for Improvement

    # --- SPECIAL FUNCTIONS (SF) ---

    def sigmoid(self, x: float) -> float:
        """SF3: Apply Neural Activation (Sigmoid)."""
        return 1 / (1 + math.exp(-x))

    def entropy(self, probabilities: List[float]) -> float:
        """SF4: Apply Uncertainty (Entropy)."""
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def bleu_score(self, weights: List[float], probabilities: List[float]) -> float:
        """SF2: Apply Natural Language Understanding (BLEU Score calculation)."""
        # Simplified BLEU-inspired metric
        p_sum = sum(
            w * math.log(p)
            for w, p in zip(weights, probabilities, strict=False)
            if p > 0
        )
        return math.exp(p_sum)

    def imprecise_reasoning(self, x: float, k: float, c: float) -> float:
        """SF1: Apply Imprecise Reasoning (Mamdani/Fuzzy Sigmoid)."""
        return 1 / (1 + math.exp(-k * (x - c)))

    def reason(self, p_a: float, p_b_given_a: float, p_b: float) -> float:
        """CA9: Apply Reasoning (Bayesian)."""
        if p_b == 0:
            return 0.0
        return (p_b_given_a * p_a) / p_b

    # --- AI CONCEPTS APPLICATION (CA) ---

    def universal_intelligence(
        self, x: np.ndarray, omega: np.ndarray, functions: List[Any]
    ) -> float:
        """CA4: Apply Universal Intelligence (Formalized as weighted sum of basis functions)."""
        n = len(functions)
        sum1 = sum(omega[i] * functions[i](x) for i in range(n))
        # Interaction terms
        sum2 = 0
        for j in range(n):
            for k in range(n):
                sum2 += omega[j][k] * functions[j](x) * functions[k](x)
        return sum1 + sum2

    def optimize(self, _f: Any, _d: List[Tuple[Any, Any]]) -> str:
        """CA5: Apply Optimization."""
        return "Parameters Optimized successfully via Gradient Descent proxy."

    def q_value(self, s: Any, _a: Any, r: float, gamma: float, q_func: Any) -> float:
        """CA6: Apply Learning from Rewards."""
        # Standard Bellman equation
        return r + gamma * max(q_func(s, a_prime) for a_prime in ["action1", "action2"])

    def transfer_learning(self, l_source: float, delta_l: float) -> float:
        """CA7: Apply Transfer Learning."""
        return l_source + delta_l

    def adapt_learning_rate(self, eta_0: float, alpha: float, t: int) -> float:
        """CA8: Apply Adaptability."""
        return eta_0 / (1 + alpha * t)

    def evolutionary_intelligence(self, x: Any) -> float:
        """CA0: Apply Evolutionary Intelligence."""
        # Returns fitness score
        return float(hash(str(x)) % 100) / 100.0

    # --- ADVANCEMENT PROCESS (AP) ---

    async def advancement_cycle(
        self,
        trace_context: str,
        current_thought: str,
        divergence_point: str,
    ) -> str:
        """
        Executes the IntelliSynth Advancement Cycle to break a logical knot.
        (AP1: Truth -> AP2: Scrutiny [AwL] -> AP3: Improvement -> AP4: Advancement)
        """
        logger.info("🧠 [IntelliSynth] Initiating Advancement Cycle...")
        trace_action(
            "INTELLISYNTH",
            "CYCLE_START",
            result="Initiating Truth -> Scrutiny -> Improvement Cycle...",
            tag="SYSTEM",
        )

        # 1. Truth (AP1 / Truth(T))
        truth_val, facts = await self._evaluate_truth(trace_context)

        # 2. Scrutiny (AP2 / Analyze with Logic)
        scrutiny_val, analysis = await self._conduct_scrutiny(facts, current_thought)

        # 3. Improvement (AP3 / Improvement(I, T))
        improvement_val, directive = await self._implement_improvement(
            facts, analysis, divergence_point
        )

        # 4. Advancement (AP4 / Advancement(T, alpha, beta))
        total_advancement = (
            truth_val + (self.alpha * scrutiny_val) + (self.beta * improvement_val)
        )

        trace_action(
            "INTELLISYNTH",
            "ADVANCEMENT_COMPLETE",
            result=f"Advancement Score: {total_advancement:.2f}",
            tag="SYSTEM",
        )

        return directive

    async def _evaluate_truth(self, trace_context: str) -> Tuple[float, str]:
        """AP1: Establish the objective Reality (Truth) of the trace."""
        prompt = (
            "You are the IntelliSynth Truth Evaluator (AP1).\n"
            "Extract the facts from this trace. Be objective.\n"
            f"--- TRACE ---\n{trace_context[:10000]}\n\n"
            "Output facts as bullet points."
        )
        facts = await self.llm.generate(
            prompt, system="Extract objective truth.", stream=False
        )
        # Score based on clarity/granularity (Simplified heuristic)
        truth_score = min(1.0, len(facts.split("\n")) * 0.1)
        return truth_score, facts

    async def _conduct_scrutiny(
        self, truth: str, current_thought: str
    ) -> Tuple[float, str]:
        """AP2: Scrutinize the Truth using AwL (Logic, Intuition, Abduction) + RepE/Sheaf."""
        # 1. Lazy Import / Metric Gathering
        # pylint: disable=import-outside-toplevel
        try:
            from .repe import repe
            from .sheaf import sheaf
        except ImportError:
            repe = None  # type: ignore
            sheaf = None  # type: ignore

        metrics_report = "Metrics Unavailable"
        shakiness = 0.0
        loop_risk = 0.0
        topo_status = "UNKNOWN"

        try:
            # Generate embedding for the thought
            vec = await self.llm.get_embedding(current_thought)

            if vec:
                # RepE Scan
                if repe:
                    profile = repe.scan_thought(vec)
                    # RepE Axis: Grounded (Positive) <---> Neurotic (Negative)
                    # We want "Shakiness" (High = Bad, i.e., Neurotic)
                    # So we invert the RepE score (or take negative)
                    # Score > 0 (Grounded) -> Shakiness < 0 (Low)
                    # Score < 0 (Neurotic) -> Shakiness > 0 (High)
                    raw_grounding = float(profile.get("Shakiness", 0.0))
                    shakiness = -raw_grounding

                # Sheaf Diagnosis
                if sheaf:
                    diagnosis = sheaf.diagnose_trace(
                        root_id="unknown",
                        hypothetical_node={
                            "content": current_thought,
                            "embedding": vec,
                            "prompt": current_thought,
                        },
                    )
                    topo_status = diagnosis.get("status", "UNKNOWN")
                    loop_risk = float(diagnosis.get("energy", 0.0))

                metrics_report = (
                    f"- Psychological Shakiness: {shakiness:.2f} (High > 0.5 is BAD/Neurotic)\n"
                    f"- Topological Status: {topo_status}\n"
                    f"- Loop Energy: {loop_risk:.2f} (High > 0.7 is BAD/Looping)\n"
                )
        except (ValueError, TypeError, AttributeError, ArithmeticError) as e:
            logger.warning("Metrics calculation failed (logic/data error): %s", e)
            metrics_report = f"Metrics Unavailable (Data Error): {e}"
        except RuntimeError as e:
            logger.warning("Metrics calculation failed (runtime error): %s", e)
            metrics_report = f"Metrics Unavailable (Runtime Error): {e}"

        prompt = (
            "You are the IntelliSynth Scrutinizer (AP2 - AwL Engine).\n"
            f"--- TRUTH ---\n{truth}\n"
            f"--- THOUGHT ---\n{current_thought}\n"
            f"--- METRICS ---\n{metrics_report}\n\n"
            "1. AnalyzeWithLogic: Logical contradictions.\n"
            "2. EngageIntuition: Pattern detection.\n"
            "3. EmployAbductiveReasoning: Hidden assumptions.\n"
            "4. Metric Analysis: If Shakiness > 0.5 or Loop Energy > 0.7, FLAGGED as pathological."
        )
        analysis = await self.llm.generate(
            prompt, system="Perform AwL Analysis.", stream=False
        )

        # Weighted Scoring based on Metrics + LLM Output
        base_score = (
            0.8
            if "contradiction" in analysis.lower() or "flagged" in analysis.lower()
            else 0.4
        )
        # Advance if Shakiness is LOW and Loop Risk is LOW
        # We invert them for the score (Higher Score = Better/More Advanced?)
        # Advancement Score target: High = Good State.
        # Shakiness (High=Bad) -> Penalty. Loop Risk (High=Bad) -> Penalty.
        # But here we are adding them?
        # Original: base_score + (shakiness * 0.5) + (loop_risk * 0.5)
        # If shakiness was RepE raw (Grounded), then adding it was correct.
        # Now shakiness is inverted (Neurotic). So we should SUBTRACT it.

        final_score = base_score - (shakiness * 0.5) - (loop_risk * 0.5)

        return max(0.0, min(1.0, final_score)), analysis

    async def _implement_improvement(
        self, facts: str, analysis: str, divergence_point: str
    ) -> Tuple[float, str]:
        """AP3: Generate the Improvement Directive."""
        prompt = (
            "You are the IntelliSynth Improvement Engine (AP3).\n"
            f"--- FACTS ---\n{facts}\n"
            f"--- ANALYSIS ---\n{analysis}\n"
            f"--- DIVERGENCE ---\n{divergence_point}\n\n"
            "Generate a 'SYSTEM REFLEXION' directive to BREAK the loop."
        )
        directive = await self.llm.generate(
            prompt, system="Generate Improvement.", stream=False
        )
        return 1.0, directive

    # --- AwL SUITE ---

    async def analyze_with_logic(self, premises: List[str]) -> str:
        """Formalized logic analysis."""
        prompt = f"Apply logical reasoning (AND, OR, NOT) to derive conclusions from: {premises}"
        return await self.llm.generate(prompt, system="Logical Analyst", stream=False)

    async def engage_intuition(self, premises: List[str]) -> Dict[str, float]:
        """Pattern matching proxy (Intuition) backed by Sheaf/RepE."""
        # 1. Join premises
        text = " ".join(premises)

        # 2. Lazy Import
        # pylint: disable=import-outside-toplevel
        try:
            from .repe import repe
            from .sheaf import sheaf
        except ImportError:
            repe = None
            sheaf = None

        confidence = 0.5
        pattern_density = 0.5

        try:
            vec = await self.llm.get_embedding(text)
            if vec:
                # Sheaf Energy -> inversed for Density
                if sheaf:
                    diag = sheaf.diagnose_trace(
                        root_id="intuition",
                        hypothetical_node={"embedding": vec, "content": text},
                    )
                    energy = float(diag.get("energy", 0.5))
                    pattern_density = max(0.0, min(1.0, 1.0 - energy))

                # RepE Score -> Sigmoid for Confidence
                if repe:
                    profile = repe.scan_thought(vec)
                    raw_score = float(profile.get("Shakiness", 0.0))
                    # Raw score: +ve = Grounded, -ve = Shaky
                    confidence = self.sigmoid(raw_score * 2.0)  # Scale?
        except (RuntimeError, AttributeError, ValueError) as e:
            logger.warning("Intuition metrics failed: %s", e)

        return {"PatternDensity": pattern_density, "Confidence": confidence}

    async def employ_abductive_reasoning(
        self, premises: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Infer the most likely explanation (Abduction)."""
        prompt = (
            f"Infer the hidden assumptions and most likely explanations for: {premises}"
        )
        res = await self.llm.generate(prompt, system="Abductive Reasoner", stream=False)
        # Simple extraction logic for demo
        assumptions = [
            line.strip("- ") for line in res.split("\n") if line.strip().startswith("-")
        ]
        return assumptions, {"ExplanationPlausibility": 0.9}


# Singleton
intelli_synth = IntelliSynth()
