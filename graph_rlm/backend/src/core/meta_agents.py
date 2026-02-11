"""
Meta-Agent Collaboration Framework.

Implements the Breaker/Synthesizer pattern for recursive Sub-REPL collaboration.
Handles complex tasks by decomposing them into fragments (Breakers) and
integrating results (Synthesizer).

Principles:
1. Breaker (Contextualization): Extract key elements, create subtopics, summarize.
2. Synthesizer (Integration): Combine fragments, ensure coherence, produce output.
3. Iterative Deepening: Loop until CoherenceThreshold met (via oMCD).
4. Feedback Loops: Integrate with Dreamer/IntelliSynth for refinement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .logger import get_logger
from .omcd import omcd
from .trace import trace_action

logger = get_logger("graph_rlm.meta_agents")


class AgentRole(Enum):
    """Role of a Sub-REPL in the collaboration."""

    BREAKER = "contextualization"
    SYNTHESIZER = "integration"
    EVALUATOR = "feedback"


@dataclass
class Fragment:
    """Result from a Breaker Sub-REPL."""

    session_id: str
    summary: str
    subtopics: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_output: str = ""


@dataclass
class CollaborationState:
    """State of a Breaker/Synthesizer collaboration."""

    root_session_id: str
    task: str
    fragments: List[Fragment] = field(default_factory=list)
    iteration: int = 0
    coherence_score: float = 0.0
    coherence_threshold: float = 0.7
    max_iterations: int = 5
    is_complete: bool = False


class MetaAgentController:
    """
    Orchestrates Breaker/Synthesizer collaboration for complex tasks.

    Usage:
        controller = meta_agents
        if controller.should_spawn_breakers(task, context_size):
            instructions = controller.get_breaker_instructions(subtask)
            # Spawn Sub-REPL with instructions
            # On return, register fragment
            controller.register_fragment(session_id, fragment)
            if controller.evaluate_coherence(session_id):
                final = controller.get_synthesizer_instructions(session_id)
    """

    def __init__(self):
        self.active_collaborations: Dict[str, CollaborationState] = {}

    def start_collaboration(
        self, root_session_id: str, task: str
    ) -> CollaborationState:
        """Initialize a new collaboration for a complex task."""
        state = CollaborationState(root_session_id=root_session_id, task=task)
        self.active_collaborations[root_session_id] = state
        trace_action(
            "META_AGENT",
            "COLLABORATION_START",
            result=f"Task: {task[:100]}...",
            tag="SYSTEM",
        )
        return state

    def get_collaboration(self, root_session_id: str) -> Optional[CollaborationState]:
        """Get the collaboration state for a session."""
        return self.active_collaborations.get(root_session_id)

    def should_spawn_breakers(
        self, task: str, context_size: int, depth: int = 0
    ) -> bool:
        """
        Determine if task complexity requires Breaker Sub-REPLs.

        Heuristics:
        - Large context size (> 5000 chars)
        - Complexity keywords in task
        - Not already at max depth
        """
        if depth >= 3:  # Prevent infinite recursion
            return False

        complexity_keywords = [
            "analyze",
            "compare",
            "synthesize",
            "document",
            "investigate",
            "research",
            "explain in detail",
            "comprehensive",
            "all aspects",
        ]
        has_complexity = any(kw in task.lower() for kw in complexity_keywords)
        is_large_context = context_size > 5000

        should_spawn = has_complexity or is_large_context

        if should_spawn:
            logger.info(
                "[MetaAgent] Breaker spawn recommended: context=%d, has_complexity=%s",
                context_size,
                has_complexity,
            )

        return should_spawn

    def get_breaker_instructions(self, subtask: str, fragment_index: int = 0) -> str:
        """Generate Breaker-specific system prompt injection."""
        return f"""
═══════════════════════════════════════════════════════════════
[BREAKER PROTOCOL] — Fragment #{fragment_index}
═══════════════════════════════════════════════════════════════
Role: CONTEXTUALIZATION (Extract & Summarize)
Task Fragment: {subtask[:500]}

INSTRUCTIONS:
1. Extract ONLY core ideas (max 3-5 key points)
2. Create structured subtopics as bullet points
3. Return a HIGH-LEVEL SUMMARY (≤500 tokens)
4. Do NOT perform full analysis — the Synthesizer will do that

OUTPUT FORMAT:
## Key Ideas
- Idea 1: [Brief explanation]
- Idea 2: [Brief explanation]

## Subtopics Identified
- Topic A: [One-line description]
- Topic B: [One-line description]

## Summary (≤200 words)
[Your concise summary here]
═══════════════════════════════════════════════════════════════
"""

    def get_synthesizer_instructions(self, root_session_id: str) -> str:
        """Generate Synthesizer-specific system prompt for final integration."""
        state = self.active_collaborations.get(root_session_id)
        if not state or not state.fragments:
            return ""

        fragment_summaries = "\n".join(
            f"### Fragment [{f.session_id[:8]}] (confidence: {f.confidence:.2f})\n{f.summary[:300]}"
            for f in state.fragments
        )

        all_subtopics = []
        for f in state.fragments:
            all_subtopics.extend(f.subtopics)
        subtopics_str = "\n".join(f"- {t}" for t in all_subtopics[:20])

        return f"""
═══════════════════════════════════════════════════════════════
[SYNTHESIZER PROTOCOL]
═══════════════════════════════════════════════════════════════
Role: INTEGRATION (Combine & Produce)
Fragments Received: {len(state.fragments)}
Iteration: {state.iteration}
Coherence Score: {state.coherence_score:.2f}

FRAGMENT SUMMARIES:
{fragment_summaries}

ALL SUBTOPICS IDENTIFIED:
{subtopics_str}

INSTRUCTIONS:
1. Combine fragments into a COHERENT NARRATIVE
2. Ensure logical flow between sections
3. Identify any GAPS requiring additional investigation
4. Produce a FINAL SYNTHESIZED ANSWER

OUTPUT FORMAT:
## Synthesized Analysis
[Your integrated analysis here]

## Gaps Identified (if any)
- [Gap that needs more investigation]

## Conclusion
[Final summary]
═══════════════════════════════════════════════════════════════
"""

    def register_fragment(
        self,
        root_session_id: str,
        fragment: Fragment,
    ) -> None:
        """Register a completed Breaker fragment."""
        state = self.active_collaborations.get(root_session_id)
        if not state:
            logger.warning("No collaboration found for session %s", root_session_id)
            return

        state.fragments.append(fragment)
        state.iteration += 1

        trace_action(
            "META_AGENT",
            "FRAGMENT_REGISTERED",
            result=f"Fragment {len(state.fragments)}: {fragment.summary[:100]}...",
            tag="SYSTEM",
        )

    def evaluate_coherence(self, root_session_id: str) -> bool:
        """
        Determine if collaboration has reached coherence threshold.

        Uses oMCD to balance exploration vs. exploitation.
        """
        state = self.active_collaborations.get(root_session_id)
        if not state or not state.fragments:
            return False

        # Calculate average confidence from fragments
        avg_confidence = sum(f.confidence for f in state.fragments) / len(
            state.fragments
        )

        # Use oMCD to evaluate if we should stop
        omcd_decision = omcd.evaluate_step(state.iteration, avg_confidence)

        state.coherence_score = avg_confidence

        # Check thresholds
        threshold_met = avg_confidence >= state.coherence_threshold
        max_reached = state.iteration >= state.max_iterations
        omcd_stop = omcd_decision["should_stop"]

        if threshold_met or max_reached or omcd_stop:
            state.is_complete = True
            trace_action(
                "META_AGENT",
                "COHERENCE_ACHIEVED",
                result=f"Score: {avg_confidence:.2f}, Iterations: {state.iteration}",
                tag="SYSTEM",
            )
            return True

        return False

    def complete_collaboration(
        self, root_session_id: str
    ) -> Optional[CollaborationState]:
        """Mark collaboration as complete and return final state."""
        state = self.active_collaborations.pop(root_session_id, None)
        if state:
            state.is_complete = True
            trace_action(
                "META_AGENT",
                "COLLABORATION_COMPLETE",
                result=f"Fragments: {len(state.fragments)}, Score: {state.coherence_score:.2f}",
                tag="SYSTEM",
            )
        return state

    def get_evaluator_instructions(self, draft: str, fragments: List[Fragment]) -> str:
        """Generate Evaluator instructions for feedback loop."""
        return f"""
═══════════════════════════════════════════════════════════════
[EVALUATOR PROTOCOL]
═══════════════════════════════════════════════════════════════
Role: FEEDBACK (Evaluate & Refine)

DRAFT TO EVALUATE:
{draft[:2000]}

ORIGINAL FRAGMENTS: {len(fragments)}

EVALUATION CRITERIA:
1. COHERENCE: Does the draft logically connect all fragments?
2. COMPLETENESS: Are all key ideas from fragments represented?
3. ACCURACY: Does the synthesis accurately reflect the source material?
4. GAPS: What's missing that requires additional Breaker investigation?

OUTPUT FORMAT:
## Coherence Score: [0.0 - 1.0]
## Completeness Score: [0.0 - 1.0]
## Accuracy Score: [0.0 - 1.0]
## Overall Score: [Average of above]

## Improvement Suggestions
- [Suggestion 1]
- [Suggestion 2]

## Gaps Requiring Investigation
- [Gap 1]
═══════════════════════════════════════════════════════════════
"""


# Singleton instance
meta_agents = MetaAgentController()
