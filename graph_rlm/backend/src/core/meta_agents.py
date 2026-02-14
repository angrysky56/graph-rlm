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

import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import settings
from .logger import get_logger
from .omcd import omcd
from .trace import trace_action

logger = get_logger("graph_rlm.meta_agents")


class AgentRole(Enum):
    """Role of a Sub-REPL in the collaboration."""

    BREAKER = "contextualization"
    SYNTHESIZER = "integration"
    EVALUATOR = "feedback"
    WORKER = "execution"


@dataclass
class Fragment:
    """Result from a Breaker Sub-REPL."""

    session_id: str
    summary: str
    subtopics: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


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
            result=f"Task: {task}",
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
        """Generate Breaker-specific system prompt injection (Legacy/Contextualization only)."""
        return f"""
═══════════════════════════════════════════════════════════════
[BREAKER PROTOCOL] — Fragment #{fragment_index}
═══════════════════════════════════════════════════════════════
Role: CONTEXTUALIZATION (Extract & Summarize)
Task Fragment: {subtask}

INSTRUCTIONS:
1. Extract core ideas.
2. Create structured subtopics.
3. Return a detailed analysis (the Synthesizer will integrate this).
4. Feel free to use all tools to provide a complete picture.

OUTPUT FORMAT:
## Analysis
[Detailed analysis here]

## Key Findings
- Finding 1: [Explanation]
- Finding 2: [Explanation]

## Subtopics Identified
- Topic A: [Description]
═══════════════════════════════════════════════════════════════
"""

    def get_worker_instructions(
        self, subtask: str, tools: Optional[List[str]] = None
    ) -> str:
        """Generate specialized Worker instructions for atomic task execution."""
        tools_str = ", ".join(tools) if tools else "All Available Tools"
        return f"""
═══════════════════════════════════════════════════════════════
[ATOMIC WORKER PROTOCOL]
═══════════════════════════════════════════════════════════════
Role: EXECUTION (Act & Solve)
Task: {subtask}
Available Tools: {tools_str}

INSTRUCTIONS:
1. You are an autonomous sub-process dedicated to this specific task.
2. EXECUTE the task as far as possible using your tools (Code, Search, etc.).
3. DO NOT summarize what you *would* do. DO IT.
4. If the task requires research, perform it. If it requires code, write/run it.
5. Return the raw output, artifacts, or definitive answers.
6. Use rlm.done() or rlm.stop() when finished.

OUTPUT FORMAT:
## Execution Results
[The actual work performed]

## Artifacts Produced
- [File paths, data points, or code blocks]
═══════════════════════════════════════════════════════════════
"""

    def generate_sub_agent_profile(self, task: str) -> Dict[str, Any]:
        """
        Analyze the task and generate a specialized persona profile.

        Heuristics for persona selection:
        - Code/File tasks -> Coder
        - Search/Info tasks -> Researcher
        - Logic/Math tasks -> Analyst
        - Default -> Generalist
        """
        task_lower = task.lower()

        if any(
            w in task_lower
            for w in ["write", "code", "file", "create", "implement", "fix"]
        ):
            return {
                "persona": "Implementation Engineer",
                "tools": ["File System", "Python REPL", "Linter", "Git"],
                "role": AgentRole.WORKER,
            }

        if any(
            w in task_lower
            for w in ["search", "find", "research", "investigate", "price", "latest"]
        ):
            return {
                "persona": "Intelligence Researcher",
                "tools": ["Web Search", "Browser", "Semantic Recall"],
                "role": AgentRole.WORKER,
            }

        if any(
            w in task_lower
            for w in ["calculate", "analyze", "math", "logic", "check", "verify"]
        ):
            return {
                "persona": "Systems Analyst",
                "tools": ["Python REPL", "Axiom Validator", "Graph Search"],
                "role": AgentRole.WORKER,
            }

        return {
            "persona": "Autonomous Generalist",
            "tools": ["All"],
            "role": AgentRole.WORKER,
        }

    def get_synthesizer_instructions(self, root_session_id: str) -> str:
        """Generate Synthesizer-specific system prompt for final integration."""
        state = self.active_collaborations.get(root_session_id)
        if not state or not state.fragments:
            return ""

        # [OPTIMIZATION] Aggregate fragments into a Digest File instead of the prompt
        try:
            kb_root = Path(settings.KNOWLEDGE_BASE_PATH)
            reports_dir = kb_root / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            digest_filename = f"synthesis_digest_{root_session_id[:8]}_{int(datetime.datetime.now().timestamp())}.md"
            digest_path = reports_dir / digest_filename

            fragment_content = []
            for f in state.fragments:
                fragment_content.append(f"### Fragment [{f.session_id[:8]}]")
                fragment_content.append(f"Confidence: {f.confidence:.2f}")
                fragment_content.append(f"Summary: {f.summary}")
                if f.subtopics:
                    fragment_content.append("Subtopics:")
                    for t in f.subtopics:
                        fragment_content.append(f"- {t}")
                fragment_content.append("\n---\n")

            digest_path.write_text("\n".join(fragment_content), encoding="utf-8")
            digest_ref = f"Fragment data saved to: {digest_path}"
        except OSError as e:
            logger.warning("Failed to create synthesis digest file: %s", e)
            digest_ref = "Error creating digest file. Proceed with available context."

        return f"""
═══════════════════════════════════════════════════════════════
[SYNTHESIZER PROTOCOL]
═══════════════════════════════════════════════════════════════
Role: INTEGRATION (Combine & Produce)
Fragments Received: {len(state.fragments)}
Iteration: {state.iteration}
Coherence Score: {state.coherence_score:.2f}

CONTEXT REFERENCE:
{digest_ref}

INSTRUCTIONS:
1. Combine fragments into a COHERENT NARRATIVE.
2. You MUST read the digest file above to see the fragment details.
3. Use 'await rlm.read_document(path)' or standard file tools to ingest the data.
4. Ensure logical flow between sections.
5. Identify any GAPS requiring additional investigation.
6. Produce a FINAL SYNTHESIZED ANSWER.

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a completed Breaker fragment."""
        state = self.active_collaborations.get(root_session_id)
        if not state:
            logger.warning("No collaboration found for session %s", root_session_id)
            return

        # Merge metadata if provided separately (legacy support or explicit override)
        if metadata:
            fragment.metadata.update(metadata)

        state.fragments.append(fragment)
        state.iteration += 1

        trace_action(
            "META_AGENT",
            "FRAGMENT_REGISTERED",
            result=f"Fragment {len(state.fragments)}: {fragment.summary}",
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
{draft}

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
