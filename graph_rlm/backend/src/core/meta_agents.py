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
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .config import settings
from .db import db
from .llm import llm
from .logger import get_logger
from .nal import TruthValue, desire_value
from .omcd import omcd
from .prompts import get_breaker_instructions as prompt_get_breaker_instructions
from .prompts import get_evaluator_instructions as prompt_get_evaluator_instructions
from .prompts import get_synthesizer_instructions as prompt_get_synthesizer_instructions
from .prompts import get_worker_instructions as prompt_get_worker_instructions
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
    round_id: str = "unknown"
    turn_id: int = 1
    fragments: List[Fragment] = field(default_factory=list)
    iteration: int = 0
    coherence_score: float = 0.0
    coherence_threshold: float = 0.7
    max_iterations: int = 10
    is_complete: bool = False


class SubAgentProfile(BaseModel):
    """Structured persona and tool assignment for a sub-agent."""

    persona: str = Field(..., description="The specialized role title.")
    tools: List[str] = Field(
        ..., description="Exact available tool names (mcp.x, skills.y, rlm.z)."
    )
    reasoning: str = Field(..., description="Why these tools were selected.")


class OperatorPreference(str, Enum):
    """SOAR operator preference ranking."""

    BETTER = "BETTER"
    ACCEPTABLE = "ACCEPTABLE"
    WORSE = "WORSE"


class Operator(BaseModel):
    """A proposed action in the SOAR Elaboration phase.

    Each operator represents a candidate action the agent could take,
    with a preference ranking and rationale.
    """

    action: str = Field(..., description="What to do (concrete, executable step).")
    tool: str = Field(
        ..., description="Which tool/approach to use (REPL, MCP, search, etc.)."
    )
    preference: str = Field(
        ..., description="BETTER, ACCEPTABLE, or WORSE."
    )
    rationale: str = Field(..., description="Why this operator was proposed.")


class OperatorProposal(BaseModel):
    """LLM output for the Elaboration phase."""

    current_goal: str = Field(..., description="The current goal being pursued.")
    current_state: str = Field(
        ..., description="Known facts about the current state."
    )
    operators: List[Operator] = Field(
        ..., description="2-3 proposed operators with preferences."
    )


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
        self.db = db

    def start_collaboration(
        self,
        root_session_id: str,
        task: str,
        round_id: str = "unknown",
        turn_id: int = 1,
    ) -> CollaborationState:
        """Initialize a new collaboration for a complex task."""
        state = CollaborationState(
            root_session_id=root_session_id,
            task=task,
            round_id=round_id,
            turn_id=turn_id,
        )
        self.active_collaborations[root_session_id] = state
        trace_action(
            "META_AGENT",
            "COLLABORATION_START",
            result=f"Task: {task}",
            tag="SYSTEM",
        )

        # PERSISTENCE: Materialize collaboration start
        try:
            start_lid = f"{root_session_id}:META:START"
            start_id = str(uuid.uuid4())
            self.db.create_thought_node(
                thought_id=start_id,
                prompt=f"META-COLLABORATION START: {task}",
                logical_id=start_lid,
                status="system",
                session_id=root_session_id,
                root_session_id=root_session_id,
                round_id=round_id,
                turn_id=turn_id,
                repl_id="BRK",
                execution_summary="Initializing Breaker/Synthesizer protocol.",
                validate=False,
            )
        except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
            logger.error(
                "Failed to persist meta-collaboration start (DB error): %s",
                db_err,
                exc_info=True,
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
        return prompt_get_breaker_instructions(subtask, fragment_index)

    def get_worker_instructions(
        self, subtask: str, tools: Optional[List[str]] = None
    ) -> str:
        """Generate specialized Worker instructions for atomic task execution."""
        return prompt_get_worker_instructions(subtask, tools)

    async def generate_sub_agent_profile(
        self,
        task: str,
        skills_manager: Optional[Any] = None,
        mcp_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the task and generate a specialized persona profile using semantic discovery.

        Queries the skills_manager and uses an LLM to synthesize a bespoke persona
        and assign exact available tools.
        """
        # 1. Gather available capabilities
        relevant_skills = []
        if skills_manager:
            try:
                # Semantic search for relevant skills
                matches = await skills_manager.find_similar_skills(task, limit=5)
                for s in matches:
                    skill_name = s.get("name")
                    func_name = s.get("function_name")
                    desc = s.get("description", "")
                    if skill_name and func_name:
                        relevant_skills.append(
                            f"- {skill_name} ({func_name}): {desc}"
                        )
            except Exception as e:
                logger.warning("Skill discovery failed during profiling for task: %s... -> %s", task[:50], e, exc_info=True)

        mcp_list = ", ".join(mcp_names) if mcp_names else "None (Use rlm commands)"

        # 2. Synthesize Persona and Tools via LLM
        discovery_prompt = f"""
Analyze this task and available capabilities to synthesize a specialized Sub-Agent Profile.

TASK:
"{task}"

AVAILABLE MCP NAMESPACES (mcp.<name>.<tool>):
{mcp_list}

RELEVANT SKILLS (from skills.<name> import <func>):
{"\n".join(relevant_skills) if relevant_skills else "None"}

INSTRUCTIONS:
1. Define a 'Persona' title (e.g., 'Financial Analyst', 'Code Architect').
2. Select the EXACT tool names available above that are most relevant.
   - For MCP: Use 'mcp.<namespace>.<tool>' format (if you know the tools) or just 'mcp.<namespace>'.
   - For Skills: Use 'skills.<name>' format.
   - Always include 'rlm' core tools (rlm.recall, rlm.query, etc.).
3. Return ONLY a JSON object with this structure:
{{
  "persona": "string",
  "tools": ["string", "string"],
  "reasoning": "string"
}}
"""
        try:
            profile_json = await llm.generate_structured(
                discovery_prompt,
                output_type=SubAgentProfile,
                system="You are the Meta-Agent Profiler. Your goal is to map tasks to physical capabilities.",
            )
            return {
                "persona": profile_json.persona,
                "tools": profile_json.tools,
                "role": AgentRole.WORKER,
                "reasoning": profile_json.reasoning,
            }
        except Exception as e:
            logger.error("LLM Profiling failed for task: %s... -> %s. Falling back to heuristics.", task[:50], e, exc_info=True)

            # FALLBACK HEURISTICS
            task_lower = task.lower()
            if any(
                w in task_lower
                for w in ["write", "code", "file", "create", "implement", "fix"]
            ):
                return {
                    "persona": "Implementation Engineer",
                    "tools": ["File System", "Python REPL", "rlm"],
                    "role": AgentRole.WORKER,
                }
            return {
                "persona": "Autonomous Generalist",
                "tools": ["rlm"],
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
            logger.warning(
                "Failed to create synthesis digest file (IO error): %s",
                e,
                exc_info=True,
            )
            digest_ref = "Error creating digest file. Proceed with available context."

        return prompt_get_synthesizer_instructions(
            fragment_count=len(state.fragments),
            iteration=state.iteration,
            coherence_score=state.coherence_score,
            digest_ref=digest_ref,
        )

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

        # PERSISTENCE: Save fragment to Graph so Scratchpad Builder sees it
        try:
            fragment_lid = f"{fragment.session_id}:FRAG:{len(state.fragments)}"
            thought_id = str(uuid.uuid4())
            self.db.create_thought_node(
                thought_id=thought_id,
                prompt=f"Meta-Agent Fragment: {fragment.summary}",
                logical_id=fragment_lid,
                result=f"## Analysis\n{fragment.raw_output}",
                session_id=fragment.session_id,
                root_session_id=root_session_id,
                round_id=state.round_id,
                turn_id=state.turn_id,
                status="fragment",
                repl_id="BRK",
                execution_summary=f"Fragment Confidence: {fragment.confidence:.2f}",
                validate=False,  # Skip guardrails for system-inserted nodes
            )
        except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
            logger.error(
                "Failed to persist fragment node (DB error): %s", db_err, exc_info=True
            )

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

            # PERSISTENCE: Materialize coherence achievement
            try:
                coh_lid = f"{root_session_id}:META:COHERENCE:{state.iteration}"
                coh_id = str(uuid.uuid4())
                self.db.create_thought_node(
                    thought_id=coh_id,
                    prompt=f"META-COHERENCE ACHIEVED: Threshold {state.coherence_threshold:.2f} met.",
                    logical_id=coh_lid,
                    status="system",
                    session_id=root_session_id,
                    root_session_id=root_session_id,
                    round_id=state.round_id,
                    turn_id=state.turn_id,
                    repl_id="SYN",
                    execution_summary=f"Score: {avg_confidence:.2f}, Iterations: {state.iteration}",
                    validate=False,
                )
            except (AttributeError, RuntimeError, KeyError, ValueError) as db_err:
                logger.error(
                    "Failed to persist coherence achievement (DB error): %s",
                    db_err,
                    exc_info=True,
                )

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
        return prompt_get_evaluator_instructions(draft, len(fragments))

    # ─────────────────────────────────────────────────────────
    # SOAR COGNITIVE CYCLE
    # ─────────────────────────────────────────────────────────

    async def propose_operators(
        self,
        goal: str,
        state_summary: str,
        available_tools: Optional[List[str]] = None,
    ) -> OperatorProposal:
        """SOAR Elaboration Phase: Propose 2-3 operators with preferences.

        The LLM analyzes the current goal and state, then proposes
        concrete action operators ranked by expected effectiveness.

        Args:
            goal: The current goal the agent is pursuing.
            state_summary: Working memory summary (from ThimacMemory gestalt).
            available_tools: List of available tool names.

        Returns:
            OperatorProposal with goal, state, and ranked operators.
        """
        tools_str = ", ".join(available_tools) if available_tools else "REPL, MCP tools, rlm commands"

        prompt = (
            f"You are the SOAR Elaboration Engine. Analyze the current cognitive state "  # noqa: E501
            f"and propose 2-3 concrete operators (actions) to advance toward the goal.\n\n"
            f"## Current Goal\n{goal}\n\n"
            f"## Current State (Working Memory)\n{state_summary}\n\n"
            f"## Available Tools\n{tools_str}\n\n"
            f"## Instructions\n"
            f"Propose 2-3 CONCRETE operators. Each must be an executable step, "
            f"not a vague intention. Rank each with a preference:\n"
            f"- BETTER: Highly likely to succeed, directly advances the goal.\n"
            f"- ACCEPTABLE: Plausible but suboptimal or indirect.\n"
            f"- WORSE: Risky, inefficient, or speculative.\n\n"
            f"At least one operator must be BETTER or ACCEPTABLE."
        )

        try:
            proposal = await llm.generate_structured(
                prompt=prompt,
                output_type=OperatorProposal,
                system="You are the SOAR Elaboration Engine. Output structured operator proposals.",
            )
            logger.info(
                "🧠 [SOAR] Elaboration: %d operators proposed for goal: %s",
                len(proposal.operators),
                goal[:80],
            )
            trace_action(
                "SOAR",
                "ELABORATION",
                result=f"Proposed {len(proposal.operators)} operators: "
                + ", ".join(
                    f"[{op.preference}] {op.action[:50]}" for op in proposal.operators
                ),
                tag="SOAR",
            )
            return proposal
        except (RuntimeError, ValueError, AttributeError, httpx.RequestError) as e:
            logger.warning("SOAR Elaboration failed: %s. Using fallback.", e)
            return OperatorProposal(
                current_goal=goal,
                current_state=state_summary[:200],
                operators=[
                    Operator(
                        action="Execute the task directly using available tools.",
                        tool="REPL",
                        preference="BETTER",
                        rationale=f"Fallback: LLM elaboration failed ({e}).",
                    )
                ],
            )

    def detect_impasse(self, proposal: OperatorProposal) -> Dict[str, Any]:
        """SOAR Decision Phase: Detect if an impasse exists.

        Impasse types:
        - TIE: Multiple operators ranked equally, cannot decide.
        - NO_KNOWLEDGE: No BETTER operators, insufficient info.
        - NONE: Clear winner exists, proceed to application.

        Args:
            proposal: The operator proposal from elaboration.

        Returns:
            Dict with 'impasse_type', 'chosen_operator' (if no impasse),
            and 'subgoal' (if impasse detected).
        """
        operators = proposal.operators
        if not operators:
            return {
                "impasse_type": "NO_KNOWLEDGE",
                "chosen_operator": None,
                "subgoal": f"Unable to propose any operators for: {proposal.current_goal}",
            }

        # Score operators using NAL desire values
        scored = []
        for op in operators:
            # Map preference to NAL truth value
            pref_map = {
                "BETTER": TruthValue(frequency=0.9, confidence=0.8),
                "ACCEPTABLE": TruthValue(frequency=0.6, confidence=0.5),
                "WORSE": TruthValue(frequency=0.3, confidence=0.3),
            }
            tv_pref = pref_map.get(
                op.preference.upper(), TruthValue(frequency=0.5, confidence=0.5)
            )
            # Goal desirability assumed high
            tv_goal = TruthValue(frequency=1.0, confidence=0.9)
            dv = desire_value(tv_goal, tv_pref)
            scored.append((dv.expectation, dv, op))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Decision procedure
        best_exp, best_tv, best_op = scored[0]

        if len(scored) >= 2:
            second_exp = scored[1][0]
            # TIE: top two within 0.05 expectation
            if abs(best_exp - second_exp) < 0.05:
                logger.info(
                    "⚡ [SOAR] IMPASSE: Tie between '%s' (%.3f) and '%s' (%.3f)",
                    scored[0][2].action[:40],
                    best_exp,
                    scored[1][2].action[:40],
                    second_exp,
                )
                trace_action(
                    "SOAR",
                    "IMPASSE_TIE",
                    result=f"Tie between {len(scored)} operators.",
                    tag="SOAR",
                )
                return {
                    "impasse_type": "TIE",
                    "chosen_operator": None,
                    "tied_operators": [s[2] for s in scored[:2]],
                    "subgoal": (
                        f"IMPASSE: Cannot decide between operators. "
                        f"Gather more information to break the tie between: "
                        f"'{scored[0][2].action[:50]}' vs '{scored[1][2].action[:50]}'."
                    ),
                }

        # NO_KNOWLEDGE: best operator has low expectation
        if best_exp < 0.55:
            logger.info(
                "⚡ [SOAR] IMPASSE: No confident operator (best=%.3f)",
                best_exp,
            )
            trace_action(
                "SOAR",
                "IMPASSE_NO_KNOWLEDGE",
                result=f"Best operator expectation too low: {best_exp:.3f}",
                tag="SOAR",
            )
            return {
                "impasse_type": "NO_KNOWLEDGE",
                "chosen_operator": None,
                "subgoal": (
                    f"IMPASSE: Insufficient knowledge to proceed confidently. "
                    f"Research or test hypothesis before committing to an action. "
                    f"Context: {proposal.current_state[:200]}"
                ),
            }

        # Clear winner
        logger.info(
            "✅ [SOAR] Decision: '%s' (expectation=%.3f, NAL=%s)",
            best_op.action[:60],
            best_exp,
            best_tv,
        )
        trace_action(
            "SOAR",
            "DECISION",
            result=f"Chosen: [{best_op.preference}] {best_op.action[:80]}",
            tag="SOAR",
        )
        return {
            "impasse_type": "NONE",
            "chosen_operator": best_op,
            "desire_value": best_tv,
        }

    async def run_cognitive_cycle(
        self,
        goal: str,
        state_summary: str,
        session_id: str,
        available_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute one full SOAR cognitive cycle.

        Phases:
            1. INPUT: Working Memory update (state_summary from ThimacMemory)
            2. ELABORATION: Propose 2-3 operators with preferences
            3. DECISION: Evaluate operators, detect impasse
            4. OUTPUT: Return chosen operator or subgoal

        The APPLICATION phase (executing the operator) is handled by
        the caller (agent.py) since it requires REPL access.

        Args:
            goal: The current task goal.
            state_summary: Working memory gestalt (from ThimacMemory).
            session_id: Current session ID for tracing.
            available_tools: Available tool names.

        Returns:
            Dict with 'phase' (APPLICATION or SUBGOAL), 'operator' or 'subgoal',
            and associated metadata.
        """
        trace_action(
            "SOAR",
            "CYCLE_START",
            result=f"Goal: {goal[:80]}",
            tag="SOAR",
        )

        # Phase 1: INPUT (Working Memory already provided via state_summary)
        logger.info("🧠 [SOAR] Phase 1: Working Memory Update (Session: %s)", session_id)

        # Phase 2: ELABORATION
        logger.info("🧠 [SOAR] Phase 2: Elaboration")
        proposal = await self.propose_operators(
            goal=goal,
            state_summary=state_summary,
            available_tools=available_tools,
        )

        # Phase 3: DECISION
        logger.info("🧠 [SOAR] Phase 3: Decision Procedure")
        decision = self.detect_impasse(proposal)

        if decision["impasse_type"] == "NONE":
            # Phase 4a: APPLICATION (return operator for caller to execute)
            chosen = decision["chosen_operator"]
            logger.info(
                "🧠 [SOAR] Phase 4: Application → %s via %s",
                chosen.action[:50],
                chosen.tool,
            )
            return {
                "phase": "APPLICATION",
                "operator": {
                    "action": chosen.action,
                    "tool": chosen.tool,
                    "preference": chosen.preference,
                    "rationale": chosen.rationale,
                },
                "working_memory": {
                    "goal": proposal.current_goal,
                    "state": proposal.current_state,
                },
                "desire_value": str(decision.get("desire_value", "")),
            }

        # Phase 4b: SUBGOALING (impasse detected)
        logger.info(
            "🧠 [SOAR] Phase 4: Subgoaling (impasse=%s)",
            decision["impasse_type"],
        )
        return {
            "phase": "SUBGOAL",
            "impasse_type": decision["impasse_type"],
            "subgoal": decision["subgoal"],
            "working_memory": {
                "goal": proposal.current_goal,
                "state": proposal.current_state,
            },
            "tied_operators": [
                {"action": op.action, "tool": op.tool, "preference": op.preference}
                for op in decision.get("tied_operators", [])
            ],
        }


# Singleton instance
meta_agents = MetaAgentController()
