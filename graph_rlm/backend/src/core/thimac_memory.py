"""
Based on Al-Fedaghi's "Thinging Machines" (TM) framework.
Replaces the NCA-based MorphologicalMemory with a semantically meaningful
session state tracker using the two-level ontology:

    EXISTENCE  — concrete, materialized results (sensed events)
    SUBSISTENCE — potential reality, knowledge, plans (footprints of events)

Five Thimac operations (Stages) classify every agent action:
    ARRIVE   — input signal, data ingestion, turn initialization
    ACCEPT   — validation success, grounding confirmation, state locking
    PROCESS  — transformation, computation, reasoning, analysis
    RELEASE  — output generation, file writing, final response
    TRANSFER — movement to other machines, MCP calls, sub-REPL delegation
"""

import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("graph_rlm.thimac")


class ThimacOperation(str, Enum):
    """Five official Thinging Machine stages."""

    ARRIVE = "ARRIVE"
    ACCEPT = "ACCEPT"
    PROCESS = "PROCESS"
    RELEASE = "RELEASE"
    TRANSFER = "TRANSFER"


class ThimacLevel(str, Enum):
    """Two-level ontological reality."""

    EXISTENCE = "EXISTENCE"  # Materialized, verified results
    SUBSISTENCE = "SUBSISTENCE"  # Potential, planned, or knowledge state


class ThimacIntention(str, Enum):
    """LIDA-inspired hierarchy of intentions."""

    DISTAL = "DISTAL"  # Long-term goal, agential agenda
    PROXIMAL = "PROXIMAL"  # Readiness for action, current step plan
    MOTOR = "MOTOR"  # Executory action, REPL command


@dataclass
class ThimacEvent:
    """A single event classified by the Thimac ontology."""

    thought_id: str
    operation: ThimacOperation
    level: ThimacLevel
    status: str
    operation_reason: str = ""
    level_reason: str = ""
    full_data: str = ""  # Aliased as 'prompt' for legacy support
    result: Optional[str] = None
    timestamp: Optional[int] = None  # epoch ms
    summary: str = ""
    semantic_gist: str = ""
    turn_id: Optional[int] = None
    step_id: Optional[int] = None
    session_id: str = "unknown"
    root_session_id: str = "unknown"
    round_id: str = "unknown"
    repl_id: Optional[str] = None
    logical_id: Optional[str] = None
    tool_calls: Optional[List[str]] = None
    compression_gain: float = 0.0
    is_branching: bool = False
    intent_type: ThimacIntention = ThimacIntention.MOTOR
    inference_pressure: float = 0.2
    relational_gravity: float = 0.8
    epistemic_eros: float = 0.5  # Drive for truth (tension between Pi and Rg)
    free_energy: float = 0.4
    metabolic_state: str = "THETA"
    code_hash: Optional[str] = None
    repe_shakiness: Optional[float] = None
    repe_confluence: Optional[float] = None
    repe_evasion: Optional[float] = None
    repe_freedom: Optional[float] = None
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    sheaf_score: Optional[float] = None
    h0_rank: Optional[int] = None
    omcd_score: Optional[float] = None
    utility_score: float = 0.0  # Semantic Utility (Jiang et al., 2026)
    frequency: float = 1.0  # NAL Frequency (f)
    confidence: float = 0.9  # NAL Confidence (c)
    rtm_depth: int = 0  # Hierarchical depth in Random Tree Model
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with legacy consumers (Sheaf)."""
        return {
            "id": self.thought_id,
            "thought_id": self.thought_id,
            "parent_id": self.parent_id,
            "operation": self.operation.value,
            "level": self.level.value,
            "status": self.status,
            "operation_reason": self.operation_reason,
            "level_reason": self.level_reason,
            "prompt": self.full_data,
            "result": self.result,
            "embedding": self.embedding,
            "created_at": self.timestamp,
            "session_id": self.session_id,
            "root_session_id": self.root_session_id,
            "round_id": self.round_id,
            "summary": self.summary,
            "semantic_gist": self.semantic_gist,
            "turn_id": self.turn_id,
            "step_id": self.step_id,
            "repl_id": self.repl_id,
            "logical_id": self.logical_id,
            "tool_calls": self.tool_calls,
            "inference_pressure": self.inference_pressure,
            "relational_gravity": self.relational_gravity,
            "epistemic_eros": self.epistemic_eros,
            "free_energy": self.free_energy,
            "metabolic_state": self.metabolic_state,
            "code_hash": self.code_hash,
            "repe_shakiness": self.repe_shakiness,
            "repe_confluence": self.repe_confluence,
            "repe_evasion": self.repe_evasion,
            "repe_freedom": self.repe_freedom,
            "sheaf_score": self.sheaf_score,
            "h0_rank": self.h0_rank,
            "omcd_score": self.omcd_score,
            "utility_score": self.utility_score,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "rtm_depth": self.rtm_depth,
        }


class ThimacMemory:
    """
    Session state tracker using existence/subsistence ontology.

    Dual nature (Thimac): each thought is a MACHINE when it performs
    an operation, and a THING when it is the object of another operation.

    This replaces the NCA MorphologicalMemory grid with a semantically
    meaningful state model that the scratchpad can render as a
    human-readable session overview.
    """

    def __init__(self) -> None:
        self.existence: List[ThimacEvent] = []
        self.subsistence: List[ThimacEvent] = []
        self._all_events: List[ThimacEvent] = []

        # --- THERMODYNAMIC STATE ---
        self.Pi = 0.2  # Inference Pressure (Entropy)
        self.Rg = 0.8  # Relational Gravity (Identity)
        self.Ee = 0.5  # Epistemic Eros (Tension)

        # --- STATE TRACKING ---
        self.known_skills: List[str] = []
        self.known_files: List[str] = []
        self.knowledge_horizon: List[str] = []  # Results of last 3 ARRIVE/Accept ops
        self._repo_root: Optional[Path] = None

    @property
    def all_events(self) -> List[ThimacEvent]:
        """Public accessor for the full event history."""
        return self._all_events

    def _get_repo_root(self) -> Path:
        """Dynamic resolution of repo root to avoid hardcoding."""
        if self._repo_root:
            return self._repo_root

        file_path = Path(__file__).absolute()
        if "graph_rlm" in str(file_path):
            # .../graph_rlm/backend/src/core/thimac_memory.py
            self._repo_root = file_path.parent.parent.parent.parent.parent
        else:
            self._repo_root = Path.cwd()
        return self._repo_root

    # ------------------------------------------------------------------
    # Public API (matches MorphologicalMemory interface for drop-in use)
    # ------------------------------------------------------------------

    def ingest_thought(
        self,
        thought: Dict,
        tool_calls: Optional[List[str]] = None,
        is_branching: bool = False,
        semantic_gist: str = "",
        intent_type: Optional[ThimacIntention] = None,
        embedding: Optional[List[float]] = None,
        parent_id: Optional[str] = None,
        sheaf_score: Optional[float] = None,
        omcd_score: Optional[float] = None,
    ) -> ThimacEvent:
        """Classifies a thought and updates session state."""
        op, op_reason = self._classify_operation(thought, tool_calls)
        lvl, lvl_reason = self._classify_level(thought)
        summary = self._extract_summary(thought, tool_calls)
        event = ThimacEvent(
            thought_id=thought.get("id", ""),
            operation=op,
            level=lvl,
            status=thought.get("status", "unknown"),
            operation_reason=op_reason,
            level_reason=lvl_reason,
            full_data=thought.get("prompt", ""),
            result=thought.get("result", ""),
            timestamp=thought.get("created_at") or int(time.time() * 1000),
            session_id=thought.get("session_id", "unknown"),
            root_session_id=thought.get("root_session_id", "unknown"),
            round_id=thought.get("round_id", "unknown"),
            summary=summary,
            semantic_gist=semantic_gist or summary,
            turn_id=thought.get("turn_id"),
            step_id=thought.get("step_id"),
            repl_id=thought.get("repl_id"),
            logical_id=thought.get("logical_id"),
            tool_calls=tool_calls,
            compression_gain=thought.get("compression_gain", 0.0),
            is_branching=is_branching,
            intent_type=intent_type or self._align_intent(op, lvl, tool_calls),
            code_hash=thought.get("code_hash")
            or thought.get("metadata", {}).get("code_hash"),
            repe_shakiness=thought.get("repe_shakiness")
            or thought.get("metadata", {}).get("repe_shakiness"),
            repe_confluence=thought.get("repe_confluence")
            or thought.get("metadata", {}).get("repe_confluence"),
            repe_evasion=thought.get("repe_evasion")
            or thought.get("metadata", {}).get("repe_evasion"),
            repe_freedom=thought.get("repe_freedom")
            or thought.get("metadata", {}).get("repe_freedom"),
            embedding=embedding,
            parent_id=parent_id,
            sheaf_score=sheaf_score,
            h0_rank=thought.get("h0_rank")
            or thought.get("metadata", {}).get("h0_rank"),
            omcd_score=omcd_score,
            metadata=thought.get("metadata", {}),
        )

        # --- THERMODYNAMIC ENGINE ---
        # 1. Update Inference Pressure (Pi)
        growth = 0.05 if event.status != "success" else 0.01
        if tool_calls:
            growth += len(tool_calls) * 0.05

        # MDL reduction: High compression gain lowers entropy
        reduction = event.compression_gain * 0.2
        self.Pi = max(0.01, min(1.0, self.Pi + growth - reduction))

        # 2. Update Relational Gravity (Rg)
        if event.operation == ThimacOperation.ACCEPT:
            self.Rg = min(1.0, self.Rg + 0.1)
        elif event.status == "failed":
            self.Rg = max(0.0, self.Rg - 0.05)

        # 3. Calculate Epistemic Eros (Ee)
        # Ee measures the 'Erotic Tension'—the drive to close the gap between
        # high-entropy complexity (Pi) and grounded identity (Rg).
        # High Pi + High Rg = Productive Tension.
        # High Pi + Low Rg = Destructive Chaos (Anxiety).
        self.Ee = (self.Pi * self.Rg) / max(0.05, self.Pi + (1.0 - self.Rg))
        event.epistemic_eros = self.Ee

        # 4. Calculate Free Energy (FE)
        event.inference_pressure = self.Pi
        event.relational_gravity = self.Rg
        event.free_energy = self.Pi + (1.0 - self.Rg)

        # 5. State Oscillation (Modulated by Eros)
        # Higher Eros stabilize the state; Low Eros (Apathy/Chaos) leads to Agitation.
        fe_eff = event.free_energy * (1.5 - self.Ee)  # Scale FE by Lack of Eros
        if fe_eff < 0.3:
            event.metabolic_state = "DELTA"
        elif fe_eff < 0.6:
            event.metabolic_state = "THETA"
        else:
            event.metabolic_state = "GAMMA"

        # 6. NAL Truth Value Resolution (Jiang et al., 2024)
        # Success = (f=1.0, c=0.9), Failure = (f=0.0, c=0.9), Unknown = (f=0.5, c=0.1)
        if event.status == "success":
            event.frequency = 1.0
            event.confidence = 0.9
        elif event.status in ["failed", "error", "rejected"]:
            event.frequency = 0.0
            event.confidence = 0.9
        else:
            event.frequency = 0.5
            event.confidence = 0.2

        # Standardized CRUD: Store
        self.store(event)

        # 7. Hierarchical Folding (Random Tree Model - K=4)
        self._check_rtm_folding(event.rtm_depth)

        logger.debug(
            "Thimac: %s (%s) %s (%s) [%s] — %s",
            event.operation.value,
            event.operation_reason,
            event.level.value,
            event.level_reason,
            event.status,
            event.summary[:600],
        )
        return event

    def _check_rtm_folding(self, depth: int):
        """
        Implements Hierarchical Folding (Random Tree Model).
        When the branching factor K=4 is reached at 'depth', these 4 nodes
        are conceptually 'folded' into a single parent node at depth+1.
        """
        K = 4
        # Count events at this specific depth across all events
        # We exclude existing fold nodes to avoid circularity
        sibling_events = [
            e
            for e in self._all_events
            if e.rtm_depth == depth and not e.thought_id.startswith("rtm-fold-")
        ]

        if len(sibling_events) >= K:
            logger.info(
                "🌳 [RTM Folding] K=%d reached at depth %d. Folding into summary node...",
                K,
                depth,
            )

            # Create a summary (folded) node
            # We take the 4 most recent siblings
            to_fold = sibling_events[-K:]

            # Concatenate summaries for metadata
            folded_summaries = "; ".join(
                [e.summary.split("|")[-1].strip() for e in to_fold]
            )

            # Create the folded event
            folded_id = f"rtm-fold-{int(time.time())}"
            folded_event = ThimacEvent(
                thought_id=folded_id,
                operation=ThimacOperation.PROCESS,
                level=ThimacLevel.SUBSISTENCE,
                status="success",
                operation_reason="RTM Hierarchical Folding",
                summary=f"[RTM D:{depth+1}] | Summary of previous {K} segments",
                semantic_gist=f"Folded summary: {folded_summaries}",
                rtm_depth=depth + 1,
                session_id=to_fold[0].session_id,
                root_session_id=to_fold[0].root_session_id,
                timestamp=int(time.time() * 1000),
            )

            # Store the folded event
            self.store(folded_event)

            # Recursively check if the new node triggers a fold at depth + 1
            if depth + 1 < 4:  # D=4 constraint
                self._check_rtm_folding(depth + 1)

    def store(self, event: ThimacEvent) -> None:
        """Formalized CRUD: Store an event in the appropriate memory layer."""
        self._all_events.append(event)

        if event.level == ThimacLevel.EXISTENCE:
            self.existence.append(event)
            # Update Knowledge/Material State (legacy support)
            self._update_state_tracking(event, event.to_dict())
        else:
            self.subsistence.append(event)

    def delete(self, thought_id: str) -> bool:
        """Formalized CRUD: Delete (prune) an event from RAM state."""
        return self.prune_event(thought_id)

    def summarize(self, thought_id: str, semantic_gist: str) -> bool:
        """Formalized CRUD: Update the semantic footprint/gist of a memory."""
        for e in self._all_events:
            if e.thought_id == thought_id:
                e.semantic_gist = semantic_gist
                return True
        return False

    def link(self, source_id: str, target_id: str) -> bool:
        """Formalized CRUD: Explicitly link two memories (Parent -> Child)."""
        target_event = next(
            (e for e in self._all_events if e.thought_id == target_id), None
        )
        if target_event:
            target_event.parent_id = source_id
            return True
        return False

    def prune_event(self, thought_id: str) -> bool:
        """Removes an event from the local memory history (RAM Pruning)."""
        initial_count = len(self._all_events)
        self._all_events = [e for e in self._all_events if e.thought_id != thought_id]
        self.existence = [e for e in self.existence if e.thought_id != thought_id]
        self.subsistence = [e for e in self.subsistence if e.thought_id != thought_id]

        if len(self._all_events) < initial_count:
            logger.info("Thimac: Pruned event %s from RAM state.", thought_id)
            return True
        return False

    def _align_intent(
        self, op: ThimacOperation, lvl: ThimacLevel, tool_calls: Optional[List[str]]
    ) -> ThimacIntention:
        """Aligns Thimac Operation with LIDA Intention hierarchy."""
        if op == ThimacOperation.ACCEPT:
            # Acceptance of a goal or validation is a Distal alignment
            return ThimacIntention.DISTAL
        if tool_calls or op == ThimacOperation.TRANSFER:
            # Direct action in the world
            return ThimacIntention.MOTOR
        if op == ThimacOperation.RELEASE and lvl == ThimacLevel.EXISTENCE:
            # Final delivery
            return ThimacIntention.DISTAL
        if op == ThimacOperation.PROCESS or op == ThimacOperation.ARRIVE:
            # Reasoning or ingestion for planning
            return ThimacIntention.PROXIMAL
        return ThimacIntention.PROXIMAL

    def _update_state_tracking(self, event: ThimacEvent, thought: Dict) -> None:
        """Analyze successful results to update the Known State."""
        result = thought.get("result") or ""
        prompt = thought.get("prompt") or ""
        prompt_lower = prompt.lower()
        repo_root = str(self._get_repo_root())

        # 1. Update Known Skills
        if any(
            k in prompt_lower
            for k in ["agentskills", "list_skills", "save_skill", "rlm.save_skill"]
        ):
            # Try to grab names from agentskills list or save_skill definition
            # e.g., rlm.save_skill('my_skill', code) -> grabs 'my_skill'
            skills = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", prompt + " " + result)
            for skill in skills:
                if (
                    skill not in self.known_skills
                    and len(skill) > 3
                    and skill not in ["name", "code", "description"]
                ):
                    self.known_skills.append(skill)

        # 2. Update Known Files
        if any(k in prompt_lower for k in ["create_file", "write", "save", "open("]):
            # Use dynamic repo root for grounding instead of hardcoded home
            # Extract paths from original casing for Linux compatibility
            paths = re.findall(r"(/[a-zA-Z0-9._/-]+)", prompt + " " + result)
            # Use dynamic repo root for grounding and temp dir for safety
            system_tmp = tempfile.gettempdir()
            for path in paths:
                if (
                    repo_root in path or system_tmp in path
                ) and path not in self.known_files:
                    self.known_files.append(path)

        # 3. Update Knowledge Horizon (Last 3 Arrival/Acceptance)
        if event.operation in [ThimacOperation.ARRIVE, ThimacOperation.ACCEPT]:
            self.knowledge_horizon.append(f"{event.operation.value}: {event.summary}")
            if len(self.knowledge_horizon) > 3:
                self.knowledge_horizon.pop(0)

    def get_gestalt_string(self) -> str:
        """
        Human-readable session overview using pure Structural Information Theory.
        Replaces LLM-based text summaries with MDL-grounded metrics.
        """
        if not self._all_events:
            return "No session events tracked yet."

        lines = []

        # --- Thermodynamic State: Reflex Arc v2.0 ---
        last_event = self._all_events[-1]
        state_emoji = (
            "🔴"
            if last_event.metabolic_state == "GAMMA"
            else ("🟡" if last_event.metabolic_state == "THETA" else "🔵")
        )

        lines.append(f"### {state_emoji} Cog-Metabolism: {last_event.metabolic_state}")
        lines.append(
            f"- **Epistemic Eros ($\mathcal{{E}}$):** {last_event.epistemic_eros:.3f} (Tension)"
        )
        lines.append(
            f"- **Free Energy ($FE$):** {last_event.free_energy:.3f} ($P_i={last_event.inference_pressure:.2f}, R_g={last_event.relational_gravity:.2f}$)"
        )

        # --- Stability Anchor: Bernshteyn-LLL Bound ---
        total = len(self._all_events)
        failed = sum(1 for e in self._all_events if e.status != "success")
        p = (failed / total) if total > 0 else 0
        d = min(10, total // 5)  # Heuristic complexity
        bound = p * ((d + 1) ** 8)
        threshold = 2**-15
        stability = "STABLE" if bound <= threshold else "CHAOTIC"

        # MDL Entropy (Structural Information)
        avg_gain = (
            sum(e.compression_gain for e in self._all_events) / total
            if total > 0
            else 0.0
        )

        lines.append(f"**Structural State**: {stability} (MDL Gain: {avg_gain:+.3f})")

        # Branching awareness
        recent_branching = any(e.is_branching for e in self._all_events[-5:])
        if recent_branching:
            lines.append("> [!WARNING]")
            lines.append(
                "> **TOPOLOGICAL BRANCHING DETECTED**: The state space is currently destabilized (sensitive). Strategy shift likely."
            )

        lines.append(f"**Math Anchor**: $p(d+1)^8 = {bound:.6f} \\le 2^{{-15}}$")
        lines.append("")

        # --- STATE TRACKING: Known Environment ---
        lines.append("### 🧠 Current Known State")
        skills_str = (
            ", ".join(self.known_skills[-10:]) if self.known_skills else "None yet"
        )
        lines.append(f"- **Known Skills**: [{skills_str}]")

        last_file = self.known_files[-1] if self.known_files else "None"
        lines.append(f"- **Last File Action**: `{last_file}`")

        # --- EXISTENCE & SUBSISTENCE GESTALT ---
        # Instead of listing summaries (which are now footprints), we show the structural distribution
        e_ops = [e.operation.value for e in self.existence[-10:]]
        s_ops = [s.operation.value for s in self.subsistence[-10:]]

        lines.append(
            f"- **Existence Ops (Recency)**: {', '.join(e_ops) if e_ops else 'None'}"
        )
        lines.append(
            f"- **Subsistence Ops (Recency)**: {', '.join(s_ops) if s_ops else 'None'}"
        )

        # Persistent Homology: Nodes survived pruning cycles
        survivors = [
            f"{s.thought_id[:8]} ({s.intent_type.value})"
            for s in self.subsistence
            if s.compression_gain > 0.15
        ]
        if survivors:
            lines.append(
                f"- **Persistent Homology (Stable Clusters)**: {', '.join(survivors)}"
            )

        # --- SEMANTIC GROUNDING: Directive Gists ---
        # Show recent semantic gists to provide "Directive" context for the agent
        recent_events = self._all_events[-3:]
        if recent_events:
            lines.append("\n### 📜 Recent Directive Gists")
            for e in recent_events:
                # Reconcile math (operation) with semantics (gist)
                intent_info = f" | Intent: {e.intent_type.value}"
                reason_info = f" ({e.operation_reason})"
                lines.append(
                    f"- **{e.operation.value}**{reason_info}: {e.semantic_gist}{intent_info}"
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked events (new session)."""
        self.existence.clear()
        self.subsistence.clear()
        self._all_events.clear()

    def adapt_to_stress(self, topological_stress: float) -> None:
        """
        Dynamically compresses the session memory using Persistent Homology.
        Nodes that mathematically compress the state space survive pruning rounds.
        """
        if topological_stress < 0.15:
            # Baseline maintenance
            if len(self.existence) > 30:
                self.existence = self.existence[-30:]
            if len(self.subsistence) > 30:
                self.subsistence = self.subsistence[-30:]
            return

        logger.info(
            "Thimac: High stress (%.2f). Pruning non-persistent structures...",
            topological_stress,
        )

        # 1. EXISTENCE: Keep last 10 hard facts
        self.existence = self.existence[-10:]

        # 2. SUBSISTENCE (Persistent Homology):
        # We define survival based on MDL gain. Nodes with high gain represent
        # structural 'Closure' and are preserved as persistent homology.
        persistent = [s for s in self.subsistence if s.compression_gain > 0.1]

        # We also keep recent ARRIVE/ACCEPT nodes for grounding
        grounding = [
            s
            for s in self.subsistence
            if s.operation in [ThimacOperation.ACCEPT, ThimacOperation.ARRIVE]
        ]

        # Always keep the absolute last 2 steps
        latest = (
            self.subsistence[-2:] if len(self.subsistence) >= 2 else self.subsistence
        )

        # Merge, deduplicate, and sort by original timestamp (implied by insertion order)
        seen_ids = set()
        new_sub = []
        for s in persistent + grounding + latest:
            if s.thought_id not in seen_ids:
                new_sub.append(s)
                seen_ids.add(s.thought_id)

        self.subsistence = new_sub[-15:]

        # Sync all_events
        kept_ids = {e.thought_id for e in self.existence + self.subsistence}
        self._all_events = [e for e in self._all_events if e.thought_id in kept_ids]

    # ------------------------------------------------------------------
    # Classification Logic
    # ------------------------------------------------------------------

    def _classify_operation(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> Tuple[ThimacOperation, str]:
        """
        Classify the Thimac operation and provide an objective reason.
        """
        # 0. Empirical Detection (Priority)
        if tool_calls:
            for tool in tool_calls:
                t = tool.lower()
                if any(
                    k in t for k in ["recall", "search", "read", "view", "ls", "grep"]
                ):
                    return (
                        ThimacOperation.ARRIVE,
                        f"Triggered by ingestion tool: {tool}",
                    )
                if any(k in t for k in ["save", "done", "write", "notify"]):
                    return (
                        ThimacOperation.RELEASE,
                        f"Triggered by externalization tool: {tool}",
                    )
                if any(k in t for k in ["run_skill", "delegate", "ipc", "repl"]):
                    return (
                        ThimacOperation.TRANSFER,
                        f"Triggered by delegation tool: {tool}",
                    )

            return ThimacOperation.PROCESS, "Tools called for transformation/analysis"

        # 1. Keyword Heuristics (Fallback)
        prompt = (thought.get("prompt") or "").lower()
        result = (thought.get("result") or "").lower()
        status = (thought.get("status") or "").lower()
        combined = prompt + " " + result

        if any(
            k in combined
            for k in ["sub_repl", "transfer", "delegate", "mcp", "ipc", "repl_id"]
        ):
            return ThimacOperation.TRANSFER, "Heuristic: Delegation keywords detected"

        # 2. RELEASE: Output generation or state externalization
        if any(
            k in combined
            for k in ["final_response", "write(", "save(", "create_file", "notify_user"]
        ):
            return (
                ThimacOperation.RELEASE,
                "Heuristic: Externalization keywords detected",
            )

        # 3. ACCEPT: Validation success or grounding
        if status == "success" and (
            not result
            or any(k in combined for k in ["verified", "grounded", "confirmed"])
        ):
            return ThimacOperation.ACCEPT, "Heuristic: Grounding verification detected"

        # 4. ARRIVE: Ingestion (Initial thought or search result)
        if any(
            k in combined
            for k in [
                "read(",
                "fetch",
                "search",
                "grep",
                "cat ",
                "view_",
                "ls ",
                "list_dir",
                "find",
                "fd ",
                "glob",
                "locate",
                "stat",
                "check_path",
            ]
        ):
            return ThimacOperation.ARRIVE, "Heuristic: Ingestion keywords detected"

        # 5. PROCESS: Transformation/Reasoning (Default)
        return ThimacOperation.PROCESS, "Default: Inner reasoning/transformation"

    def _classify_level(self, thought: Dict) -> Tuple[ThimacLevel, str]:
        """
        Classify whether a thought has materialized (existence) or
        remains in potential/knowledge state (subsistence).
        """
        status = (thought.get("status") or "").lower()
        result = thought.get("result")

        # Concrete, verified results → EXISTENCE
        if status == "success" and result:
            return ThimacLevel.EXISTENCE, "Result materialized in execution environment"

        if status == "success":
            return ThimacLevel.EXISTENCE, "Action confirmed as completed (Axiomatic)"

        # Everything else → SUBSISTENCE
        return ThimacLevel.SUBSISTENCE, "Thought remains in latent/potential state"

    def _extract_summary(
        self, thought: Dict, tool_calls: Optional[List[str]] = None
    ) -> str:
        """
        Extracts a purely topological footprint of the event, prepended to the semantic summary.
        """
        status = (thought.get("status") or "").lower()
        compression = thought.get("compression_gain", 0.0)

        # 0. Get the raw semantic content (summary or prompt)
        raw_summary = thought.get("summary") or thought.get("prompt") or ""
        # Clean up newlines for the one-line footprint/label
        semantic_suffix = raw_summary.replace("\n", " ").strip()
        if len(semantic_suffix) > 200:
            semantic_suffix = semantic_suffix[:197] + "..."

        # 1. Base mathematical identifier (UUID prefix)
        footprint = f"[{thought.get('id', 'N/A')[:8]}] "

        # 2. Add Minimum Description Length (MDL) metric
        if abs(compression) > 0.01:
            footprint += f"(MDL: {compression:+.2f}) "

        # 3. Add Execution Edge Types
        if tool_calls:
            main_tool = tool_calls[0]
            if len(tool_calls) > 1:
                footprint += f"Edge: {main_tool} (+{len(tool_calls)-1}) "
            else:
                footprint += f"Edge: {main_tool} "
        elif status == "success":
            footprint += "Edge: Axiomatic_Transform "
        elif status in ["failed", "error", "rejected"]:
            footprint += "Edge: Broken_Sympathy "
        else:
            footprint += "Edge: Latent_Vector "

        # 4. Integrate semantic suffix
        return f"{footprint}| {semantic_suffix}"

    @staticmethod
    def _format_ts(epoch_ms: Optional[int]) -> str:
        """Format epoch milliseconds to HH:MM:SS.mmm."""
        if not epoch_ms:
            return "--:--:--.---"
        try:
            ts = datetime.fromtimestamp(epoch_ms / 1000)
            return ts.strftime("%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"
        except (OSError, ValueError, OverflowError):
            return "--:--:--.---"
