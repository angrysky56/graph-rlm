# Graph-RLM Self-Healing Pipeline

The self-healing mechanism in Graph-RLM is a multi-tiered system designed to move from **Reactive Resolution** (fixing errors) to **Adaptive Prevention** (learning from them). It is modeled after biological immune systems.

## Tier 1: Innate Immunity (Reactive Resolution)
This is the immediate response to failures during REPL execution.
- **Dependency Healing**: Detects `ModuleNotFoundError`, identifies the missing package via regex, installs it via `uv`, and triggers an immediate **Recursive Retry** of the same code block.
- **Syntax/Logic Healing**: Detects `Exception` or `AssertionError`. Instead of returning a failure to the user, the system injects a `SYSTEM REFLEXION` node into the Graph.
- **Execution Budgeting**: Monitors `REPL_TIMEOUT`. If a hang is detected, the process is killed, and a "timeout" constraint is injected to force the agent to optimize its next attempt.

## Tier 2: Epistemic Integrity (Proactive Filtering)
This layer operates *before* code execution to prevent known failure modes.
- **Axiomatic Verification (CAG)**: Every proposed Python action is passed through `sheaf.check_axiomatic_consistency`. It compares the code against the **Axiom Library**. If it violates a known constraint (e.g., mass loss in a physics simulation), it is blocked.
- **Sheaf Topology Monitor**: Measures "Consistency Energy" across the thought graph. High energy (Logical Knots) indicates the agent is looping or contradicting itself. This triggers a **Militant Reflexion** that overrides the LLM's current path.
- **RepE (Representation Engineering)**: Scans the latent embeddings of thoughts for "Pathogens" (laziness, obsequiousness, malice). If detected, it steers the reasoning away from these states.

## Tier 3: Adaptive Immunity (Meta-Cognitive Learning)
This is where the system "dreams" to get smarter.
- **The Dreamer**: A background process ([dream.py](file:///home/ty/Repositories/ai_workspace/graph-rlm/graph_rlm/backend/src/core/dream.py)) that periodically sweeps the graph for nodes marked `STATUS: FAILED` or `REFLEXION`.
- **Insight Synthesis**: The Dreamer analyzes the *delta* between the failure and the eventual success. It synthesizes an **Insight**—a general rule that explains why the failure happened and how to avoid it.
- **Rule Codification**: These insights are appended to [rules.md](file:///home/ty/Repositories/ai_workspace/graph-rlm/graph_rlm/backend/rules.md) (immediate guardrails) or codified as permanent **Axioms** in the Knowledge Base (used by CAG).

## The Pipeline Summary
1. **FAIL**: REPL throws an error or Sheaf finds a knot.
2. **REFLECT**: A Critic (Sheaf/Axiom) injects a verbal correction into the Graph.
3. **RECURSE**: The Agent consumes the correction and tries again (Reactive).
4. **DREAM**: The Dreamer analyzes the failure sequence after the task is "Done".
5. **CURE**: A new rule is created, preventing that specific class of error from ever occurring again.

---
> [!IMPORTANT]
> Package installation is the simplest form of Tier 1. The "real" self-healing is Tier 3—the ability of the system to update its own [rules.md](file:///home/ty/Repositories/ai_workspace/graph-rlm/graph_rlm/backend/rules.md) so that it stops making the same mistakes over time.
