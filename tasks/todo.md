# Graph-RLM: Principled Computational Physics Engine Upgrade

**Goal:** Elevate Graph-RLM from an agentic loop into a bounded physics/topology engine by implementing Hamiltonian dynamics, Čech cohomology, thermodynamic penalties, and MDL formalization.
**Acceptance Criteria:**

- NLP heuristics in `sheaf.py` are completely replaced by mathematical boundary operators.
- The agent naturally traverses optimal geodesics without hallucinating forced completions.
- Memory consolidation relies on algorithmic compression (MDL) rather than LLM summarization.

---

## [x] Phase 1: Rulial Topology & The Boundary of Chaos (Priority 1)

- [x] **Restate goal:** Replace `sheaf.py` keyword heuristics with dynamic Laplacian edge weights and Čech cohomology checks in `topology.py`.
- [x] **Locate existing:** `src/core/topology.py` (compute_sheaf_laplacian), `src/core/sheaf.py` (line 411 verification obstruction).
- [x] **Design:**
  - Define semantic restriction maps using normalized embedding distance (Intent vs Outcome).
  - Use distance to weight edges in `compute_sheaf_laplacian` rather than fixed `1.0`.
  - Calculate `inconsistency_energy` mathematically via the Laplacian. Set `EDGE_OF_CHAOS` bounds.
- [x] **Implement smallest safe slice:** Modify `topology.py` to accept nodes and compute semantic divergences dynamically. Update `sheaf.py` to use these topological metrics.
- [x] **Add/adjust tests:** Write a unit test simulating a logic loop and verify the consistency energy spikes mathematically, without keyword triggers.
- [x] **Run verification:** Run test suites `test_auto_pruning.py` and `verify_empirical_grounding.py`.
- [x] **Summarize changes:** Document how topological defects automatically trigger stress metrics.
- [x] **Record lessons:** Update `tasks/lessons.md`.

## [/] Phase 2: The "Rosetta Stone" (Hamiltonian Dynamics)

- [ ] **Restate goal:** Unify `omcd.py` and `sheaf.py` to evaluate thought continuation as a Hamiltonian energy conservation problem.
- [ ] **Locate existing:** `src/core/omcd.py`, `src/core/agent.py`.
- [ ] **Design:**
  - $V$ (Potential Energy) = Semantic distance to the global goal.
  - $T$ (Kinetic Energy) = Computational cost exerted.
  - Enforce energy conservation: reject thoughts requiring a massive "energy jump" (hallucinations/unearned leaps).
- [ ] **Implement smallest safe slice:** Update `omcd.evaluate_step` to calculate a Hamiltonian invariant $H = T + V$, failing states where $H$ diverges illegally.
- [ ] **Add/adjust tests:** Simulate an agent hallucinating a massive, unjustified leap in logic (like FalkorDBProxy) and verify rejection.
- [ ] **Run verification:** Run test suite against Hamiltonian constraints.
- [ ] **Summarize changes:** Document mapping of Gradient Descent to Principle of Least Action.
- [ ] **Record lessons:** Update `tasks/lessons.md`.

## Phase 3: Thermodynamic Cost of Determinism

- [ ] **Restate goal:** Penalize Intrinsic Reward in `navigator.py` when the agent forces a deterministic bottleneck, restricting freedom rapidly.
- [ ] **Locate existing:** `src/core/navigator.py` (compute_interest_gradient).
- [ ] **Design:** Subtract a "Determinism Penalty" from the Curiosity Score if `current_s_tau` (Freedom) drops severely compared to `baseline_s_tau` across consecutive steps.
- [ ] **Implement smallest safe slice:** Introduce the scalar penalty to `raw_score` in `compute_interest_gradient`.
- [ ] **Add/adjust tests:** Provide mock histories where `Freedom` crashes to 0; verify the Navigator discourages that path.
- [ ] **Run verification:** Lint, typecheck, run REPL simulations.
- [ ] **Summarize changes:** Document how agent now naturally follows the geodesic of the Multiway Graph.
- [ ] **Record lessons:** Update `tasks/lessons.md`.

## Phase 4: Gestalt Formalization via MDL and Persistent Homology

- [ ] **Restate goal:** Replace LLM Gestalt summarization with Structural Information Theory (MDL via LZMA) and Persistent Homology for memory caching.
- [ ] **Locate existing:** `src/core/thimac_memory.py`, `src/core/navigator.py` (compute_compression_size).
- [ ] **Design:**
  - Utilize compression ratios as the sole heuristic for the "Law of Prägnanz".
  - Track nodes via Persistent Homology over multiple pruning cycles; survivors define "Closure" and are promoted to `AXIOMS`.
- [ ] **Implement smallest safe slice:** Refactor `_sync_thimac` to rely on `navigator` compression progress to bucket insights rather than extracting summaries natively.
- [ ] **Add/adjust tests:** Verify empirical grounding drops noisy processes but preserves highly compressible sequences natively.
- [ ] **Run verification:** Run `verify_empirical_grounding.py`.
- [ ] **Summarize changes:** Document translation of psychological Gestalt into pure computation.
- [ ] **Record lessons:** Update `tasks/lessons.md`.
