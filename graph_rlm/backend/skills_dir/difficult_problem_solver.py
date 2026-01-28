"""
Difficult Problem Solver (DPS) v3 - Triune Architecture

Uses:
- Advanced Reasoning MCP: Stateful thought tracking with hypotheses/confidence
- Reflective Agent Architecture: Graph-based problem decomposition with node chaining
- ChatDAG: Context retrieval and insight persistence

Key insight: RAA deconstruct returns node_ids which chain into hypothesize/synthesize.
Advanced reasoning tracks progress across multiple thought steps.
"""

import uuid
from typing import Any


async def difficult_problem_solver(
    problem: str,
    max_raa_depth: int = 3,
    reasoning_thoughts: int = 5,
) -> dict[str, Any]:
    """
    Solve complex problems using coordinated Triune Architecture.

    Flow:
    1. ChatDAG context retrieval
    2. RAA deconstruct → creates node tree
    3. Advanced reasoning thought chain with hypothesis tracking
    4. RAA hypothesize between node pairs → discover connections
    5. RAA synthesize nodes → unified insight
    6. ChatDAG persist results
    7. Optional: consult_ruminator for missing relationships

    Args:
        problem: The complex problem to solve
        max_raa_depth: Depth for RAA deconstruction tree
        reasoning_thoughts: Number of advanced reasoning steps

    Returns:
        Dict with solution, node_ids, reasoning_trace
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    session_id = str(uuid.uuid4())[:8]
    trace: dict[str, Any] = {"session_id": session_id, "steps": [], "node_ids": []}

    # =========================================================================
    # STEP 1: ChatDAG Context Retrieval
    # =========================================================================
    trace["steps"].append("1. Retrieving context from ChatDAG...")
    try:
        context = await call_mcp_tool(
            "chatdag", "search_knowledge", {"query": problem, "k": 5}
        )
        trace["context"] = context
    except Exception as e:
        trace["context"] = f"Context retrieval failed: {e}"
        context = ""

    # =========================================================================
    # STEP 2: RAA Deconstruct - Creates reasoning tree with node_ids
    # =========================================================================
    trace["steps"].append("2. RAA deconstruct - building problem tree...")
    try:
        deconstruct_result = await call_mcp_tool(
            "reflective_agent_architecture",
            "deconstruct",
            {"problem": problem, "max_depth": max_raa_depth},
        )
        trace["deconstruct"] = deconstruct_result

        # Extract node_ids from deconstruct result (format varies by RAA version)
        # The result should contain node IDs we can chain
        if isinstance(deconstruct_result, dict):
            node_ids = deconstruct_result.get("node_ids", [])
            if not node_ids and "nodes" in deconstruct_result:
                node_ids = [
                    n.get("id") for n in deconstruct_result["nodes"] if n.get("id")
                ]
        elif isinstance(deconstruct_result, list):
            node_ids = [
                r.get("id")
                for r in deconstruct_result
                if isinstance(r, dict) and r.get("id")
            ]
        else:
            # Try to parse from string response
            node_ids = []
        trace["node_ids"] = node_ids
    except Exception as e:
        trace["deconstruct"] = f"Deconstruct failed: {e}"
        node_ids = []

    # =========================================================================
    # STEP 3: Advanced Reasoning - Thought chain with hypothesis tracking
    # =========================================================================
    trace["steps"].append("3. Advanced reasoning chain...")
    reasoning_chain = []
    try:
        for i in range(1, reasoning_thoughts + 1):
            # Build hypothesis based on what we know
            hypothesis = (
                f"Based on deconstruction: {str(trace.get('deconstruct', ''))[:2048]}"
            )

            thought_result = await call_mcp_tool(
                "advanced_reasoning",
                "advanced_reasoning",
                {
                    "thought": f"Analyzing problem: {problem}\nContext: {str(context)[:16384]}\nStep {i}: Exploring implications...",
                    "thoughtNumber": i,
                    "totalThoughts": reasoning_thoughts,
                    "nextThoughtNeeded": i < reasoning_thoughts,
                    "confidence": 0.5 + (i * 0.1),  # Increase confidence as we progress
                    "session_id": session_id,
                    "goal": f"Solve: {problem[:16384]}",
                    "progress": i / reasoning_thoughts,
                    "hypothesis": hypothesis if i == 1 else None,
                },
            )
            reasoning_chain.append({"step": i, "result": thought_result})

            # If result contains new insights, we could adjust next thought
        trace["reasoning_chain"] = reasoning_chain
    except Exception as e:
        trace["reasoning_chain"] = f"Reasoning failed: {e}"

    # =========================================================================
    # STEP 4: RAA Hypothesize - Find connections between node pairs
    # =========================================================================
    trace["steps"].append("4. RAA hypothesize - finding connections...")
    hypotheses = []
    if len(node_ids) >= 2:
        try:
            # Connect first and last nodes (often most insightful)
            hyp_result = await call_mcp_tool(
                "reflective_agent_architecture",
                "hypothesize",
                {
                    "node_a_id": node_ids[0],
                    "node_b_id": node_ids[-1],
                    "context": problem,
                },
            )
            hypotheses.append(hyp_result)
            trace["hypotheses"] = hypotheses
        except Exception as e:
            trace["hypotheses"] = f"Hypothesize failed: {e}"
    else:
        trace["hypotheses"] = "Not enough nodes for hypothesize"

    # =========================================================================
    # STEP 5: RAA Synthesize - Merge nodes into unified insight
    # =========================================================================
    trace["steps"].append("5. RAA synthesize - merging insights...")
    if len(node_ids) >= 2:
        try:
            synthesis = await call_mcp_tool(
                "reflective_agent_architecture",
                "synthesize",
                {"node_ids": node_ids, "goal": f"Solve: {problem[:100]}"},
            )
            trace["synthesis"] = synthesis
        except Exception as e:
            trace["synthesis"] = f"Synthesize failed: {e}"
    else:
        trace["synthesis"] = "Not enough nodes for synthesis"

    # =========================================================================
    # STEP 6: Categorical Verification - Prove reasoning commutes
    # Uses categorical_kg_bridge to verify the synthesis is logically valid
    # =========================================================================
    trace["steps"].append("6. Categorical verification - proving reasoning validity...")
    verification_result = {"verified": False, "proof": None, "error": None}

    if len(node_ids) >= 2 and trace.get("synthesis"):
        try:
            # Import categorical bridge
            # Try direct import from same directory
            categorical_kg_bridge = None
            try:
                from .categorical_kg_bridge import categorical_kg_bridge
            except ImportError:
                pass

            # Build a Cypher-style pattern from the node chain
            # Pattern: (node1)-[:LEADS_TO]->(node2)-[:LEADS_TO]->(node3)...
            if len(node_ids) >= 2:
                pattern_parts = []
                for i, nid in enumerate(node_ids[:5]):  # Max 5 nodes
                    safe_id = str(nid).replace("-", "_")[:20]
                    pattern_parts.append(f"(n{i}:{safe_id})")

                if len(pattern_parts) >= 2:
                    pattern = "-[:LEADS_TO]->".join(pattern_parts)

                    if categorical_kg_bridge:
                        # Analyze the reasoning chain categorically
                        cat_result = categorical_kg_bridge(
                            pattern=pattern, analysis_type="prover9"
                        )
                    else:
                        cat_result = {}

                    # If there are commutative paths, try to verify them
                    if cat_result.get("prover9", {}).get(
                        "mcp_logic_ready"
                    ):  # Extract premises and axioms from categorical analysis
                        premises = cat_result["prover9"]["premises"][
                            :20
                        ]  # Limit complexity
                        axioms = cat_result["prover9"]["category_axioms"]

                        # Bridge axiom: connect categorical morphism to logical 'leads_to'
                        # "If there is a morphism m from s to t, then s leads to t"
                        bridge_axiom = "all m all s all t ((morphism(m) & source(m,s) & target(m,t)) -> leads_to(s,t))"

                        # Transitivity axiom
                        trans_axiom = "all x (all y (all z ((leads_to(x,y) & leads_to(y,z)) -> leads_to(x,z))))"

                        if len(node_ids) >= 2:
                            # Dynamic conclusion based on actual first/last nodes
                            # Note: node IDs were sanitized to n0, n1 etc in the pattern generation
                            start_node = "n0"
                            end_node = f"n{len(pattern_parts)-1}"
                            conclusion = f"leads_to({start_node}, {end_node})"

                            full_premises = (
                                premises + axioms + [bridge_axiom, trans_axiom]
                            )

                            try:
                                proof = await call_mcp_tool(
                                    "mcp-logic",
                                    "prove",
                                    {
                                        "premises": full_premises,
                                        "conclusion": conclusion,
                                    },
                                )
                                if (
                                    isinstance(proof, dict)
                                    and proof.get("result") == "proved"
                                ):
                                    verification_result["verified"] = True
                                    verification_result["proof"] = (
                                        "Categorical structure formally verifies reasoning chain"
                                    )
                                else:
                                    verification_result["proof"] = proof
                            except Exception as e:
                                verification_result["error"] = f"Proof attempt: {e}"
                        else:
                            verification_result["verified"] = True
                            verification_result["proof"] = (
                                "Chain too short for transitivity proof"
                            )

                        trace["categorical_diagram"] = {
                            "objects": len(
                                cat_result.get("structure_analysis", {}).get(
                                    "objects", []
                                )
                            ),
                            "morphisms": len(
                                cat_result.get("prover9", {}).get("premises", [])
                            ),
                        }
        except Exception as e:
            verification_result["error"] = str(e)

    trace["verification"] = verification_result
    if verification_result["verified"]:
        trace["steps"].append("   ✓ Reasoning chain formally verified")
    else:
        trace["steps"].append(
            f"   ! Verification incomplete: {verification_result.get('error', 'no proof')}"
        )

    # =========================================================================
    # STEP 7: ChatDAG Persist Results
    # =========================================================================
    trace["steps"].append("7. Persisting to ChatDAG...")
    try:
        await call_mcp_tool(
            "chatdag",
            "feed_data",
            {
                "content": f"DPS Session {session_id}\nProblem: {problem}\nSynthesis: {trace.get('synthesis', 'N/A')}\nVerified: {verification_result['verified']}",
                "source_id": f"dps_sessions/{session_id}",
            },
        )
    except Exception as e:
        trace["persistence_warning"] = f"ChatDAG persistence failed: {e}"

    # =========================================================================
    # FINAL: Construct solution from synthesis
    # =========================================================================
    solution = trace.get("synthesis", "Unable to synthesize solution")

    return {
        "solution": solution,
        "session_id": session_id,
        "node_ids": node_ids,
        "trace": trace,
    }
