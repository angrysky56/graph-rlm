"""
Auto-generated wrapper for reflective-agent-architecture MCP server.

This module provides Python function wrappers for all tools
exposed by the reflective-agent-architecture server.

Do not edit manually.
"""

from typing import Any


def deconstruct(problem: str, max_depth: int | None = None) -> Any:
    """Break a complex problem into component thought-nodes with hierarchical relationships. Creates a reasoning tree similar to Meta's COCONUT but materialized as a queryable graph.

    Args:
        problem: The complex problem to decompose
        max_depth: Maximum decomposition depth

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if problem is not None:
        params["problem"] = problem
    if max_depth is not None:
        params["max_depth"] = max_depth


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="deconstruct",
            arguments=params,
        )
    return asyncio.run(_async_call())


def hypothesize(node_a_id: str, node_b_id: str, context: str | None = None) -> Any:
    """Find novel connections between two concepts using topology tunneling - combines graph paths, vector similarity, and analogical pattern matching to discover 'Aha!' moments between distant concepts.

    Args:
        node_a_id: First thought-node ID
        node_b_id: Second thought-node ID
        context: Optional context to guide hypothesis generation

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if node_a_id is not None:
        params["node_a_id"] = node_a_id
    if node_b_id is not None:
        params["node_b_id"] = node_b_id
    if context is not None:
        params["context"] = context


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="hypothesize",
            arguments=params,
        )
    return asyncio.run(_async_call())


def synthesize(node_ids: list[str], goal: str | None = None) -> Any:
    """Merge multiple thought-nodes into a unified insight by operating in latent space. Computes centroids and finds common patterns.

    Args:
        node_ids: List of thought-node IDs to synthesize (minimum 2)
        goal: Optional goal to guide synthesis

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if node_ids is not None:
        params["node_ids"] = node_ids
    if goal is not None:
        params["goal"] = goal


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="synthesize",
            arguments=params,
        )
    return asyncio.run(_async_call())


def evolve_formula(data_points: list[dict[str, Any]], n_generations: int | None = None, hybrid: bool | None = None) -> Any:
    """Uses Genetic Programming (Symbolic Regression) to evolve a mathematical formula that fits a given dataset. Use this when the Director detects high entropy/complexity and simple patterns (like linear regression) fail. It discovers the 'hidden instruction set' of the data.

    Args:
        data_points: List of data points to fit
        n_generations: Number of evolutionary generations (default 10)
        hybrid: If true, enables Evolutionary Optimization (local refinement of constants). Slower but more precise.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if data_points is not None:
        params["data_points"] = data_points
    if n_generations is not None:
        params["n_generations"] = n_generations
    if hybrid is not None:
        params["hybrid"] = hybrid


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="evolve_formula",
            arguments=params,
        )
    return asyncio.run(_async_call())


def constrain(node_id: str, rules: list[str], mode: str | None = None, conclusion: str | None = None, strict: bool | None = None) -> Any:
    """Validate logical constraints using formal logic (Prover9/Mace4).

MODES (based on philosophical logic foundations):
- ENTAILMENT: "Does conclusion follow from premises?" - Proves derivability. Requires `conclusion` param.
- CONSISTENCY: "Can all statements be true together?" - Finds contradictions. Default mode.
- SATISFIABILITY: "Find a world where this holds" - Constructs a model.

For ENTAILMENT: rules are premises, conclusion is what to prove.
For CONSISTENCY/SATISFIABILITY: rules are the statements to check.

Syntax: Prover9 FOL format (e.g., "all x (human(x) -> mortal(x))", "human(socrates)").

    Args:
        node_id: Thought-node ID for context logging
        rules: FOL statements (Prover9 syntax). For entailment=premises, for consistency/satisfiability=statements to check
        mode: Validation mode: entailment (prove), consistency (no contradiction), satisfiability (find model)
        conclusion: For entailment mode: the statement to prove from rules as premises
        strict: Use Prover9/Mace4 (True) or embedding similarity (False)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if node_id is not None:
        params["node_id"] = node_id
    if rules is not None:
        params["rules"] = rules
    if mode is not None:
        params["mode"] = mode
    if conclusion is not None:
        params["conclusion"] = conclusion
    if strict is not None:
        params["strict"] = strict


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="constrain",
            arguments=params,
        )
    return asyncio.run(_async_call())


def resolve_meta_paradox(conflict: str, waitForPreviousTools: bool | None = None) -> Any:
    """Resolve an internal system conflict (Meta-Paradox) by treating it as a cognitive object. Deconstructs the conflict, hypothesizes a synthesis, and generates a resolution plan.

    Args:
        conflict: Description of the internal conflict (e.g., 'Validator says Yes but Critique says No')
        waitForPreviousTools: If true, wait for all previous tool calls from this turn to complete before executing (sequential). If false or omitted, execute this tool immediately (parallel with other tools).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if conflict is not None:
        params["conflict"] = conflict
    if waitForPreviousTools is not None:
        params["waitForPreviousTools"] = waitForPreviousTools


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="resolve_meta_paradox",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_active_goals() -> Any:
    """Get all currently active goals with their weights and metadata.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="get_active_goals",
            arguments=params,
        )
    return asyncio.run(_async_call())


def diagnose_pointer() -> Any:
    """Perform sheaf-theoretic diagnosis of the GoalController (Pointer). Checks for topological obstructions (H^1 > 0) or tension loops that might be causing the agent to get stuck.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="diagnose_pointer",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_cognitive_state() -> Any:
    """Get the agent's latest cognitive state (Proprioception). Returns the current 'shape' of thought (e.g., 'Focused', 'Looping') and its stability.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="check_cognitive_state",
            arguments=params,
        )
    return asyncio.run(_async_call())


def recall_work(query: str | None = None, operation_type: str | None = None, limit: int | None = None) -> Any:
    """Search the agent's past work history to recall previous operations, results, and cognitive states.

    Args:
        query: Text to search for in parameters or results
        operation_type: Filter by operation type (e.g., 'hypothesize')
        limit: Max number of results (default 100)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if operation_type is not None:
        params["operation_type"] = operation_type
    if limit is not None:
        params["limit"] = limit


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="recall_work",
            arguments=params,
        )
    return asyncio.run(_async_call())


def teach_cognitive_state(label: str) -> Any:
    """Teach the agent that its *current* thought pattern corresponds to a specific state label (Reinforcement Learning).

    Args:
        label: Name of the state (e.g., 'Creative', 'Stuck')

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if label is not None:
        params["label"] = label


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="teach_cognitive_state",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_known_archetypes() -> Any:
    """List all cognitive states the agent currently recognizes.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="get_known_archetypes",
            arguments=params,
        )
    return asyncio.run(_async_call())


def visualize_thought() -> Any:
    """Get an ASCII visualization of the last thought's topology.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="visualize_thought",
            arguments=params,
        )
    return asyncio.run(_async_call())


def run_sleep_cycle(epochs: int | None = None) -> Any:
    """Trigger a Sleep Cycle (Offline Learning) to consolidate recent memories and potentially crystallize new tools.

    Args:
        epochs: Number of training epochs (default 1)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if epochs is not None:
        params["epochs"] = epochs


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="run_sleep_cycle",
            arguments=params,
        )
    return asyncio.run(_async_call())


def diagnose_antifragility() -> Any:
    """Diagnose the system's antifragility by analyzing its topological and learning properties, and suggest adaptation strategies.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="diagnose_antifragility",
            arguments=params,
        )
    return asyncio.run(_async_call())


def orthogonal_dimensions_analyzer(concept_a: str, concept_b: str, context: str | None = None) -> Any:
    """Analyze the relationship between two concepts as orthogonal dimensions (Statistical Compression vs Causal Understanding).

    Args:
        concept_a: First concept (e.g., 'Deep Learning')
        concept_b: Second concept (e.g., 'Symbolic Logic')
        context: Optional context for the analysis

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if concept_a is not None:
        params["concept_a"] = concept_a
    if concept_b is not None:
        params["concept_b"] = concept_b
    if context is not None:
        params["context"] = context


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="orthogonal_dimensions_analyzer",
            arguments=params,
        )
    return asyncio.run(_async_call())


def revise(belief: str, evidence: str, constraints: list[str] | None = None) -> Any:
    """Refine a belief or concept using Hybrid Operator C (LTN + Hopfield). Adjusts a thought-node to better match evidence while respecting logical constraints and energy barriers.

    Args:
        belief: The current belief or thought content to revise
        evidence: New evidence or target concept to align with
        constraints: List of natural language constraints the revision must satisfy

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if belief is not None:
        params["belief"] = belief
    if evidence is not None:
        params["evidence"] = evidence
    if constraints is not None:
        params["constraints"] = constraints


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="revise",
            arguments=params,
        )
    return asyncio.run(_async_call())


def manage_advisor(action: str, params: dict[str, Any]) -> Any:
    """Consolidated tool for managing Advisors (CRUD + Knowledge) AND user's own cognitive state. Actions: set_goal, propose_goal, explore, set_mode, consult_curiosity, create, update, delete, list, get, link_knowledge, get_knowledge, get_context.

    Args:
        action: The management action to perform.
        params: Parameters for the action (e.g., {'goal_description': '...'} for set_goal).

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if action is not None:
        params["action"] = action
    if params is not None:
        params["params"] = params


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="manage_advisor",
            arguments=params,
        )
    return asyncio.run(_async_call())


def consult_advisor(advisor_id: str, query: str) -> Any:
    """Consult a specific Advisor as an autonomous agent. They can perform research, use tools, and save insights to The Library.

    Args:
        advisor_id: ID of the advisor to consult.
        query: The question or task for the advisor.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if advisor_id is not None:
        params["advisor_id"] = advisor_id
    if query is not None:
        params["query"] = query


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="consult_advisor",
            arguments=params,
        )
    return asyncio.run(_async_call())


def inspect_graph(mode: str, label: str | None = None, filters: dict[str, Any] | None = None, start_id: str | None = None, rel_type: str | None = None, direction: str | None = None, depth: int | None = None, limit: int | None = None) -> Any:
    """Inspect the graph using dynamic queries. Search for nodes, traverse relationships, or explore local context.

    Args:
        mode: Operation mode: 'nodes' (search), 'relationships' (traverse), 'context' (neighborhood).
        label: Node label to search for (required for mode='nodes').
        filters: Property filters for node search (e.g., {'name': 'Value'}).
        start_id: Starting node ID (required for 'relationships' and 'context').
        rel_type: Relationship type to traverse (required for mode='relationships').
        direction: Traversal direction.
        depth: Traversal depth (for 'context' mode).
        limit: Max results to return.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if mode is not None:
        params["mode"] = mode
    if label is not None:
        params["label"] = label
    if filters is not None:
        params["filters"] = filters
    if start_id is not None:
        params["start_id"] = start_id
    if rel_type is not None:
        params["rel_type"] = rel_type
    if direction is not None:
        params["direction"] = direction
    if depth is not None:
        params["depth"] = depth
    if limit is not None:
        params["limit"] = limit


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="inspect_graph",
            arguments=params,
        )
    return asyncio.run(_async_call())


def compute_grok_depth(speaker_id: str, listener_id: str, utterance_raw: str, speaker_intent: str | None = None, speaker_affect: dict[str, Any] | None = None, listener_affect: dict[str, Any] | None = None, context: str | None = None) -> Any:
    """Compute the Grok-Depth empathetic alignment score between two mind-states across Grok-Lang's six cognitive levels (Signal, Symbol, Syntax, Semantics, Pragmatics, Meta). Returns a total score (0-1) and per-level alignments with a diagnostic interpretation.

    Args:
        speaker_id: Identifier for the speaker/sender
        listener_id: Identifier for the listener/receiver
        utterance_raw: The raw utterance text (e.g., 'Fine.')
        speaker_intent: The speaker's intended speech act type
        speaker_affect: Speaker's affective state (VAD model)
        listener_affect: Listener's perceived affective state (VAD model)
        context: Optional context for the exchange

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if speaker_id is not None:
        params["speaker_id"] = speaker_id
    if listener_id is not None:
        params["listener_id"] = listener_id
    if utterance_raw is not None:
        params["utterance_raw"] = utterance_raw
    if speaker_intent is not None:
        params["speaker_intent"] = speaker_intent
    if speaker_affect is not None:
        params["speaker_affect"] = speaker_affect
    if listener_affect is not None:
        params["listener_affect"] = listener_affect
    if context is not None:
        params["context"] = context


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="compute_grok_depth",
            arguments=params,
        )
    return asyncio.run(_async_call())


def consult_computational_empathy(query_type: str, query_param: str | None = None) -> Any:
    """Query the Emotion Evolution Framework for evolutionary psychology insights, empathic response templates, and computational empathy architecture.

This tool provides access to:
- Basic emotions (fear, anger, disgust, joy, sadness, surprise) with neural correlates
- Complex emotions (guilt, pride, jealousy, romantic_love)
- Evolutionary layers of emotional processing (1-4)
- AI interaction guidelines and 7 key principles
- Empathic response templates (distress, joy, anxiety)
- Computational Empathy Architecture for value integration
- Valence-arousal to emotion mapping
- ACIP consciousness integration
- Emotional regulation strategies

    Args:
        query_type: Type of query to perform
        query_param: Parameter for the query (e.g., emotion name like 'fear', layer number like '2', context like 'distress', or 'valence,arousal' like '-0.5,0.8')

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query_type is not None:
        params["query_type"] = query_type
    if query_param is not None:
        params["query_param"] = query_param


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="consult_computational_empathy",
            arguments=params,
        )
    return asyncio.run(_async_call())


def prove(premises: list[str], conclusion: str) -> Any:
    """Prove a logical statement using Prover9.

Syntax: Prover9 FOL format (e.g., "all x (human(x) -> mortal(x))", "human(socrates)").
- Universal: all x (P(x)) — NOT ∀x
- Existential: exists x (P(x)) — wrap formula in parentheses!
- Implication: -> — NOT ⇒
- Negation: -P(x) — NOT ¬ or !
- Conjunction: & — NOT ∧
- Disjunction: | — NOT ∨
- Predicates: lowercase preferred (human, mortal)

    Args:
        premises: List of logical premises
        conclusion: Statement to prove

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if conclusion is not None:
        params["conclusion"] = conclusion


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="prove",
            arguments=params,
        )
    return asyncio.run(_async_call())


def find_counterexample(premises: list[str], conclusion: str, domain_size: int | None = None) -> Any:
    """Use Mace4 to find a counterexample showing the conclusion doesn't follow from premises.

Syntax: Same as prove tool - use Prover9 FOL format.

    Args:
        premises: List of logical premises
        conclusion: Conclusion to disprove
        domain_size: Optional: specific domain size to search

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if conclusion is not None:
        params["conclusion"] = conclusion
    if domain_size is not None:
        params["domain_size"] = domain_size


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="find_counterexample",
            arguments=params,
        )
    return asyncio.run(_async_call())


def find_model(premises: list[str], domain_size: int | None = None) -> Any:
    """Use Mace4 to find a finite model satisfying the given premises.

Syntax: Same as prove tool - use Prover9 FOL format.

    Args:
        premises: List of logical premises
        domain_size: Optional: specific domain size to search (default: incrementally search 2-10)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if premises is not None:
        params["premises"] = premises
    if domain_size is not None:
        params["domain_size"] = domain_size


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="find_model",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_well_formed(statements: list[str]) -> Any:
    """Check if logical statements are well-formed with detailed syntax validation.

    Args:
        statements: Logical statements to check

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if statements is not None:
        params["statements"] = statements


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="check_well_formed",
            arguments=params,
        )
    return asyncio.run(_async_call())


def verify_commutativity(path_a: list[str], path_b: list[str], object_start: str, object_end: str, with_category_axioms: bool | None = None) -> Any:
    """Verify that a categorical diagram commutes by generating FOL premises and conclusion.

    Args:
        path_a: List of morphism names in first path
        path_b: List of morphism names in second path
        object_start: Starting object
        object_end: Ending object
        with_category_axioms: Include basic category theory axioms (default: true)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if path_a is not None:
        params["path_a"] = path_a
    if path_b is not None:
        params["path_b"] = path_b
    if object_start is not None:
        params["object_start"] = object_start
    if object_end is not None:
        params["object_end"] = object_end
    if with_category_axioms is not None:
        params["with_category_axioms"] = with_category_axioms


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="verify_commutativity",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_category_axioms(concept: str, functor_name: str | None = None, functor_f: str | None = None, functor_g: str | None = None, component: str | None = None) -> Any:
    """Get FOL axioms for category theory concepts (category, functor, natural transformation).

    Args:
        concept: Which concept's axioms to retrieve
        functor_name: For functor axioms: name of the functor (default: F)
        functor_f: For natural transformation: first functor
        functor_g: For natural transformation: second functor
        component: For natural transformation: component name

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if concept is not None:
        params["concept"] = concept
    if functor_name is not None:
        params["functor_name"] = functor_name
    if functor_f is not None:
        params["functor_f"] = functor_f
    if functor_g is not None:
        params["functor_g"] = functor_g
    if component is not None:
        params["component"] = component


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="get_category_axioms",
            arguments=params,
        )
    return asyncio.run(_async_call())


def consult_ruminator(focus_node_id: str | None = None, mode: str | None = None) -> Any:
    """Consult the Category-Theoretic Ruminator to perform 'Diagram Chasing' on the knowledge graph.

        It identifies 'open triangles' (non-commutative diagrams) starting from a focus node and uses an LLM (acting as a Functor) to propose missing relationships (morphisms) to make the diagram commute.

    Args:
        focus_node_id: Optional: The ID of the node to focus rumination on. If omitted, the system selects a node with 'structural tension'.
        mode: Operational mode (currently only 'diagram_chasing')

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if focus_node_id is not None:
        params["focus_node_id"] = focus_node_id
    if mode is not None:
        params["mode"] = mode


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="reflective-agent-architecture",
            tool_name="consult_ruminator",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['deconstruct', 'hypothesize', 'synthesize', 'evolve_formula', 'constrain', 'resolve_meta_paradox', 'get_active_goals', 'diagnose_pointer', 'check_cognitive_state', 'recall_work', 'teach_cognitive_state', 'get_known_archetypes', 'visualize_thought', 'run_sleep_cycle', 'diagnose_antifragility', 'orthogonal_dimensions_analyzer', 'revise', 'manage_advisor', 'consult_advisor', 'inspect_graph', 'compute_grok_depth', 'consult_computational_empathy', 'prove', 'find_counterexample', 'find_model', 'check_well_formed', 'verify_commutativity', 'get_category_axioms', 'consult_ruminator']
