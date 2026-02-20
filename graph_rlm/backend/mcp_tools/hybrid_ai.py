"""
Auto-generated wrapper for hybrid-ai MCP server.

This module provides Python function wrappers for all tools
exposed by the hybrid-ai server.

Do not edit manually.
"""

from typing import Any


def create_mcp_neuron(weights: list[float] | Any = None, threshold: float | None = None, name: Any | None = None, **kwargs) -> Any:
    """Create a McCulloch-Pitts neuron for transparent decision-making.

Args:
    weights: Weights for the neuron [bias, w1, w2, ..., wn]
    threshold: Activation threshold (default: 0.0)
    name: Optional name for the neuron (for use in networks)

Returns:
    Dictionary with neuron configuration and usage instructions

Example:
    # Create an AND gate neuron
    create_mcp_neuron([-1.0, 0.6, 0.6], name="safety_check")

    Args:
        weights: 
        threshold: 
        name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if weights is not None:
        mcp_args["weights"] = weights
    if threshold is not None:
        mcp_args["threshold"] = threshold
    if name is not None:
        mcp_args["name"] = name

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="create_mcp_neuron",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def evaluate_neuron(inputs: list[float] | Any = None, weights: list[float] | Any = None, threshold: float | None = None, **kwargs) -> Any:
    """Evaluate an MCP neuron with given inputs and return explainable result.

Args:
    inputs: Input values [x1, x2, ..., xn]
    weights: Neuron weights [bias, w1, w2, ..., wn]
    threshold: Activation threshold

Returns:
    Detailed evaluation with explanation

Example:
    # Check if both safety conditions are met
    evaluate_neuron([1, 1], [-1.0, 0.6, 0.6])
    # Returns: {"output": 1, "fired": true, ...}

    Args:
        inputs: 
        weights: 
        threshold: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if inputs is not None:
        mcp_args["inputs"] = inputs
    if weights is not None:
        mcp_args["weights"] = weights
    if threshold is not None:
        mcp_args["threshold"] = threshold

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="evaluate_neuron",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def logic_gate(gate_type: str | Any = None, inputs: list[float] | Any = None, **kwargs) -> Any:
    """Apply a standard logic gate using MCP neurons.

Args:
    gate_type: Type of gate (AND, OR, NOT, NAND, NOR, XOR)
    inputs: Input values (1-2 inputs depending on gate)

Returns:
    Gate output with explanation

Example:
    # Safety check: require both conditions
    logic_gate("AND", [1, 1])  # Returns 1 (safe)
    logic_gate("AND", [1, 0])  # Returns 0 (not safe)

    Args:
        gate_type: 
        inputs: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if gate_type is not None:
        mcp_args["gate_type"] = gate_type
    if inputs is not None:
        mcp_args["inputs"] = inputs

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="logic_gate",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def create_decision_rule(rule_name: str | Any = None, weights: list[float] | Any = None, threshold: float | None = None, description: str | None = None, **kwargs) -> Any:
    """Create a named decision rule in the network for reuse.

This is useful for encoding business rules, safety constraints,
or other explicit decision logic that should be transparent and auditable.

Args:
    rule_name: Unique name for this rule
    weights: Neuron weights [bias, w1, w2, ..., wn]
    threshold: Activation threshold
    description: Human-readable description of what this rule checks

Returns:
    Rule creation status

Example:
    # Safety rule: both conditions must be true
    create_decision_rule(
        "safety_check",
        [-1.0, 0.6, 0.6],
        description="Requires both safety_sensor_1 and safety_sensor_2"
    )

    Args:
        rule_name: 
        weights: 
        threshold: 
        description: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if rule_name is not None:
        mcp_args["rule_name"] = rule_name
    if weights is not None:
        mcp_args["weights"] = weights
    if threshold is not None:
        mcp_args["threshold"] = threshold
    if description is not None:
        mcp_args["description"] = description

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="create_decision_rule",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def apply_decision_rule(rule_name: str | Any = None, inputs: list[float] | Any = None, **kwargs) -> Any:
    """Apply a named decision rule with full explanation.

This provides complete transparency for why a decision was made,
making it suitable for regulated industries, safety-critical systems,
or any application requiring explainability.

Args:
    rule_name: Name of the rule to apply
    inputs: Input values to evaluate

Returns:
    Decision output with complete explanation trace

Example:
    apply_decision_rule("safety_check", [1, 1])
    # Returns detailed explanation of why safety check passed

    Args:
        rule_name: 
        inputs: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if rule_name is not None:
        mcp_args["rule_name"] = rule_name
    if inputs is not None:
        mcp_args["inputs"] = inputs

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="apply_decision_rule",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def get_decision_log(**kwargs) -> Any:
    """Get the complete decision log for auditability.

Returns all decisions made by the network with their explanations,
perfect for compliance, debugging, or understanding system behavior.

Returns:
    Complete log of all decisions with explanations

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="get_decision_log",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def clear_decision_log(**kwargs) -> Any:
    """Clear the decision log.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="clear_decision_log",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))


def post_nn_decision(nn_outputs: dict[str, Any] | Any = None, decision_rule: str | Any = None, threshold: float | None = None, **kwargs) -> Any:
    """Hybrid AI workflow: Use NN outputs with transparent decision rules.

This implements the "post-NN calculator" pattern where:
1. A neural network provides perception/analysis (nn_outputs)
2. Simple, transparent MCP neurons make the final decision
3. Every decision is fully explainable and auditable

Args:
    ctx: FastMCP context
    nn_outputs: Dictionary of NN outputs (e.g., {"is_safe": 0.95, "is_valid": 0.88})
    decision_rule: Name of the rule to apply or logic expression
    threshold: Threshold to convert NN probabilities to binary (default: 0.5)

Returns:
    Final decision with complete explanation chain

Example:
    # NN says object detection confidence is high
    post_nn_decision(
        {"pedestrian_detected": 0.95, "collision_imminent": 0.92},
        "emergency_brake",
        threshold=0.9
    )

    Args:
        nn_outputs: 
        decision_rule: 
        threshold: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if nn_outputs is not None:
        mcp_args["nn_outputs"] = nn_outputs
    if decision_rule is not None:
        mcp_args["decision_rule"] = decision_rule
    if threshold is not None:
        mcp_args["threshold"] = threshold

    async def _async_call():
        return await call_mcp_tool(
            server_name="hybrid-ai",
            tool_name="post_nn_decision",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return a wrapper that normalizes the result
            async def _normalized_async_call():
                return normalize_mcp_result(await _async_call())
            return _normalized_async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return normalize_mcp_result(asyncio.run(_async_call()))



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['create_mcp_neuron', 'evaluate_neuron', 'logic_gate', 'create_decision_rule', 'apply_decision_rule', 'get_decision_log', 'clear_decision_log', 'post_nn_decision']
