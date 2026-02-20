"""
Auto-generated wrapper for fermat-mcp MCP server.

This module provides Python function wrappers for all tools
exposed by the fermat-mcp server.

Do not edit manually.
"""

from typing import Any


def mpl_mcp_plot_barchart(values: list[float] | Any = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: str | None = None, save: bool | None = None, dpi: int | None = None, orientation: str | None = None, **kwargs) -> Any:
    """Plots barchart of given datavalues

    Args:
        values: 
        labels: 
        title: 
        xlabel: 
        ylabel: 
        color: 
        save: 
        dpi: 
        orientation: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if values is not None:
        mcp_args["values"] = values
    if labels is not None:
        mcp_args["labels"] = labels
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if color is not None:
        mcp_args["color"] = color
    if save is not None:
        mcp_args["save"] = save
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if orientation is not None:
        mcp_args["orientation"] = orientation

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_barchart",
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


def mpl_mcp_plot_scatter(x_data: list[float] | Any = None, y_data: list[float] | Any = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: Any | None = None, size: Any | None = None, alpha: float | None = None, marker: str | None = None, edgecolors: Any | None = None, linewidths: float | None = None, save: bool | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None, **kwargs) -> Any:
    """Plots scatter chart of given datavalues

    Args:
        x_data: 
        y_data: 
        labels: 
        title: 
        xlabel: 
        ylabel: 
        color: 
        size: 
        alpha: 
        marker: 
        edgecolors: 
        linewidths: 
        save: 
        dpi: 
        figsize: 
        grid: 
        legend: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if x_data is not None:
        mcp_args["x_data"] = x_data
    if y_data is not None:
        mcp_args["y_data"] = y_data
    if labels is not None:
        mcp_args["labels"] = labels
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if color is not None:
        mcp_args["color"] = color
    if size is not None:
        mcp_args["size"] = size
    if alpha is not None:
        mcp_args["alpha"] = alpha
    if marker is not None:
        mcp_args["marker"] = marker
    if edgecolors is not None:
        mcp_args["edgecolors"] = edgecolors
    if linewidths is not None:
        mcp_args["linewidths"] = linewidths
    if save is not None:
        mcp_args["save"] = save
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if figsize is not None:
        mcp_args["figsize"] = figsize
    if grid is not None:
        mcp_args["grid"] = grid
    if legend is not None:
        mcp_args["legend"] = legend

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_scatter",
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


def mpl_mcp_plot_chart(x_data: list[float] | Any = None, y_data: Any | Any = None, plot_type: str | None = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: Any | None = None, save: bool | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None, **kwargs) -> Any:
    """Plots line/scatter/bar chart of given datavalues

    Args:
        x_data: 
        y_data: 
        plot_type: 
        labels: 
        title: 
        xlabel: 
        ylabel: 
        color: 
        save: 
        dpi: 
        figsize: 
        grid: 
        legend: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if x_data is not None:
        mcp_args["x_data"] = x_data
    if y_data is not None:
        mcp_args["y_data"] = y_data
    if plot_type is not None:
        mcp_args["plot_type"] = plot_type
    if labels is not None:
        mcp_args["labels"] = labels
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if color is not None:
        mcp_args["color"] = color
    if save is not None:
        mcp_args["save"] = save
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if figsize is not None:
        mcp_args["figsize"] = figsize
    if grid is not None:
        mcp_args["grid"] = grid
    if legend is not None:
        mcp_args["legend"] = legend

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_chart",
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


def mpl_mcp_plot_stem(x_data: Any | Any = None, y_data: Any | Any = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, colors: Any | None = None, linefmt: str | None = None, markerfmt: str | None = None, basefmt: str | None = None, bottom: float | None = None, orientation: str | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None, **kwargs) -> Any:
    """Plots stem chart of given datavalues

    Args:
        x_data: 
        y_data: 
        labels: 
        title: 
        xlabel: 
        ylabel: 
        colors: 
        linefmt: 
        markerfmt: 
        basefmt: 
        bottom: 
        orientation: 
        dpi: 
        figsize: 
        grid: 
        legend: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if x_data is not None:
        mcp_args["x_data"] = x_data
    if y_data is not None:
        mcp_args["y_data"] = y_data
    if labels is not None:
        mcp_args["labels"] = labels
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if colors is not None:
        mcp_args["colors"] = colors
    if linefmt is not None:
        mcp_args["linefmt"] = linefmt
    if markerfmt is not None:
        mcp_args["markerfmt"] = markerfmt
    if basefmt is not None:
        mcp_args["basefmt"] = basefmt
    if bottom is not None:
        mcp_args["bottom"] = bottom
    if orientation is not None:
        mcp_args["orientation"] = orientation
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if figsize is not None:
        mcp_args["figsize"] = figsize
    if grid is not None:
        mcp_args["grid"] = grid
    if legend is not None:
        mcp_args["legend"] = legend

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_stem",
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


def mpl_mcp_plot_stack(x_data: Any | Any = None, y_data: Any | Any = None, chart_type: str | None = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, colors: Any | None = None, alpha: float | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None, **kwargs) -> Any:
    """Plots stacked area/bar chart of given datavalues

    Args:
        x_data: 
        y_data: 
        chart_type: 
        labels: 
        title: 
        xlabel: 
        ylabel: 
        colors: 
        alpha: 
        dpi: 
        figsize: 
        grid: 
        legend: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if x_data is not None:
        mcp_args["x_data"] = x_data
    if y_data is not None:
        mcp_args["y_data"] = y_data
    if chart_type is not None:
        mcp_args["chart_type"] = chart_type
    if labels is not None:
        mcp_args["labels"] = labels
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if colors is not None:
        mcp_args["colors"] = colors
    if alpha is not None:
        mcp_args["alpha"] = alpha
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if figsize is not None:
        mcp_args["figsize"] = figsize
    if grid is not None:
        mcp_args["grid"] = grid
    if legend is not None:
        mcp_args["legend"] = legend

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_stack",
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


def mpl_mcp_eqn_chart(equations: Any | Any = None, x_min: float | None = None, x_max: float | None = None, num_points: int | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, grid: bool | None = None, legend: bool | None = None, figsize: list[int] | None = None, linewidth: float | None = None, linestyle: str | None = None, alpha: float | None = None, dpi: int | None = None, save: bool | None = None, **kwargs) -> Any:
    """Plots mathematical equations

    Args:
        equations: 
        x_min: 
        x_max: 
        num_points: 
        title: 
        xlabel: 
        ylabel: 
        grid: 
        legend: 
        figsize: 
        linewidth: 
        linestyle: 
        alpha: 
        dpi: 
        save: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if equations is not None:
        mcp_args["equations"] = equations
    if x_min is not None:
        mcp_args["x_min"] = x_min
    if x_max is not None:
        mcp_args["x_max"] = x_max
    if num_points is not None:
        mcp_args["num_points"] = num_points
    if title is not None:
        mcp_args["title"] = title
    if xlabel is not None:
        mcp_args["xlabel"] = xlabel
    if ylabel is not None:
        mcp_args["ylabel"] = ylabel
    if grid is not None:
        mcp_args["grid"] = grid
    if legend is not None:
        mcp_args["legend"] = legend
    if figsize is not None:
        mcp_args["figsize"] = figsize
    if linewidth is not None:
        mcp_args["linewidth"] = linewidth
    if linestyle is not None:
        mcp_args["linestyle"] = linestyle
    if alpha is not None:
        mcp_args["alpha"] = alpha
    if dpi is not None:
        mcp_args["dpi"] = dpi
    if save is not None:
        mcp_args["save"] = save

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_eqn_chart",
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


def numpy_mcp_numerical_operation(operation: str | Any = None, a: Any | None = None, b: Any | None = None, shape: Any | None = None, new_shape: Any | None = None, axis: int | None = None, q: Any | None = None, start: Any | None = None, stop: Any | None = None, step: Any | None = None, num: Any | None = None, fill_value: Any | None = None, **kwargs) -> Any:
    """Do numerical operation like add, sub, mul, div, power, abs, exp, log, sqrt, sin, cos, tan, mean, median, std, var, min, max, argmin, argmax, percentile, dot, matmul, inv, det, eig, solve, svd

    Args:
        operation: 
        a: 
        b: 
        shape: 
        new_shape: 
        axis: 
        q: 
        start: 
        stop: 
        step: 
        num: 
        fill_value: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if a is not None:
        mcp_args["a"] = a
    if b is not None:
        mcp_args["b"] = b
    if shape is not None:
        mcp_args["shape"] = shape
    if new_shape is not None:
        mcp_args["new_shape"] = new_shape
    if axis is not None:
        mcp_args["axis"] = axis
    if q is not None:
        mcp_args["q"] = q
    if start is not None:
        mcp_args["start"] = start
    if stop is not None:
        mcp_args["stop"] = stop
    if step is not None:
        mcp_args["step"] = step
    if num is not None:
        mcp_args["num"] = num
    if fill_value is not None:
        mcp_args["fill_value"] = fill_value

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="numpy_mcp_numerical_operation",
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


def numpy_mcp_matlib_operation(operation: str | Any = None, data: Any | None = None, shape: Any | None = None, m: Any | None = None, n: Any | None = None, k: int | None = None, start: Any | None = None, stop: Any | None = None, step: Any | None = None, num: Any | None = None, axis: int | None = None, **kwargs) -> Any:
    """Do matrix operations: rand-mat, zeros, ones, eye, identity, arange, linspace, reshape, flatten, concatenate, transpose, stack

    Args:
        operation: 
        data: 
        shape: 
        m: 
        n: 
        k: 
        start: 
        stop: 
        step: 
        num: 
        axis: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if data is not None:
        mcp_args["data"] = data
    if shape is not None:
        mcp_args["shape"] = shape
    if m is not None:
        mcp_args["m"] = m
    if n is not None:
        mcp_args["n"] = n
    if k is not None:
        mcp_args["k"] = k
    if start is not None:
        mcp_args["start"] = start
    if stop is not None:
        mcp_args["stop"] = stop
    if step is not None:
        mcp_args["step"] = step
    if num is not None:
        mcp_args["num"] = num
    if axis is not None:
        mcp_args["axis"] = axis

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="numpy_mcp_matlib_operation",
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


def sympy_mcp_algebra_operation(operation: str | Any = None, expr: str | Any = None, syms: Any | None = None, rational: bool | None = None, ratio: float | None = None, measure: Any | None = None, deep: bool | None = None, modulus: Any | None = None, power_base: bool | None = None, power_exp: bool | None = None, mul: bool | None = None, log: bool | None = None, multinomial: bool | None = None, basic: bool | None = None, frac: bool | None = None, sign: bool | None = None, evaluate: bool | None = None, exact: bool | None = None, **kwargs) -> Any:
    """Do algebraic operations like simplify, expand, factor, collect

    Args:
        operation: 
        expr: 
        syms: 
        rational: 
        ratio: 
        measure: 
        deep: 
        modulus: 
        power_base: 
        power_exp: 
        mul: 
        log: 
        multinomial: 
        basic: 
        frac: 
        sign: 
        evaluate: 
        exact: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if expr is not None:
        mcp_args["expr"] = expr
    if syms is not None:
        mcp_args["syms"] = syms
    if rational is not None:
        mcp_args["rational"] = rational
    if ratio is not None:
        mcp_args["ratio"] = ratio
    if measure is not None:
        mcp_args["measure"] = measure
    if deep is not None:
        mcp_args["deep"] = deep
    if modulus is not None:
        mcp_args["modulus"] = modulus
    if power_base is not None:
        mcp_args["power_base"] = power_base
    if power_exp is not None:
        mcp_args["power_exp"] = power_exp
    if mul is not None:
        mcp_args["mul"] = mul
    if log is not None:
        mcp_args["log"] = log
    if multinomial is not None:
        mcp_args["multinomial"] = multinomial
    if basic is not None:
        mcp_args["basic"] = basic
    if frac is not None:
        mcp_args["frac"] = frac
    if sign is not None:
        mcp_args["sign"] = sign
    if evaluate is not None:
        mcp_args["evaluate"] = evaluate
    if exact is not None:
        mcp_args["exact"] = exact

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_algebra_operation",
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


def sympy_mcp_calculus_operation(operation: str | Any = None, expr: str | Any = None, sym: Any | None = None, n: int | None = None, lower: Any | None = None, upper: Any | None = None, point: Any | None = None, direction: str | None = None, series_n: int | None = None, **kwargs) -> Any:
    """Do calculus operations like diff, integrate, limit, series

    Args:
        operation: 
        expr: 
        sym: 
        n: 
        lower: 
        upper: 
        point: 
        direction: 
        series_n: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if expr is not None:
        mcp_args["expr"] = expr
    if sym is not None:
        mcp_args["sym"] = sym
    if n is not None:
        mcp_args["n"] = n
    if lower is not None:
        mcp_args["lower"] = lower
    if upper is not None:
        mcp_args["upper"] = upper
    if point is not None:
        mcp_args["point"] = point
    if direction is not None:
        mcp_args["direction"] = direction
    if series_n is not None:
        mcp_args["series_n"] = series_n

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_calculus_operation",
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


def sympy_mcp_equation_operation(operation: str | Any = None, equations: Any | Any = None, symbols: Any | None = None, domain: Any | None = None, check: bool | None = None, simplify: bool | None = None, rational: Any | None = None, minimal: bool | None = None, force: bool | None = None, implicit: bool | None = None, **kwargs) -> Any:
    """Do symbolic equation operations like solve, solveset, linsolve, nonlinsolve

    Args:
        operation: 
        equations: 
        symbols: 
        domain: 
        check: 
        simplify: 
        rational: 
        minimal: 
        force: 
        implicit: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if equations is not None:
        mcp_args["equations"] = equations
    if symbols is not None:
        mcp_args["symbols"] = symbols
    if domain is not None:
        mcp_args["domain"] = domain
    if check is not None:
        mcp_args["check"] = check
    if simplify is not None:
        mcp_args["simplify"] = simplify
    if rational is not None:
        mcp_args["rational"] = rational
    if minimal is not None:
        mcp_args["minimal"] = minimal
    if force is not None:
        mcp_args["force"] = force
    if implicit is not None:
        mcp_args["implicit"] = implicit

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_equation_operation",
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


def sympy_mcp_matrix_operation(operation: str | Any = None, data: Any | Any = None, rational: bool | None = None, nrows: Any | None = None, ncols: Any | None = None, simplify: bool | None = None, **kwargs) -> Any:
    """Do symbolic matrix operations like create, det, inv, rref, eigenvals

    Args:
        operation: 
        data: 
        rational: 
        nrows: 
        ncols: 
        simplify: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args: dict[str, Any] = {}
    # Mapping for snake_case to CamelCase resilience
    param_map = {}
    if operation is not None:
        mcp_args["operation"] = operation
    if data is not None:
        mcp_args["data"] = data
    if rational is not None:
        mcp_args["rational"] = rational
    if nrows is not None:
        mcp_args["nrows"] = nrows
    if ncols is not None:
        mcp_args["ncols"] = ncols
    if simplify is not None:
        mcp_args["simplify"] = simplify

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_matrix_operation",
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
    return ['mpl_mcp_plot_barchart', 'mpl_mcp_plot_scatter', 'mpl_mcp_plot_chart', 'mpl_mcp_plot_stem', 'mpl_mcp_plot_stack', 'mpl_mcp_eqn_chart', 'numpy_mcp_numerical_operation', 'numpy_mcp_matlib_operation', 'sympy_mcp_algebra_operation', 'sympy_mcp_calculus_operation', 'sympy_mcp_equation_operation', 'sympy_mcp_matrix_operation']
