"""
Auto-generated wrapper for fermat-mcp MCP server.

This module provides Python function wrappers for all tools
exposed by the fermat-mcp server.

Do not edit manually.
"""

from typing import Any


def mpl_mcp_plot_barchart(values: list[float], labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: str | None = None, save: bool | None = None, dpi: int | None = None, orientation: str | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if values is not None:
        params["values"] = values
    if labels is not None:
        params["labels"] = labels
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if color is not None:
        params["color"] = color
    if save is not None:
        params["save"] = save
    if dpi is not None:
        params["dpi"] = dpi
    if orientation is not None:
        params["orientation"] = orientation


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_barchart",
            arguments=params,
        )
    return asyncio.run(_async_call())


def mpl_mcp_plot_scatter(x_data: list[float], y_data: list[float], labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: Any | None = None, size: Any | None = None, alpha: float | None = None, marker: str | None = None, edgecolors: Any | None = None, linewidths: float | None = None, save: bool | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if x_data is not None:
        params["x_data"] = x_data
    if y_data is not None:
        params["y_data"] = y_data
    if labels is not None:
        params["labels"] = labels
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if color is not None:
        params["color"] = color
    if size is not None:
        params["size"] = size
    if alpha is not None:
        params["alpha"] = alpha
    if marker is not None:
        params["marker"] = marker
    if edgecolors is not None:
        params["edgecolors"] = edgecolors
    if linewidths is not None:
        params["linewidths"] = linewidths
    if save is not None:
        params["save"] = save
    if dpi is not None:
        params["dpi"] = dpi
    if figsize is not None:
        params["figsize"] = figsize
    if grid is not None:
        params["grid"] = grid
    if legend is not None:
        params["legend"] = legend


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_scatter",
            arguments=params,
        )
    return asyncio.run(_async_call())


def mpl_mcp_plot_chart(x_data: list[float], y_data: Any, plot_type: str | None = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, color: Any | None = None, save: bool | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if x_data is not None:
        params["x_data"] = x_data
    if y_data is not None:
        params["y_data"] = y_data
    if plot_type is not None:
        params["plot_type"] = plot_type
    if labels is not None:
        params["labels"] = labels
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if color is not None:
        params["color"] = color
    if save is not None:
        params["save"] = save
    if dpi is not None:
        params["dpi"] = dpi
    if figsize is not None:
        params["figsize"] = figsize
    if grid is not None:
        params["grid"] = grid
    if legend is not None:
        params["legend"] = legend


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_chart",
            arguments=params,
        )
    return asyncio.run(_async_call())


def mpl_mcp_plot_stem(x_data: Any, y_data: Any, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, colors: Any | None = None, linefmt: str | None = None, markerfmt: str | None = None, basefmt: str | None = None, bottom: float | None = None, orientation: str | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if x_data is not None:
        params["x_data"] = x_data
    if y_data is not None:
        params["y_data"] = y_data
    if labels is not None:
        params["labels"] = labels
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if colors is not None:
        params["colors"] = colors
    if linefmt is not None:
        params["linefmt"] = linefmt
    if markerfmt is not None:
        params["markerfmt"] = markerfmt
    if basefmt is not None:
        params["basefmt"] = basefmt
    if bottom is not None:
        params["bottom"] = bottom
    if orientation is not None:
        params["orientation"] = orientation
    if dpi is not None:
        params["dpi"] = dpi
    if figsize is not None:
        params["figsize"] = figsize
    if grid is not None:
        params["grid"] = grid
    if legend is not None:
        params["legend"] = legend


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_stem",
            arguments=params,
        )
    return asyncio.run(_async_call())


def mpl_mcp_plot_stack(x_data: Any, y_data: Any, chart_type: str | None = None, labels: Any | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, colors: Any | None = None, alpha: float | None = None, dpi: int | None = None, figsize: Any | None = None, grid: bool | None = None, legend: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if x_data is not None:
        params["x_data"] = x_data
    if y_data is not None:
        params["y_data"] = y_data
    if chart_type is not None:
        params["chart_type"] = chart_type
    if labels is not None:
        params["labels"] = labels
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if colors is not None:
        params["colors"] = colors
    if alpha is not None:
        params["alpha"] = alpha
    if dpi is not None:
        params["dpi"] = dpi
    if figsize is not None:
        params["figsize"] = figsize
    if grid is not None:
        params["grid"] = grid
    if legend is not None:
        params["legend"] = legend


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_plot_stack",
            arguments=params,
        )
    return asyncio.run(_async_call())


def mpl_mcp_eqn_chart(equations: Any, x_min: float | None = None, x_max: float | None = None, num_points: int | None = None, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, grid: bool | None = None, legend: bool | None = None, figsize: list[int] | None = None, linewidth: float | None = None, linestyle: str | None = None, alpha: float | None = None, dpi: int | None = None, save: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if equations is not None:
        params["equations"] = equations
    if x_min is not None:
        params["x_min"] = x_min
    if x_max is not None:
        params["x_max"] = x_max
    if num_points is not None:
        params["num_points"] = num_points
    if title is not None:
        params["title"] = title
    if xlabel is not None:
        params["xlabel"] = xlabel
    if ylabel is not None:
        params["ylabel"] = ylabel
    if grid is not None:
        params["grid"] = grid
    if legend is not None:
        params["legend"] = legend
    if figsize is not None:
        params["figsize"] = figsize
    if linewidth is not None:
        params["linewidth"] = linewidth
    if linestyle is not None:
        params["linestyle"] = linestyle
    if alpha is not None:
        params["alpha"] = alpha
    if dpi is not None:
        params["dpi"] = dpi
    if save is not None:
        params["save"] = save


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="mpl_mcp_eqn_chart",
            arguments=params,
        )
    return asyncio.run(_async_call())


def numpy_mcp_numerical_operation(operation: str, a: Any | None = None, b: Any | None = None, shape: Any | None = None, new_shape: Any | None = None, axis: int | None = None, q: Any | None = None, start: Any | None = None, stop: Any | None = None, step: Any | None = None, num: Any | None = None, fill_value: Any | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if a is not None:
        params["a"] = a
    if b is not None:
        params["b"] = b
    if shape is not None:
        params["shape"] = shape
    if new_shape is not None:
        params["new_shape"] = new_shape
    if axis is not None:
        params["axis"] = axis
    if q is not None:
        params["q"] = q
    if start is not None:
        params["start"] = start
    if stop is not None:
        params["stop"] = stop
    if step is not None:
        params["step"] = step
    if num is not None:
        params["num"] = num
    if fill_value is not None:
        params["fill_value"] = fill_value


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="numpy_mcp_numerical_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def numpy_mcp_matlib_operation(operation: str, data: Any | None = None, shape: Any | None = None, m: Any | None = None, n: Any | None = None, k: int | None = None, start: Any | None = None, stop: Any | None = None, step: Any | None = None, num: Any | None = None, axis: int | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if data is not None:
        params["data"] = data
    if shape is not None:
        params["shape"] = shape
    if m is not None:
        params["m"] = m
    if n is not None:
        params["n"] = n
    if k is not None:
        params["k"] = k
    if start is not None:
        params["start"] = start
    if stop is not None:
        params["stop"] = stop
    if step is not None:
        params["step"] = step
    if num is not None:
        params["num"] = num
    if axis is not None:
        params["axis"] = axis


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="numpy_mcp_matlib_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def sympy_mcp_algebra_operation(operation: str, expr: str, syms: Any | None = None, rational: bool | None = None, ratio: float | None = None, measure: Any | None = None, deep: bool | None = None, modulus: Any | None = None, power_base: bool | None = None, power_exp: bool | None = None, mul: bool | None = None, log: bool | None = None, multinomial: bool | None = None, basic: bool | None = None, frac: bool | None = None, sign: bool | None = None, evaluate: bool | None = None, exact: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if expr is not None:
        params["expr"] = expr
    if syms is not None:
        params["syms"] = syms
    if rational is not None:
        params["rational"] = rational
    if ratio is not None:
        params["ratio"] = ratio
    if measure is not None:
        params["measure"] = measure
    if deep is not None:
        params["deep"] = deep
    if modulus is not None:
        params["modulus"] = modulus
    if power_base is not None:
        params["power_base"] = power_base
    if power_exp is not None:
        params["power_exp"] = power_exp
    if mul is not None:
        params["mul"] = mul
    if log is not None:
        params["log"] = log
    if multinomial is not None:
        params["multinomial"] = multinomial
    if basic is not None:
        params["basic"] = basic
    if frac is not None:
        params["frac"] = frac
    if sign is not None:
        params["sign"] = sign
    if evaluate is not None:
        params["evaluate"] = evaluate
    if exact is not None:
        params["exact"] = exact


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_algebra_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def sympy_mcp_calculus_operation(operation: str, expr: str, sym: Any | None = None, n: int | None = None, lower: Any | None = None, upper: Any | None = None, point: Any | None = None, direction: str | None = None, series_n: int | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if expr is not None:
        params["expr"] = expr
    if sym is not None:
        params["sym"] = sym
    if n is not None:
        params["n"] = n
    if lower is not None:
        params["lower"] = lower
    if upper is not None:
        params["upper"] = upper
    if point is not None:
        params["point"] = point
    if direction is not None:
        params["direction"] = direction
    if series_n is not None:
        params["series_n"] = series_n


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_calculus_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def sympy_mcp_equation_operation(operation: str, equations: Any, symbols: Any | None = None, domain: Any | None = None, check: bool | None = None, simplify: bool | None = None, rational: Any | None = None, minimal: bool | None = None, force: bool | None = None, implicit: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if equations is not None:
        params["equations"] = equations
    if symbols is not None:
        params["symbols"] = symbols
    if domain is not None:
        params["domain"] = domain
    if check is not None:
        params["check"] = check
    if simplify is not None:
        params["simplify"] = simplify
    if rational is not None:
        params["rational"] = rational
    if minimal is not None:
        params["minimal"] = minimal
    if force is not None:
        params["force"] = force
    if implicit is not None:
        params["implicit"] = implicit


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_equation_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())


def sympy_mcp_matrix_operation(operation: str, data: Any, rational: bool | None = None, nrows: Any | None = None, ncols: Any | None = None, simplify: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if operation is not None:
        params["operation"] = operation
    if data is not None:
        params["data"] = data
    if rational is not None:
        params["rational"] = rational
    if nrows is not None:
        params["nrows"] = nrows
    if ncols is not None:
        params["ncols"] = ncols
    if simplify is not None:
        params["simplify"] = simplify


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="fermat-mcp",
            tool_name="sympy_mcp_matrix_operation",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['mpl_mcp_plot_barchart', 'mpl_mcp_plot_scatter', 'mpl_mcp_plot_chart', 'mpl_mcp_plot_stem', 'mpl_mcp_plot_stack', 'mpl_mcp_eqn_chart', 'numpy_mcp_numerical_operation', 'numpy_mcp_matlib_operation', 'sympy_mcp_algebra_operation', 'sympy_mcp_calculus_operation', 'sympy_mcp_equation_operation', 'sympy_mcp_matrix_operation']
