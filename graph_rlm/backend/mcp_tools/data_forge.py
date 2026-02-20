"""
Auto-generated wrapper for data-forge MCP server.

This module provides Python function wrappers for all tools
exposed by the data-forge server.

Do not edit manually.
"""

from typing import Any


def load_data(file_path: str | Any = None, alias: Any | None = None, engine: str | None = None, **kwargs) -> Any:
    """
Loads a dataset file (CSV/Parquet) into the server's working memory.

Args:
    file_path: Absolute path to the data file.
    alias: Optional human-readable name for the dataset.
    engine: The processing engine to use. Supports 'pandas' (default) and 'polars' (high performance).

Returns:
    A unique dataset_id (e.g., 'df_a1b2c3d4') to be used in subsequent tool calls.


    Args:
        file_path: 
        alias: 
        engine: 

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
    if file_path is not None:
        mcp_args["file_path"] = file_path
    if alias is not None:
        mcp_args["alias"] = alias
    if engine is not None:
        mcp_args["engine"] = engine

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="load_data",
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


def get_dataset_info(dataset_id: str | Any = None, **kwargs) -> Any:
    """
Returns the schema and summary (df.info()) of a loaded dataset.

Args:
    dataset_id: The ID returned by load_data.


    Args:
        dataset_id: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="get_dataset_info",
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


def list_active_datasets(**kwargs) -> Any:
    """
Lists all currently loaded datasets and their IDs.


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

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="list_active_datasets",
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


def validate_dataset(dataset_id: str | Any = None, schema: dict[str, Any] | Any = None, **kwargs) -> Any:
    """
Validates the dataset against a provided schema using Pandera.

Args:
    dataset_id: The ID of the dataset to validate.
    schema: A dictionary defining the schema (columns, checks, coercion).
            Example: {"columns": {"col_a": {"type": "int", "checks": {"ge": 0}}}}


    Args:
        dataset_id: 
        schema: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if schema is not None:
        mcp_args["schema"] = schema

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="validate_dataset",
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


def clean_dataset(dataset_id: str | Any = None, operations: list[Any] | Any = None, **kwargs) -> Any:
    """
Applies a sequence of Pyjanitor cleaning functions to the dataset.

Args:
    dataset_id: The ID of the dataset.
    operations: List of operations. Can be a string (e.g. "clean_names") or a dict
                (e.g. {"method": "currency_column_to_numeric", "args": ["Price"]})


    Args:
        dataset_id: 
        operations: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if operations is not None:
        mcp_args["operations"] = operations

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="clean_dataset",
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


def get_dataset_profile(dataset_id: str | Any = None, **kwargs) -> Any:
    """
Generates a statistical profile of the dataset using YData Profiling.
Returns a JSON summary of key insights (alerts, variable list).

Args:
    dataset_id: The ID of the dataset.


    Args:
        dataset_id: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="get_dataset_profile",
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


def generate_chart(dataset_id: str | Any = None, chart_type: str | Any = None, x: Any | None = None, y: Any | None = None, title: Any | None = None, **kwargs) -> Any:
    """
Generates a chart from the dataset and saves it as an image.

Args:
    dataset_id: The ID of the dataset.
    chart_type: Type of chart to generate.
    x: Column name for X axis.
    y: Column name for Y axis (optional for some charts).
    title: Title of the chart.

Returns:
    Absolute path to the generated PNG image.


    Args:
        dataset_id: 
        chart_type: 
        x: 
        y: 
        title: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if chart_type is not None:
        mcp_args["chart_type"] = chart_type
    if x is not None:
        mcp_args["x"] = x
    if y is not None:
        mcp_args["y"] = y
    if title is not None:
        mcp_args["title"] = title

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="generate_chart",
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


def scan_semantic_voids(dataset_id: str | Any = None, text_column: str | Any = None, **kwargs) -> Any:
    """
Performs Topological Data Analysis to find "semantic voids" or gaps in the dataset's text column.
Useful for identifying missing research topics, unaddressed customer complaints, or concept holes.
Generates a persistence barcode and 3D manifold plot.


    Args:
        dataset_id: 
        text_column: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if text_column is not None:
        mcp_args["text_column"] = text_column

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="scan_semantic_voids",
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


def run_sql_query(query: str | Any = None, dataset_id: Any | None = None, **kwargs) -> Any:
    """
Executes a SQL query on your datasets using DuckDB.
Use this to filter, aggregate, join, or reshape data.

You can reference datasets by their ID (e.g., 'SELECT * FROM df_a1b2').
If you provide a 'dataset_id' argument, you can refer to it as table 'this'.


    Args:
        query: 
        dataset_id: 

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
    if query is not None:
        mcp_args["query"] = query
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="run_sql_query",
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


def extract_signals(dataset_id: str | Any = None, value_column: str | Any = None, id_column: Any | None = None, sort_column: Any | None = None, **kwargs) -> Any:
    """
Extracts time-series signals (features) using tsfresh.

Args:
    dataset_id: The ID of the dataset.
    value_column: Column containing the time-series values.
    id_column: Column identifying separate time series (e.g. 'symbol').
    sort_column: Column to sort by (usually time/date).

Returns:
    JSON summary of the extraction result.


    Args:
        dataset_id: 
        value_column: 
        id_column: 
        sort_column: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if value_column is not None:
        mcp_args["value_column"] = value_column
    if id_column is not None:
        mcp_args["id_column"] = id_column
    if sort_column is not None:
        mcp_args["sort_column"] = sort_column

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="extract_signals",
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


def load_hf_dataset(dataset_name: str | Any = None, split: Any | None = None, config_name: Any | None = None, **kwargs) -> Any:
    """
Loads a dataset from the Hugging Face Hub (requires 'datasets' library).

Args:
    dataset_name: Name of the dataset on HF Hub (e.g., 'mnist', 'glue').
    split: The split to load (default: 'train').
    config_name: Optional configuration name (subset) of the dataset.

Returns:
    The new dataset_id.


    Args:
        dataset_name: 
        split: 
        config_name: 

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
    if dataset_name is not None:
        mcp_args["dataset_name"] = dataset_name
    if split is not None:
        mcp_args["split"] = split
    if config_name is not None:
        mcp_args["config_name"] = config_name

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="load_hf_dataset",
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


def extract_tables(url: str | Any = None, **kwargs) -> Any:
    """
Extracts tables from a web URL using pandas (requires 'lxml' or 'html5lib').

Args:
    url: The URL to scrape tables from.

Returns:
    A JSON summary of extracted tables and their IDs.


    Args:
        url: 

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
    if url is not None:
        mcp_args["url"] = url

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="extract_tables",
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


def generate_map(dataset_id: str | Any = None, lat_col: str | Any = None, lon_col: str | Any = None, **kwargs) -> Any:
    """
Generates a geospatial map (scatter plot) from a dataset.

Args:
    dataset_id: The ID of the dataset.
    lat_col: Column name for Latitude.
    lon_col: Column name for Longitude.

Returns:
    Absolute path to the generated map image (PNG).


    Args:
        dataset_id: 
        lat_col: 
        lon_col: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id
    if lat_col is not None:
        mcp_args["lat_col"] = lat_col
    if lon_col is not None:
        mcp_args["lon_col"] = lon_col

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="generate_map",
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


def start_explorer(dataset_id: str | Any = None, **kwargs) -> Any:
    """
Launches an interactive D-Tale explorer for the dataset.

Args:
    dataset_id: The ID of the dataset.

Returns:
    The URL to access the explorer.


    Args:
        dataset_id: 

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
    if dataset_id is not None:
        mcp_args["dataset_id"] = dataset_id

    # Merge additional kwargs with mapping support
    for k, v in kwargs.items():
        if v is not None:
            # Map snake_case alias to original CamelCase key if it exists in schema
            target_key = param_map.get(k, k)
            if target_key not in mcp_args:
                mcp_args[target_key] = v

    async def _async_call():
        return await call_mcp_tool(
            server_name="data-forge",
            tool_name="start_explorer",
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
    return ['load_data', 'get_dataset_info', 'list_active_datasets', 'validate_dataset', 'clean_dataset', 'get_dataset_profile', 'generate_chart', 'scan_semantic_voids', 'run_sql_query', 'extract_signals', 'load_hf_dataset', 'extract_tables', 'generate_map', 'start_explorer']
