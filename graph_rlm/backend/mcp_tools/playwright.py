"""
Auto-generated wrapper for playwright MCP server.

This module provides Python function wrappers for all tools
exposed by the playwright server.

Do not edit manually.
"""

from typing import Any


def browser_close(**kwargs) -> Any:
    """Close the page

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
            server_name="playwright",
            tool_name="browser_close",
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


def browser_resize(width: float | Any = None, height: float | Any = None, **kwargs) -> Any:
    """Resize the browser window

    Args:
        width: Width of the browser window
        height: Height of the browser window

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if width is not None:
        mcp_args["width"] = width
    if height is not None:
        mcp_args["height"] = height

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_resize",
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


def browser_console_messages(level: str | Any = None, filename: str | None = None, **kwargs) -> Any:
    """Returns all console messages

    Args:
        level: Level of the console messages to return. Each level includes the messages of more severe levels. Defaults to "info".
        filename: Filename to save the console messages to. If not provided, messages are returned as text.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if level is not None:
        mcp_args["level"] = level
    if filename is not None:
        mcp_args["filename"] = filename

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_console_messages",
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


def browser_handle_dialog(accept: bool | Any = None, promptText: str | None = None, **kwargs) -> Any:
    """Handle a dialog

    Args:
        accept: Whether to accept the dialog.
        promptText: The text of the prompt in case of a prompt dialog.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if accept is not None:
        mcp_args["accept"] = accept
    if promptText is not None:
        mcp_args["promptText"] = promptText

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_handle_dialog",
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


def browser_evaluate(function: str | Any = None, element: str | None = None, ref: str | None = None, **kwargs) -> Any:
    """Evaluate JavaScript expression on page or element

    Args:
        function: () => { /* code */ } or (element) => { /* code */ } when element is provided
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if function is not None:
        mcp_args["function"] = function
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_evaluate",
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


def browser_file_upload(paths: list[str] | None = None, **kwargs) -> Any:
    """Upload one or multiple files

    Args:
        paths: The absolute paths to the files to upload. Can be single file or multiple files. If omitted, file chooser is cancelled.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if paths is not None:
        mcp_args["paths"] = paths

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_file_upload",
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


def browser_fill_form(fields: list[dict[str, Any]] | Any = None, **kwargs) -> Any:
    """Fill multiple form fields

    Args:
        fields: Fields to fill in

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if fields is not None:
        mcp_args["fields"] = fields

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_fill_form",
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


def browser_install(**kwargs) -> Any:
    """Install the browser specified in the config. Call this if you get an error about the browser not being installed.

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
            server_name="playwright",
            tool_name="browser_install",
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


def browser_press_key(key: str | Any = None, **kwargs) -> Any:
    """Press a key on the keyboard

    Args:
        key: Name of the key to press or a character to generate, such as `ArrowLeft` or `a`

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if key is not None:
        mcp_args["key"] = key

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_press_key",
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


def browser_type(ref: str | Any = None, text: str | Any = None, element: str | None = None, submit: bool | None = None, slowly: bool | None = None, **kwargs) -> Any:
    """Type text into editable element

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot
        text: Text to type into the element
        submit: Whether to submit entered text (press Enter after)
        slowly: Whether to type one character at a time. Useful for triggering key handlers in the page. By default entire text is filled in at once.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref
    if text is not None:
        mcp_args["text"] = text
    if submit is not None:
        mcp_args["submit"] = submit
    if slowly is not None:
        mcp_args["slowly"] = slowly

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_type",
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


def browser_navigate(url: str | Any = None, **kwargs) -> Any:
    """Navigate to a URL

    Args:
        url: The URL to navigate to

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if url is not None:
        mcp_args["url"] = url

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_navigate",
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


def browser_navigate_back(**kwargs) -> Any:
    """Go back to the previous page in the history

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
            server_name="playwright",
            tool_name="browser_navigate_back",
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


def browser_network_requests(includeStatic: bool | Any = None, filename: str | None = None, **kwargs) -> Any:
    """Returns all network requests since loading the page

    Args:
        includeStatic: Whether to include successful static resources like images, fonts, scripts, etc. Defaults to false.
        filename: Filename to save the network requests to. If not provided, requests are returned as text.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if includeStatic is not None:
        mcp_args["includeStatic"] = includeStatic
    if filename is not None:
        mcp_args["filename"] = filename

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_network_requests",
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


def browser_run_code(code: str | Any = None, **kwargs) -> Any:
    """Run Playwright code snippet

    Args:
        code: A JavaScript function containing Playwright code to execute. It will be invoked with a single argument, page, which you can use for any page interaction. For example: `async (page) => { await page.getByRole('button', { name: 'Submit' }).click(); return await page.title(); }`

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if code is not None:
        mcp_args["code"] = code

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_run_code",
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


def browser_take_screenshot(type: str | Any = None, filename: str | None = None, element: str | None = None, ref: str | None = None, fullPage: bool | None = None, **kwargs) -> Any:
    """Take a screenshot of the current page. You can't perform actions based on the screenshot, use browser_snapshot for actions.

    Args:
        type: Image format for the screenshot. Default is png.
        filename: File name to save the screenshot to. Defaults to `page-{timestamp}.{png|jpeg}` if not specified. Prefer relative file names to stay within the output directory.
        element: Human-readable element description used to obtain permission to screenshot the element. If not provided, the screenshot will be taken of viewport. If element is provided, ref must be provided too.
        ref: Exact target element reference from the page snapshot. If not provided, the screenshot will be taken of viewport. If ref is provided, element must be provided too.
        fullPage: When true, takes a screenshot of the full scrollable page, instead of the currently visible viewport. Cannot be used with element screenshots.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    # Resilience: Handle 'type' keyword safety and aliases
    actual_type = type or kwargs.get('node_type') or kwargs.get('thought_type')
    if actual_type is not None:
        mcp_args["type"] = actual_type
    if filename is not None:
        mcp_args["filename"] = filename
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref
    if fullPage is not None:
        mcp_args["fullPage"] = fullPage

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_take_screenshot",
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


def browser_snapshot(filename: str | None = None, **kwargs) -> Any:
    """Capture accessibility snapshot of the current page, this is better than screenshot

    Args:
        filename: Save snapshot to markdown file instead of returning it in the response.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if filename is not None:
        mcp_args["filename"] = filename

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_snapshot",
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


def browser_click(ref: str | Any = None, element: str | None = None, doubleClick: bool | None = None, button: str | None = None, modifiers: list[str] | None = None, **kwargs) -> Any:
    """Perform click on a web page

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot
        doubleClick: Whether to perform a double click instead of a single click
        button: Button to click, defaults to left
        modifiers: Modifier keys to press

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref
    if doubleClick is not None:
        mcp_args["doubleClick"] = doubleClick
    if button is not None:
        mcp_args["button"] = button
    if modifiers is not None:
        mcp_args["modifiers"] = modifiers

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_click",
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


def browser_drag(startElement: str | Any = None, startRef: str | Any = None, endElement: str | Any = None, endRef: str | Any = None, **kwargs) -> Any:
    """Perform drag and drop between two elements

    Args:
        startElement: Human-readable source element description used to obtain the permission to interact with the element
        startRef: Exact source element reference from the page snapshot
        endElement: Human-readable target element description used to obtain the permission to interact with the element
        endRef: Exact target element reference from the page snapshot

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if startElement is not None:
        mcp_args["startElement"] = startElement
    if startRef is not None:
        mcp_args["startRef"] = startRef
    if endElement is not None:
        mcp_args["endElement"] = endElement
    if endRef is not None:
        mcp_args["endRef"] = endRef

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_drag",
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


def browser_hover(ref: str | Any = None, element: str | None = None, **kwargs) -> Any:
    """Hover over element on page

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_hover",
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


def browser_select_option(ref: str | Any = None, values: list[str] | Any = None, element: str | None = None, **kwargs) -> Any:
    """Select an option in a dropdown

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot
        values: Array of values to select in the dropdown. This can be a single value or multiple values.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if element is not None:
        mcp_args["element"] = element
    if ref is not None:
        mcp_args["ref"] = ref
    if values is not None:
        mcp_args["values"] = values

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_select_option",
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


def browser_tabs(action: str | Any = None, index: float | None = None, **kwargs) -> Any:
    """List, create, close, or select a browser tab.

    Args:
        action: Operation to perform
        index: Tab index, used for close/select. If omitted for close, current tab is closed.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if action is not None:
        mcp_args["action"] = action
    if index is not None:
        mcp_args["index"] = index

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_tabs",
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


def browser_wait_for(time: float | None = None, text: str | None = None, textGone: str | None = None, **kwargs) -> Any:
    """Wait for text to appear or disappear or a specified time to pass

    Args:
        time: The time to wait in seconds
        text: The text to wait for
        textGone: The text to wait for to disappear

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if time is not None:
        mcp_args["time"] = time
    if text is not None:
        mcp_args["text"] = text
    if textGone is not None:
        mcp_args["textGone"] = textGone

    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_wait_for",
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
    return ['browser_close', 'browser_resize', 'browser_console_messages', 'browser_handle_dialog', 'browser_evaluate', 'browser_file_upload', 'browser_fill_form', 'browser_install', 'browser_press_key', 'browser_type', 'browser_navigate', 'browser_navigate_back', 'browser_network_requests', 'browser_run_code', 'browser_take_screenshot', 'browser_snapshot', 'browser_click', 'browser_drag', 'browser_hover', 'browser_select_option', 'browser_tabs', 'browser_wait_for']
