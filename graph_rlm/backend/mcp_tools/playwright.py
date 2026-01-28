"""
Auto-generated wrapper for playwright MCP server.

This module provides Python function wrappers for all tools
exposed by the playwright server.

Do not edit manually.
"""

from typing import Any


def browser_close() -> Any:
    """Close the page

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_close",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_resize(width: float, height: float) -> Any:
    """Resize the browser window

    Args:
        width: Width of the browser window
        height: Height of the browser window

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_resize",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_console_messages(level: str, filename: str | None = None) -> Any:
    """Returns all console messages

    Args:
        level: Level of the console messages to return. Each level includes the messages of more severe levels. Defaults to "info".
        filename: Filename to save the console messages to. If not provided, messages are returned as text.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if level is not None:
        params["level"] = level
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_console_messages",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_handle_dialog(accept: bool, promptText: str | None = None) -> Any:
    """Handle a dialog

    Args:
        accept: Whether to accept the dialog.
        promptText: The text of the prompt in case of a prompt dialog.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if accept is not None:
        params["accept"] = accept
    if promptText is not None:
        params["promptText"] = promptText


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_handle_dialog",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_evaluate(function: str, element: str | None = None, ref: str | None = None) -> Any:
    """Evaluate JavaScript expression on page or element

    Args:
        function: () => { /* code */ } or (element) => { /* code */ } when element is provided
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if function is not None:
        params["function"] = function
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_evaluate",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_file_upload(paths: list[str] | None = None) -> Any:
    """Upload one or multiple files

    Args:
        paths: The absolute paths to the files to upload. Can be single file or multiple files. If omitted, file chooser is cancelled.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if paths is not None:
        params["paths"] = paths


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_file_upload",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_fill_form(fields: list[dict[str, Any]]) -> Any:
    """Fill multiple form fields

    Args:
        fields: Fields to fill in

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if fields is not None:
        params["fields"] = fields


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_fill_form",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_install() -> Any:
    """Install the browser specified in the config. Call this if you get an error about the browser not being installed.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_install",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_press_key(key: str) -> Any:
    """Press a key on the keyboard

    Args:
        key: Name of the key to press or a character to generate, such as `ArrowLeft` or `a`

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if key is not None:
        params["key"] = key


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_press_key",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_type(ref: str, text: str, element: str | None = None, submit: bool | None = None, slowly: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref
    if text is not None:
        params["text"] = text
    if submit is not None:
        params["submit"] = submit
    if slowly is not None:
        params["slowly"] = slowly


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_type",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_navigate(url: str) -> Any:
    """Navigate to a URL

    Args:
        url: The URL to navigate to

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if url is not None:
        params["url"] = url


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_navigate",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_navigate_back() -> Any:
    """Go back to the previous page in the history

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_navigate_back",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_network_requests(includeStatic: bool, filename: str | None = None) -> Any:
    """Returns all network requests since loading the page

    Args:
        includeStatic: Whether to include successful static resources like images, fonts, scripts, etc. Defaults to false.
        filename: Filename to save the network requests to. If not provided, requests are returned as text.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if includeStatic is not None:
        params["includeStatic"] = includeStatic
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_network_requests",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_run_code(code: str) -> Any:
    """Run Playwright code snippet

    Args:
        code: A JavaScript function containing Playwright code to execute. It will be invoked with a single argument, page, which you can use for any page interaction. For example: `async (page) => { await page.getByRole('button', { name: 'Submit' }).click(); return await page.title(); }`

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if code is not None:
        params["code"] = code


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_run_code",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_take_screenshot(type: str, filename: str | None = None, element: str | None = None, ref: str | None = None, fullPage: bool | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if type is not None:
        params["type"] = type
    if filename is not None:
        params["filename"] = filename
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref
    if fullPage is not None:
        params["fullPage"] = fullPage


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_take_screenshot",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_snapshot(filename: str | None = None) -> Any:
    """Capture accessibility snapshot of the current page, this is better than screenshot

    Args:
        filename: Save snapshot to markdown file instead of returning it in the response.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if filename is not None:
        params["filename"] = filename


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_snapshot",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_click(ref: str, element: str | None = None, doubleClick: bool | None = None, button: str | None = None, modifiers: list[str] | None = None) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref
    if doubleClick is not None:
        params["doubleClick"] = doubleClick
    if button is not None:
        params["button"] = button
    if modifiers is not None:
        params["modifiers"] = modifiers


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_click",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_drag(startElement: str, startRef: str, endElement: str, endRef: str) -> Any:
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

    # Build parameters dict, excluding None values
    params = {}
    if startElement is not None:
        params["startElement"] = startElement
    if startRef is not None:
        params["startRef"] = startRef
    if endElement is not None:
        params["endElement"] = endElement
    if endRef is not None:
        params["endRef"] = endRef


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_drag",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_hover(ref: str, element: str | None = None) -> Any:
    """Hover over element on page

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_hover",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_select_option(ref: str, values: list[str], element: str | None = None) -> Any:
    """Select an option in a dropdown

    Args:
        element: Human-readable element description used to obtain permission to interact with the element
        ref: Exact target element reference from the page snapshot
        values: Array of values to select in the dropdown. This can be a single value or multiple values.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if element is not None:
        params["element"] = element
    if ref is not None:
        params["ref"] = ref
    if values is not None:
        params["values"] = values


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_select_option",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_tabs(action: str, index: float | None = None) -> Any:
    """List, create, close, or select a browser tab.

    Args:
        action: Operation to perform
        index: Tab index, used for close/select. If omitted for close, current tab is closed.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if action is not None:
        params["action"] = action
    if index is not None:
        params["index"] = index


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_tabs",
            arguments=params,
        )
    return asyncio.run(_async_call())


def browser_wait_for(time: float | None = None, text: str | None = None, textGone: str | None = None) -> Any:
    """Wait for text to appear or disappear or a specified time to pass

    Args:
        time: The time to wait in seconds
        text: The text to wait for
        textGone: The text to wait for to disappear

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if time is not None:
        params["time"] = time
    if text is not None:
        params["text"] = text
    if textGone is not None:
        params["textGone"] = textGone


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="playwright",
            tool_name="browser_wait_for",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['browser_close', 'browser_resize', 'browser_console_messages', 'browser_handle_dialog', 'browser_evaluate', 'browser_file_upload', 'browser_fill_form', 'browser_install', 'browser_press_key', 'browser_type', 'browser_navigate', 'browser_navigate_back', 'browser_network_requests', 'browser_run_code', 'browser_take_screenshot', 'browser_snapshot', 'browser_click', 'browser_drag', 'browser_hover', 'browser_select_option', 'browser_tabs', 'browser_wait_for']
