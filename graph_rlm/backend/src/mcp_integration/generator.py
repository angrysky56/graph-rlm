"""
Python code generator for MCP tool wrappers.

Creates importable Python modules from discovered MCP server capabilities.
Implements Anthropic's "filesystem-based progressive disclosure" pattern.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ToolGenerator:
    """Generates Python wrapper code for MCP tools."""

    def __init__(self, output_dir: str | Path = "./mcp_tools") -> None:
        """
        Initialize generator.

        Args:
            output_dir: Where to write generated Python files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sanitize_name(self, name: str) -> str:
        """Convert tool/server name to valid Python identifier (snake_case).

        Args:
            name: Original name (may contain hyphens, CamelCase, etc.)

        Returns:
            Valid Python identifier in snake_case
        """

        # Handle CamelCase/PascalCase -> snake_case
        # e.g., "LocalREPL" -> "local_repl", "AstAnalyzer" -> "ast_analyzer"
        # Pass 1: Handle acronyms followed by a word (e.g., HTTPResponse -> HTTP_Response)
        # Matches any character followed by (Upper, then lower+)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)

        # Pass 2: Handle boundaries between lower/mixed and Upper (e.g., LocalREPL -> Local_REPL)
        # Matches (lower/digit) followed by (Upper)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()

        # Replace hyphens and other non-alphanumeric chars with underscores
        name = re.sub(r"[-\s]+", "_", name)

        # Remove any remaining invalid characters
        name = re.sub(r"[^a-z0-9_]", "", name)

        # Collapse multiple underscores
        name = re.sub(r"_{2,}", "_", name)

        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f"_{name}"

        return name

    def generate_tool_function(
        self,
        tool_name: str,
        tool_schema: dict[str, Any],
        server_name: str,
    ) -> str:
        """
        Generate Python function code for a single tool.

        Args:
            tool_name: Name of the tool
            tool_schema: Tool schema from discovery
            server_name: Parent server name

        Returns:
            Python function definition as string
        """
        description = tool_schema.get("description", "No description available")
        input_schema = tool_schema.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Generate function signature
        # Generate function signature
        params = []

        # Sort properties: required first, then optional
        sorted_props = sorted(properties.items(), key=lambda x: x[0] not in required)

        for param_name, param_info in sorted_props:
            param_type = self._python_type_from_json_schema(param_info)
            is_required = param_name in required

            if is_required:
                # Still make it optional in Python but handle check in body if desired
                # Actually, making it optional with None allows for kwarg resilience
                params.append(f"{param_name}: {param_type} | Any = None")
            else:
                params.append(f"{param_name}: {param_type} | None = None")

        # Always add **kwargs for resilience
        params.append("**kwargs")
        params_str = ", ".join(params)

        # Generate docstring
        docstring_lines = [f'    """{description}']

        if properties:
            docstring_lines.append("")
            docstring_lines.append("    Args:")
            for param_name, param_info in properties.items():
                param_desc = param_info.get("description", "")
                docstring_lines.append(f"        {param_name}: {param_desc}")

        docstring_lines.append("")
        docstring_lines.append("    Returns:")
        docstring_lines.append("        Tool execution result")
        docstring_lines.append('    """')
        docstring = "\n".join(docstring_lines)

        # Generate function body
        func_name = self.sanitize_name(tool_name)

        # Build resilience logic for common parameter hallucinations
        resilience_logic = ""

        # 1. Handle 'type' collisions or aliases
        if "type" in properties:
            resilience_logic += (
                "    # Resilience: Handle 'type' keyword safety and aliases\n"
            )
            resilience_logic += (
                "    actual_type = type or kwargs.get('node_type') "
                "or kwargs.get('thought_type')\n"
            )
        elif "thoughtType" in properties:
            resilience_logic += "    # Resilience: Handle aliases for 'thoughtType'\n"
            resilience_logic += (
                "    actual_thoughtType = thoughtType or kwargs.get('type') "
                "or kwargs.get('node_type') or kwargs.get('thought_type')\n"
            )

        function_code = f"""
def {func_name}({params_str}) -> Any:
{docstring}
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {{}}
"""
        if resilience_logic:
            function_code += resilience_logic

        for param_name in properties.keys():
            val_name = param_name
            if param_name == "type" and "actual_type" in resilience_logic:
                val_name = "actual_type"
            elif (
                param_name == "thoughtType" and "actual_thoughtType" in resilience_logic
            ):
                val_name = "actual_thoughtType"

            function_code += f"""    if {val_name} is not None:
        mcp_args["{param_name}"] = {val_name}
"""

        function_code += """
    async def _async_call():
        return await call_mcp_tool(
            server_name="{server_name}",
            tool_name="{tool_name}",
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
""".replace("{server_name}", server_name).replace("{tool_name}", tool_name)

        return function_code

    def _python_type_from_json_schema(self, schema: dict[str, Any]) -> str:
        """
        Convert JSON schema type to Python type hint.

        Args:
            schema: JSON schema object

        Returns:
            Python type string
        """
        schema_type = schema.get("type", "any")

        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
            "null": "None",
        }

        if schema_type in type_map:
            base_type = type_map[schema_type]

            # Handle arrays with item types
            if schema_type == "array" and "items" in schema:
                item_type = self._python_type_from_json_schema(schema["items"])
                return f"list[{item_type}]"

            # Handle objects with additional properties
            if schema_type == "object":
                return "dict[str, Any]"

            return base_type

        return "Any"

    def generate_server_module(
        self,
        server_name: str,
        server_info: dict[str, Any],
    ) -> None:
        """
        Generate Python module for an entire server.

        Creates a file like mcp_tools/chroma.py with all tools.

        Args:
            server_name: Name of the server
            server_info: Server capabilities from discovery
        """
        # Sanitize server name for filename
        module_name = self.sanitize_name(server_name)
        module_path = self.output_dir / f"{module_name}.py"

        # Start with module docstring and imports
        module_code = f'''"""
Auto-generated wrapper for {server_name} MCP server.

This module provides Python function wrappers for all tools
exposed by the {server_name} server.

Do not edit manually.
"""

from typing import Any

'''

        # Generate function for each tool
        tools = server_info.get("tools", {})

        if not tools:
            module_code += f"""
# No tools found for {server_name}
"""
        else:
            for tool_name, tool_schema in tools.items():
                try:
                    tool_func = self.generate_tool_function(
                        tool_name,
                        tool_schema,
                        server_name,
                    )
                    module_code += tool_func + "\n"
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to generate function for %s.%s: %s",
                        server_name,
                        tool_name,
                        e,
                    )
                    # Continue generating other tools
                    continue

            # Generate list_tools helper
            tool_names = list(tools.keys())
            module_code += f'''

def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return {tool_names!r}
'''

        # Write to file if changed
        if not module_path.exists() or module_path.read_text() != module_code:
            module_path.write_text(module_code)

    def generate_index_module(self, server_names: list[str]) -> None:
        """
        Generate mcp_tools/__init__.py for server discovery.

        Args:
            server_names: List of server names
        """
        init_path = self.output_dir / "__init__.py"

        init_code = '''"""
MCP Tools - Auto-generated importable wrappers for MCP servers.

This package is automatically generated by mcp-coordinator.
DO NOT EDIT MANUALLY.

Usage:
    from graph_rlm.backend.mcp_tools import list_servers
    from graph_rlm.backend.mcp_tools.chroma import query, add_documents
"""

from typing import Any
from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool as call_tool

# Explicitly mark exports to satisfy linters
call_tool = call_tool

def run_skill(name: str, args: dict | None = None) -> Any:
    """
    Execute a skill by name.

    Args:
        name: Name of the skill to execute
        args: Arguments to pass to the skill function

    Returns:
        Result of the skill execution
    """
    import asyncio
    from graph_rlm.backend.src.mcp_integration.skill_harness import execute_skill
    return asyncio.run(execute_skill(name, args or {}))

def list_servers() -> list[str]:
    """Get list of all available MCP servers."""
'''

        init_code += f"    return {server_names!r}\n"

        if not init_path.exists() or init_path.read_text() != init_code:
            init_path.write_text(init_code)

    def generate_readme(self) -> None:
        """Generate README for mcp_tools directory."""
        readme_path = self.output_dir / "README.md"

        readme_content = """# MCP Tools (Auto-Generated)

This directory contains auto-generated Python wrappers for your MCP servers.

**⚠️ DO NOT EDIT FILES IN THIS DIRECTORY MANUALLY**

Files here are generated by `graph-rlm` based on your MCP server configuration.

## Regenerating

To regenerate these files (e.g., after adding/updating servers):

```python
```python
from graph_rlm.backend.src.mcp_integration.generator import ToolGenerator, generate_from_config
# OR
# from graph_rlm.backend.src.mcp_integration.discovery import discover_all_servers

# See main.py for example
```

## Usage

```python
# Discover available servers
from graph_rlm.backend.mcp_tools import list_servers
print(list_servers())

# Import specific server tools
from graph_rlm.backend.mcp_tools.chroma import query, add_documents

# Use tools in your code
results = await query(collection="papers", query_text="transformers")
```

## Structure

Each server gets its own module:
- `mcp_tools/chroma.py` - Tools from the chroma server
- `mcp_tools/arxiv.py` - Tools from the arxiv server
- etc.

Each module provides:
- Individual tool functions with full type hints
- `list_tools()` function to see what's available
"""

        if not readme_path.exists() or readme_path.read_text() != readme_content:
            readme_path.write_text(readme_content)

    def generate_all(self, servers_info: dict[str, dict[str, Any]]) -> tuple[int, int]:
        """
        Generate complete mcp_tools package.

        Args:
            servers_info: Dictionary of server capabilities from discovery

        Returns:
            Number of servers successfully generated
        """
        server_names = []
        total_tools = 0

        # Generate module for each server
        for server_name, server_info in servers_info.items():
            if "error" in server_info:
                logger.info("Skipping %s: %s", server_name, server_info["error"])
                continue

            try:
                self.generate_server_module(server_name, server_info)
                server_names.append(server_name)
                total_tools += len(server_info.get("tools", {}))
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to generate module for %s: %s", server_name, e)
                # Continue generating other servers
                continue

        # Generate package index
        self.generate_index_module(server_names)

        # Generate README
        self.generate_readme()

        logger.info(
            "Generated %d server modules with %d tools in %s",
            len(server_names),
            total_tools,
            self.output_dir,
        )
        return len(server_names), total_tools


def generate_from_config(
    config_path: str | Path,
    output_dir: str | Path = "./mcp_tools",
) -> None:
    """
    High-level function to generate tools from config.

    This automatically runs server discovery first to ensure the latest
    capabilities are available before generation.

    Args:
        config_path: Path to MCP server configuration
        output_dir: Where to generate Python modules
    """
    from .discovery import discover_all_servers

    # Discover all servers
    print(f"Discovering servers from {config_path}...")
    servers_info = asyncio.run(discover_all_servers(config_path))

    # Generate Python wrappers
    print("Generating Python wrappers...")
    generator = ToolGenerator(output_dir)
    generator.generate_all(servers_info)

    print("✓ Generation complete!")
