"""
Auto-generated wrapper for docker-hub MCP server.

This module provides Python function wrappers for all tools
exposed by the docker-hub server.

Do not edit manually.
"""

from typing import Any


def list_repositories_by_namespace(namespace: str, page: float | None = None, page_size: float | None = None, ordering: str | None = None, media_types: str | None = None, content_types: str | None = None) -> Any:
    """List paginated repositories by namespace

    Args:
        namespace: The namespace to list repositories from
        page: The page number to list repositories from
        page_size: The page size to list repositories from
        ordering: The ordering of the repositories. Use "-" to reverse the ordering. For example, "last_updated" will order the repositories by last updated in descending order while "-last_updated" will order the repositories by last updated in ascending order.
        media_types: Comma-delimited list of media types. Only repositories containing one or more artifacts with one of these media types will be returned. Default is empty to get all repositories.
        content_types: Comma-delimited list of content types. Only repositories containing one or more artifacts with one of these content types will be returned. Default is empty to get all repositories.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if ordering is not None:
        params["ordering"] = ordering
    if media_types is not None:
        params["media_types"] = media_types
    if content_types is not None:
        params["content_types"] = content_types


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="listRepositoriesByNamespace",
            arguments=params,
        )
    return asyncio.run(_async_call())


def create_repository(namespace: str, name: str | None = None, description: str | None = None, is_private: bool | None = None, full_description: str | None = None, registry: str | None = None) -> Any:
    """Create a new repository in the given namespace. You MUST ask the user for the repository name and if the repository has to be public or private. Can optionally pass a description.
IMPORTANT: Before calling this tool, you must ensure you have:
 The repository name (name).

    Args:
        namespace: The namespace of the repository. Required.
        name: The name of the repository (required). Must contain a combination of alphanumeric characters and may contain the special characters ., _, or -. Letters must be lowercase.
        description: The description of the repository
        is_private: Whether the repository is private
        full_description: A detailed description of the repository
        registry: The registry to create the repository in

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if name is not None:
        params["name"] = name
    if description is not None:
        params["description"] = description
    if is_private is not None:
        params["is_private"] = is_private
    if full_description is not None:
        params["full_description"] = full_description
    if registry is not None:
        params["registry"] = registry


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="createRepository",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_repository_info(namespace: str, repository: str) -> Any:
    """Get the details of a repository in the given namespace.

    Args:
        namespace: The namespace of the repository (required). If not provided the `library` namespace will be used for official images.
        repository: The repository name (required)

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="getRepositoryInfo",
            arguments=params,
        )
    return asyncio.run(_async_call())


def update_repository_info(namespace: str, repository: str, description: str | None = None, full_description: str | None = None, status: Any | None = None) -> Any:
    """Update the details of a repository in the given namespace. Description, overview and status are the only fields that can be updated. While description and overview changes are fine, a status change is a dangerous operation so the user must explicitly ask for it.

    Args:
        namespace: The namespace of the repository (required)
        repository: The repository name (required)
        description: The description of the repository. If user asks for updating the description of the repository, this is the field that should be updated.
        full_description: The full description (overview)of the repository. If user asks for updating the full description or the overview of the repository, this is the field that should be updated. 
        status: The status of the repository. If user asks for updating the status of the repository, this is the field that should be updated. This is a dangerous operation and should be done with caution so user must be prompted to confirm the operation. Valid status are `active` (1) and `inactive` (0). Normally do not update the status if it is not strictly required by the user. It is not possible to change an `inactive` repository to `active` if it has no images.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository
    if description is not None:
        params["description"] = description
    if full_description is not None:
        params["full_description"] = full_description
    if status is not None:
        params["status"] = status


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="updateRepositoryInfo",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_repository(namespace: str, repository: str) -> Any:
    """Check if a repository exists in the given namespace.

    Args:
        namespace: 
        repository: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="checkRepository",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_repository_tags(repository: str, namespace: str | None = None, page: float | None = None, page_size: float | None = None, architecture: str | None = None, os: str | None = None) -> Any:
    """List paginated tags by repository

    Args:
        namespace: The namespace of the repository. If not provided the 'library' namespace will be used for official images.
        repository: The repository to list tags from
        page: The page number to list tags from
        page_size: The page size to list tags from
        architecture: The architecture to list tags from. If not provided, all architectures will be listed.
        os: The operating system to list tags from. If not provided, all operating systems will be listed.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if architecture is not None:
        params["architecture"] = architecture
    if os is not None:
        params["os"] = os


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="listRepositoryTags",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_repository_tag(namespace: str, repository: str, tag: str) -> Any:
    """Get the details of a tag in a repository. It can be use to show the latest tag details for example.

    Args:
        namespace: 
        repository: 
        tag: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository
    if tag is not None:
        params["tag"] = tag


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="getRepositoryTag",
            arguments=params,
        )
    return asyncio.run(_async_call())


def check_repository_tag(namespace: str, repository: str, tag: str) -> Any:
    """Check if a tag exists in a repository

    Args:
        namespace: 
        repository: 
        tag: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if namespace is not None:
        params["namespace"] = namespace
    if repository is not None:
        params["repository"] = repository
    if tag is not None:
        params["tag"] = tag


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="checkRepositoryTag",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_namespaces(page: float | None = None, page_size: float | None = None) -> Any:
    """List paginated namespaces

    Args:
        page: The page number to list repositories from
        page_size: The page size to list repositories from

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="listNamespaces",
            arguments=params,
        )
    return asyncio.run(_async_call())


def get_personal_namespace() -> Any:
    """Get the personal namespace name

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="getPersonalNamespace",
            arguments=params,
        )
    return asyncio.run(_async_call())


def list_all_namespaces_member_of() -> Any:
    """List all namespaces the user is a member of

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="listAllNamespacesMemberOf",
            arguments=params,
        )
    return asyncio.run(_async_call())


def search(query: str, badges: list[str] | None = None, type: str | None = None, categories: list[str] | None = None, architectures: list[str] | None = None, operating_systems: list[str] | None = None, extension_reviewed: bool | None = None, from: float | None = None, size: float | None = None, sort: Any | None = None, order: Any | None = None, images: list[str] | None = None) -> Any:
    """Search for repositories in Docker Hub. It sorts results by best match if no sort criteria is provided. If user asks for secure, production-ready images the "dockerHardenedImages" tool should be called first to get the list of DHI images available in the user organisations (if any) and fallback to search tool if no DHI images are available or user is not authenticated.

    Args:
        query: The query to search for
        badges: The trusted content to search for
        type: The type of the repository to search for
        categories: The categories names to filter search results
        architectures: The architectures to filter search results
        operating_systems: The operating systems to filter search results
        extension_reviewed: Whether to filter search results to only include reviewed extensions
        from: The number of repositories to skip
        size: The number of repositories to return
        sort: The criteria to sort the search results by. If the `sort` field is not set, the best match is used by default. When search extensions, documents are sort alphabetically if none is provided. Do not use it unless user explicitly asks for it.
        order: The order to sort the search results by
        images: The images to filter search results

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if query is not None:
        params["query"] = query
    if badges is not None:
        params["badges"] = badges
    if type is not None:
        params["type"] = type
    if categories is not None:
        params["categories"] = categories
    if architectures is not None:
        params["architectures"] = architectures
    if operating_systems is not None:
        params["operating_systems"] = operating_systems
    if extension_reviewed is not None:
        params["extension_reviewed"] = extension_reviewed
    if from is not None:
        params["from"] = from
    if size is not None:
        params["size"] = size
    if sort is not None:
        params["sort"] = sort
    if order is not None:
        params["order"] = order
    if images is not None:
        params["images"] = images


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="search",
            arguments=params,
        )
    return asyncio.run(_async_call())


def docker_hardened_images(organisation: str) -> Any:
    """This API is used to list Docker Hardened Images (DHIs) available in the user organisations. The tool takes the organisation name as input and returns the list of DHI images available in the organisation. It depends on the "listNamespaces" tool to be called first to get the list of organisations the user has access to.

    Args:
        organisation: The organisation for which the DHIs are listed for. If user does not explicitly ask for a specific organisation, the "listNamespaces" tool should be called first to get the list of organisations the user has access to.

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool

    # Build parameters dict, excluding None values
    params = {}
    if organisation is not None:
        params["organisation"] = organisation


    import asyncio
    async def _async_call():
        return await call_mcp_tool(
            server_name="docker-hub",
            tool_name="dockerHardenedImages",
            arguments=params,
        )
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['listRepositoriesByNamespace', 'createRepository', 'getRepositoryInfo', 'updateRepositoryInfo', 'checkRepository', 'listRepositoryTags', 'getRepositoryTag', 'checkRepositoryTag', 'listNamespaces', 'getPersonalNamespace', 'listAllNamespacesMemberOf', 'search', 'dockerHardenedImages']
