"""
Auto-generated wrapper for chroma MCP server.

This module provides Python function wrappers for all tools
exposed by the chroma server.

Do not edit manually.
"""

from typing import Any


def chroma_list_collections(limit: Any | None = None, offset: Any | None = None, **kwargs) -> Any:
    """List all collection names in the Chroma database with pagination support.

Args:
    limit: Optional maximum number of collections to return
    offset: Optional number of collections to skip before returning results

Returns:
    List of collection names or ["__NO_COLLECTIONS_FOUND__"] if database is empty


    Args:
        limit: 
        offset: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if limit is not None:
        mcp_args["limit"] = limit
    if offset is not None:
        mcp_args["offset"] = offset

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_list_collections",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_create_collection(collection_name: str | Any = None, embedding_function_name: str | None = None, metadata: Any | None = None, **kwargs) -> Any:
    """Create a new Chroma collection with configurable HNSW parameters.

Args:
    collection_name: Name of the collection to create
    embedding_function_name: Name of the embedding function to use. Options: 'default', 'cohere', 'openai', 'jina', 'voyageai', 'ollama', 'roboflow'
    metadata: Optional metadata dict to add to the collection


    Args:
        collection_name: 
        embedding_function_name: 
        metadata: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if embedding_function_name is not None:
        mcp_args["embedding_function_name"] = embedding_function_name
    if metadata is not None:
        mcp_args["metadata"] = metadata

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_create_collection",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_peek_collection(collection_name: str | Any = None, limit: int | None = None, **kwargs) -> Any:
    """Peek at documents in a Chroma collection.

Args:
    collection_name: Name of the collection to peek into
    limit: Number of documents to peek at


    Args:
        collection_name: 
        limit: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if limit is not None:
        mcp_args["limit"] = limit

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_peek_collection",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_get_collection_info(collection_name: str | Any = None, **kwargs) -> Any:
    """Get information about a Chroma collection.

Args:
    collection_name: Name of the collection to get info about


    Args:
        collection_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_get_collection_info",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_get_collection_count(collection_name: str | Any = None, **kwargs) -> Any:
    """Get the number of documents in a Chroma collection.

Args:
    collection_name: Name of the collection to count


    Args:
        collection_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_get_collection_count",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_modify_collection(collection_name: str | Any = None, new_name: Any | None = None, new_metadata: Any | None = None, **kwargs) -> Any:
    """Modify a Chroma collection's name or metadata.

Args:
    collection_name: Name of the collection to modify
    new_name: Optional new name for the collection
    new_metadata: Optional new metadata for the collection


    Args:
        collection_name: 
        new_name: 
        new_metadata: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if new_name is not None:
        mcp_args["new_name"] = new_name
    if new_metadata is not None:
        mcp_args["new_metadata"] = new_metadata

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_modify_collection",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_fork_collection(collection_name: str | Any = None, new_collection_name: str | Any = None, **kwargs) -> Any:
    """Fork a Chroma collection.

Args:
    collection_name: Name of the collection to fork
    new_collection_name: Name of the new collection to create
    metadata: Optional metadata dict to add to the new collection


    Args:
        collection_name: 
        new_collection_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if new_collection_name is not None:
        mcp_args["new_collection_name"] = new_collection_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_fork_collection",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_delete_collection(collection_name: str | Any = None, **kwargs) -> Any:
    """Delete a Chroma collection.

Args:
    collection_name: Name of the collection to delete


    Args:
        collection_name: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_delete_collection",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_add_documents(collection_name: str | Any = None, documents: list[str] | Any = None, ids: list[str] | Any = None, metadatas: Any | None = None, **kwargs) -> Any:
    """Add documents to a Chroma collection.

Args:
    collection_name: Name of the collection to add documents to
    documents: List of text documents to add
    ids: List of IDs for the documents (required)
    metadatas: Optional list of metadata dictionaries for each document


    Args:
        collection_name: 
        documents: 
        ids: 
        metadatas: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if documents is not None:
        mcp_args["documents"] = documents
    if ids is not None:
        mcp_args["ids"] = ids
    if metadatas is not None:
        mcp_args["metadatas"] = metadatas

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_add_documents",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_query_documents(collection_name: str | Any = None, query_texts: list[str] | Any = None, n_results: int | None = None, where: Any | None = None, where_document: Any | None = None, include: list[str] | None = None, **kwargs) -> Any:
    """Query documents from a Chroma collection with advanced filtering.

Args:
    collection_name: Name of the collection to query
    query_texts: List of query texts to search for
    n_results: Number of results to return per query
    where: Optional metadata filters using Chroma's query operators
           Examples:
           - Simple equality: {"metadata_field": "value"}
           - Comparison: {"metadata_field": {"$gt": 5}}
           - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
           - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
    where_document: Optional document content filters
           Examples:
           - Contains: {"$contains": "value"}
           - Not contains: {"$not_contains": "value"}
           - Regex: {"$regex": "[a-z]+"}
           - Not regex: {"$not_regex": "[a-z]+"}
           - Logical AND: {"$and": [{"$contains": "value1"}, {"$not_regex": "[a-z]+"}]}
           - Logical OR: {"$or": [{"$regex": "[a-z]+"}, {"$not_contains": "value2"}]}
    include: List of what to include in response. By default, this will include documents, metadatas, and distances.


    Args:
        collection_name: 
        query_texts: 
        n_results: 
        where: 
        where_document: 
        include: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if query_texts is not None:
        mcp_args["query_texts"] = query_texts
    if n_results is not None:
        mcp_args["n_results"] = n_results
    if where is not None:
        mcp_args["where"] = where
    if where_document is not None:
        mcp_args["where_document"] = where_document
    if include is not None:
        mcp_args["include"] = include

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_query_documents",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_get_documents(collection_name: str | Any = None, ids: Any | None = None, where: Any | None = None, where_document: Any | None = None, include: list[str] | None = None, limit: Any | None = None, offset: Any | None = None, **kwargs) -> Any:
    """Get documents from a Chroma collection with optional filtering.

Args:
    collection_name: Name of the collection to get documents from
    ids: Optional list of document IDs to retrieve
    where: Optional metadata filters using Chroma's query operators
           Examples:
           - Simple equality: {"metadata_field": "value"}
           - Comparison: {"metadata_field": {"$gt": 5}}
           - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
           - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
    where_document: Optional document content filters
           Examples:
           - Contains: {"$contains": "value"}
           - Not contains: {"$not_contains": "value"}
           - Regex: {"$regex": "[a-z]+"}
           - Not regex: {"$not_regex": "[a-z]+"}
           - Logical AND: {"$and": [{"$contains": "value1"}, {"$not_regex": "[a-z]+"}]}
           - Logical OR: {"$or": [{"$regex": "[a-z]+"}, {"$not_contains": "value2"}]}
    include: List of what to include in response. By default, this will include documents, and metadatas.
    limit: Optional maximum number of documents to return
    offset: Optional number of documents to skip before returning results

Returns:
    Dictionary containing the matching documents, their IDs, and requested includes


    Args:
        collection_name: 
        ids: 
        where: 
        where_document: 
        include: 
        limit: 
        offset: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if ids is not None:
        mcp_args["ids"] = ids
    if where is not None:
        mcp_args["where"] = where
    if where_document is not None:
        mcp_args["where_document"] = where_document
    if include is not None:
        mcp_args["include"] = include
    if limit is not None:
        mcp_args["limit"] = limit
    if offset is not None:
        mcp_args["offset"] = offset

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_get_documents",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_update_documents(collection_name: str | Any = None, ids: list[str] | Any = None, embeddings: Any | None = None, metadatas: Any | None = None, documents: Any | None = None, **kwargs) -> Any:
    """Update documents in a Chroma collection.

Args:
    collection_name: Name of the collection to update documents in
    ids: List of document IDs to update (required)
    embeddings: Optional list of new embeddings for the documents.
                Must match length of ids if provided.
    metadatas: Optional list of new metadata dictionaries for the documents.
               Must match length of ids if provided.
    documents: Optional list of new text documents.
               Must match length of ids if provided.

Returns:
    A confirmation message indicating the number of documents updated.

Raises:
    ValueError: If 'ids' is empty or if none of 'embeddings', 'metadatas',
                or 'documents' are provided, or if the length of provided
                update lists does not match the length of 'ids'.
    Exception: If the collection does not exist or if the update operation fails.


    Args:
        collection_name: 
        ids: 
        embeddings: 
        metadatas: 
        documents: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if ids is not None:
        mcp_args["ids"] = ids
    if embeddings is not None:
        mcp_args["embeddings"] = embeddings
    if metadatas is not None:
        mcp_args["metadatas"] = metadatas
    if documents is not None:
        mcp_args["documents"] = documents

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_update_documents",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())


def chroma_delete_documents(collection_name: str | Any = None, ids: list[str] | Any = None, **kwargs) -> Any:
    """Delete documents from a Chroma collection.

Args:
    collection_name: Name of the collection to delete documents from
    ids: List of document IDs to delete

Returns:
    A confirmation message indicating the number of documents deleted.

Raises:
    ValueError: If 'ids' is empty
    Exception: If the collection does not exist or if the delete operation fails.


    Args:
        collection_name: 
        ids: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if collection_name is not None:
        mcp_args["collection_name"] = collection_name
    if ids is not None:
        mcp_args["ids"] = ids

    async def _async_call():
        return await call_mcp_tool(
            server_name="chroma",
            tool_name="chroma_delete_documents",
            arguments=mcp_args,
        )

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we are in an async context, return the coroutine
            return _async_call()
    except RuntimeError:
        pass

    # If we are in a sync context (e.g. standard REPL), run to completion
    return asyncio.run(_async_call())



def list_tools() -> list[str]:
    """Get list of all available tools in this server."""
    return ['chroma_list_collections', 'chroma_create_collection', 'chroma_peek_collection', 'chroma_get_collection_info', 'chroma_get_collection_count', 'chroma_modify_collection', 'chroma_fork_collection', 'chroma_delete_collection', 'chroma_add_documents', 'chroma_query_documents', 'chroma_get_documents', 'chroma_update_documents', 'chroma_delete_documents']
