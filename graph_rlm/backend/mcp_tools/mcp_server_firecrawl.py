"""
Auto-generated wrapper for mcp-server-firecrawl MCP server.

This module provides Python function wrappers for all tools
exposed by the mcp-server-firecrawl server.

Do not edit manually.
"""

from typing import Any


def firecrawl_scrape(url: str | Any = None, formats: list[Any] | None = None, parsers: list[Any] | None = None, onlyMainContent: bool | None = None, includeTags: list[str] | None = None, excludeTags: list[str] | None = None, waitFor: float | None = None, actions: list[dict[str, Any]] | None = None, mobile: bool | None = None, skipTlsVerification: bool | None = None, removeBase64Images: bool | None = None, location: dict[str, Any] | None = None, storeInCache: bool | None = None, zeroDataRetention: bool | None = None, maxAge: float | None = None, proxy: str | None = None, **kwargs) -> Any:
    """
Scrape content from a single URL with advanced options.
This is the most powerful, fastest and most reliable scraper tool, if available you should always default to using this tool for any web scraping needs.

**Best for:** Single page content extraction, when you know exactly which page contains the information.
**Not recommended for:** Multiple pages (call scrape multiple times or use crawl), unknown page location (use search).
**Common mistakes:** Using markdown format when extracting specific data points (use JSON instead).
**Other Features:** Use 'branding' format to extract brand identity (colors, fonts, typography, spacing, UI components) for design analysis or style replication.

**CRITICAL - Format Selection (you MUST follow this):**
When the user asks for SPECIFIC data points, you MUST use JSON format with a schema. Only use markdown when the user needs the ENTIRE page content.

**Use JSON format when user asks for:**
- Parameters, fields, or specifications (e.g., "get the header parameters", "what are the required fields")
- Prices, numbers, or structured data (e.g., "extract the pricing", "get the product details")
- API details, endpoints, or technical specs (e.g., "find the authentication endpoint")
- Lists of items or properties (e.g., "list the features", "get all the options")
- Any specific piece of information from a page

**Use markdown format ONLY when:**
- User wants to read/summarize an entire article or blog post
- User needs to see all content on a page without specific extraction
- User explicitly asks for the full page content

**Handling JavaScript-rendered pages (SPAs):**
If JSON extraction returns empty, minimal, or just navigation content, the page is likely JavaScript-rendered or the content is on a different URL. Try these steps IN ORDER:
1. **Add waitFor parameter:** Set `waitFor: 5000` to `waitFor: 10000` to allow JavaScript to render before extraction
2. **Try a different URL:** If the URL has a hash fragment (#section), try the base URL or look for a direct page URL
3. **Use firecrawl_map to find the correct page:** Large documentation sites or SPAs often spread content across multiple URLs. Use `firecrawl_map` with a `search` parameter to discover the specific page containing your target content, then scrape that URL directly.
   Example: If scraping "https://docs.example.com/reference" fails to find webhook parameters, use `firecrawl_map` with `{"url": "https://docs.example.com/reference", "search": "webhook"}` to find URLs like "/reference/webhook-events", then scrape that specific page.
4. **Use firecrawl_agent:** As a last resort for heavily dynamic pages where map+scrape still fails, use the agent which can autonomously navigate and research

**Usage Example (JSON format - REQUIRED for specific data extraction):**
```json
{
  "name": "firecrawl_scrape",
  "arguments": {
    "url": "https://example.com/api-docs",
    "formats": [{
      "type": "json",
      "prompt": "Extract the header parameters for the authentication endpoint",
      "schema": {
        "type": "object",
        "properties": {
          "parameters": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": { "type": "string" },
                "type": { "type": "string" },
                "required": { "type": "boolean" },
                "description": { "type": "string" }
              }
            }
          }
        }
      }
    }]
  }
}
```
**Usage Example (markdown format - ONLY when full content genuinely needed):**
```json
{
  "name": "firecrawl_scrape",
  "arguments": {
    "url": "https://example.com/article",
    "formats": ["markdown"],
    "onlyMainContent": true
  }
}
```
**Usage Example (branding format - extract brand identity):**
```json
{
  "name": "firecrawl_scrape",
  "arguments": {
    "url": "https://example.com",
    "formats": ["branding"]
  }
}
```
**Branding format:** Extracts comprehensive brand identity (colors, fonts, typography, spacing, logo, UI components) for design analysis or style replication.
**Performance:** Add maxAge parameter for 500% faster scrapes using cached data.
**Returns:** JSON structured data, markdown, branding profile, or other formats as specified.



    Args:
        url: 
        formats: 
        parsers: 
        onlyMainContent: 
        includeTags: 
        excludeTags: 
        waitFor: 
        actions: 
        mobile: 
        skipTlsVerification: 
        removeBase64Images: 
        location: 
        storeInCache: 
        zeroDataRetention: 
        maxAge: 
        proxy: 

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
    if formats is not None:
        mcp_args["formats"] = formats
    if parsers is not None:
        mcp_args["parsers"] = parsers
    if onlyMainContent is not None:
        mcp_args["onlyMainContent"] = onlyMainContent
    if includeTags is not None:
        mcp_args["includeTags"] = includeTags
    if excludeTags is not None:
        mcp_args["excludeTags"] = excludeTags
    if waitFor is not None:
        mcp_args["waitFor"] = waitFor
    if actions is not None:
        mcp_args["actions"] = actions
    if mobile is not None:
        mcp_args["mobile"] = mobile
    if skipTlsVerification is not None:
        mcp_args["skipTlsVerification"] = skipTlsVerification
    if removeBase64Images is not None:
        mcp_args["removeBase64Images"] = removeBase64Images
    if location is not None:
        mcp_args["location"] = location
    if storeInCache is not None:
        mcp_args["storeInCache"] = storeInCache
    if zeroDataRetention is not None:
        mcp_args["zeroDataRetention"] = zeroDataRetention
    if maxAge is not None:
        mcp_args["maxAge"] = maxAge
    if proxy is not None:
        mcp_args["proxy"] = proxy

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_scrape",
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


def firecrawl_map(url: str | Any = None, search: str | None = None, sitemap: str | None = None, includeSubdomains: bool | None = None, limit: float | None = None, ignoreQueryParameters: bool | None = None, **kwargs) -> Any:
    """
Map a website to discover all indexed URLs on the site.

**Best for:** Discovering URLs on a website before deciding what to scrape; finding specific sections or pages within a large site; locating the correct page when scrape returns empty or incomplete results.
**Not recommended for:** When you already know which specific URL you need (use scrape); when you need the content of the pages (use scrape after mapping).
**Common mistakes:** Using crawl to discover URLs instead of map; jumping straight to firecrawl_agent when scrape fails instead of using map first to find the right page.

**IMPORTANT - Use map before agent:** If `firecrawl_scrape` returns empty, minimal, or irrelevant content, use `firecrawl_map` with the `search` parameter to find the specific page URL containing your target content. This is faster and cheaper than using `firecrawl_agent`. Only use the agent as a last resort after map+scrape fails.

**Prompt Example:** "Find the webhook documentation page on this API docs site."
**Usage Example (discover all URLs):**
```json
{
  "name": "firecrawl_map",
  "arguments": {
    "url": "https://example.com"
  }
}
```
**Usage Example (search for specific content - RECOMMENDED when scrape fails):**
```json
{
  "name": "firecrawl_map",
  "arguments": {
    "url": "https://docs.example.com/api",
    "search": "webhook events"
  }
}
```
**Returns:** Array of URLs found on the site, filtered by search query if provided.


    Args:
        url: 
        search: 
        sitemap: 
        includeSubdomains: 
        limit: 
        ignoreQueryParameters: 

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
    if search is not None:
        mcp_args["search"] = search
    if sitemap is not None:
        mcp_args["sitemap"] = sitemap
    if includeSubdomains is not None:
        mcp_args["includeSubdomains"] = includeSubdomains
    if limit is not None:
        mcp_args["limit"] = limit
    if ignoreQueryParameters is not None:
        mcp_args["ignoreQueryParameters"] = ignoreQueryParameters

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_map",
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


def firecrawl_search(query: str | Any = None, limit: float | None = None, tbs: str | None = None, filter: str | None = None, location: str | None = None, sources: list[dict[str, Any]] | None = None, scrapeOptions: dict[str, Any] | None = None, enterprise: list[str] | None = None, **kwargs) -> Any:
    """
Search the web and optionally extract content from search results. This is the most powerful web search tool available, and if available you should always default to using this tool for any web search needs.

The query also supports search operators, that you can use if needed to refine the search:
| Operator | Functionality | Examples |
---|-|-|
| `""` | Non-fuzzy matches a string of text | `"Firecrawl"`
| `-` | Excludes certain keywords or negates other operators | `-bad`, `-site:firecrawl.dev`
| `site:` | Only returns results from a specified website | `site:firecrawl.dev`
| `inurl:` | Only returns results that include a word in the URL | `inurl:firecrawl`
| `allinurl:` | Only returns results that include multiple words in the URL | `allinurl:git firecrawl`
| `intitle:` | Only returns results that include a word in the title of the page | `intitle:Firecrawl`
| `allintitle:` | Only returns results that include multiple words in the title of the page | `allintitle:firecrawl playground`
| `related:` | Only returns results that are related to a specific domain | `related:firecrawl.dev`
| `imagesize:` | Only returns images with exact dimensions | `imagesize:1920x1080`
| `larger:` | Only returns images larger than specified dimensions | `larger:1920x1080`

**Best for:** Finding specific information across multiple websites, when you don't know which website has the information; when you need the most relevant content for a query.
**Not recommended for:** When you need to search the filesystem. When you already know which website to scrape (use scrape); when you need comprehensive coverage of a single website (use map or crawl.
**Common mistakes:** Using crawl or map for open-ended questions (use search instead).
**Prompt Example:** "Find the latest research papers on AI published in 2023."
**Sources:** web, images, news, default to web unless needed images or news.
**Scrape Options:** Only use scrapeOptions when you think it is absolutely necessary. When you do so default to a lower limit to avoid timeouts, 5 or lower.
**Optimal Workflow:** Search first using firecrawl_search without formats, then after fetching the results, use the scrape tool to get the content of the relevantpage(s) that you want to scrape

**Usage Example without formats (Preferred):**
```json
{
  "name": "firecrawl_search",
  "arguments": {
    "query": "top AI companies",
    "limit": 5,
    "sources": [
      { "type": "web" }
    ]
  }
}
```
**Usage Example with formats:**
```json
{
  "name": "firecrawl_search",
  "arguments": {
    "query": "latest AI research papers 2023",
    "limit": 5,
    "lang": "en",
    "country": "us",
    "sources": [
      { "type": "web" },
      { "type": "images" },
      { "type": "news" }
    ],
    "scrapeOptions": {
      "formats": ["markdown"],
      "onlyMainContent": true
    }
  }
}
```
**Returns:** Array of search results (with optional scraped content).


    Args:
        query: 
        limit: 
        tbs: 
        filter: 
        location: 
        sources: 
        scrapeOptions: 
        enterprise: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if query is not None:
        mcp_args["query"] = query
    if limit is not None:
        mcp_args["limit"] = limit
    if tbs is not None:
        mcp_args["tbs"] = tbs
    if filter is not None:
        mcp_args["filter"] = filter
    if location is not None:
        mcp_args["location"] = location
    if sources is not None:
        mcp_args["sources"] = sources
    if scrapeOptions is not None:
        mcp_args["scrapeOptions"] = scrapeOptions
    if enterprise is not None:
        mcp_args["enterprise"] = enterprise

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_search",
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


def firecrawl_crawl(url: str | Any = None, prompt: str | None = None, excludePaths: list[str] | None = None, includePaths: list[str] | None = None, maxDiscoveryDepth: float | None = None, sitemap: str | None = None, limit: float | None = None, allowExternalLinks: bool | None = None, allowSubdomains: bool | None = None, crawlEntireDomain: bool | None = None, delay: float | None = None, maxConcurrency: float | None = None, webhook: Any | None = None, deduplicateSimilarURLs: bool | None = None, ignoreQueryParameters: bool | None = None, scrapeOptions: dict[str, Any] | None = None, **kwargs) -> Any:
    """
 Starts a crawl job on a website and extracts content from all pages.
 
 **Best for:** Extracting content from multiple related pages, when you need comprehensive coverage.
 **Not recommended for:** Extracting content from a single page (use scrape); when token limits are a concern (use map + batch_scrape); when you need fast results (crawling can be slow).
 **Warning:** Crawl responses can be very large and may exceed token limits. Limit the crawl depth and number of pages, or use map + batch_scrape for better control.
 **Common mistakes:** Setting limit or maxDiscoveryDepth too high (causes token overflow) or too low (causes missing pages); using crawl for a single page (use scrape instead). Using a /* wildcard is not recommended.
 **Prompt Example:** "Get all blog posts from the first two levels of example.com/blog."
 **Usage Example:**
 ```json
 {
   "name": "firecrawl_crawl",
   "arguments": {
     "url": "https://example.com/blog/*",
     "maxDiscoveryDepth": 5,
     "limit": 20,
     "allowExternalLinks": false,
     "deduplicateSimilarURLs": true,
     "sitemap": "include"
   }
 }
 ```
 **Returns:** Operation ID for status checking; use firecrawl_check_crawl_status to check progress.
 
 

    Args:
        url: 
        prompt: 
        excludePaths: 
        includePaths: 
        maxDiscoveryDepth: 
        sitemap: 
        limit: 
        allowExternalLinks: 
        allowSubdomains: 
        crawlEntireDomain: 
        delay: 
        maxConcurrency: 
        webhook: 
        deduplicateSimilarURLs: 
        ignoreQueryParameters: 
        scrapeOptions: 

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
    if prompt is not None:
        mcp_args["prompt"] = prompt
    if excludePaths is not None:
        mcp_args["excludePaths"] = excludePaths
    if includePaths is not None:
        mcp_args["includePaths"] = includePaths
    if maxDiscoveryDepth is not None:
        mcp_args["maxDiscoveryDepth"] = maxDiscoveryDepth
    if sitemap is not None:
        mcp_args["sitemap"] = sitemap
    if limit is not None:
        mcp_args["limit"] = limit
    if allowExternalLinks is not None:
        mcp_args["allowExternalLinks"] = allowExternalLinks
    if allowSubdomains is not None:
        mcp_args["allowSubdomains"] = allowSubdomains
    if crawlEntireDomain is not None:
        mcp_args["crawlEntireDomain"] = crawlEntireDomain
    if delay is not None:
        mcp_args["delay"] = delay
    if maxConcurrency is not None:
        mcp_args["maxConcurrency"] = maxConcurrency
    if webhook is not None:
        mcp_args["webhook"] = webhook
    if deduplicateSimilarURLs is not None:
        mcp_args["deduplicateSimilarURLs"] = deduplicateSimilarURLs
    if ignoreQueryParameters is not None:
        mcp_args["ignoreQueryParameters"] = ignoreQueryParameters
    if scrapeOptions is not None:
        mcp_args["scrapeOptions"] = scrapeOptions

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_crawl",
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


def firecrawl_check_crawl_status(id: str | Any = None, **kwargs) -> Any:
    """
Check the status of a crawl job.

**Usage Example:**
```json
{
  "name": "firecrawl_check_crawl_status",
  "arguments": {
    "id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```
**Returns:** Status and progress of the crawl job, including results if available.


    Args:
        id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_check_crawl_status",
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


def firecrawl_extract(urls: list[str] | Any = None, prompt: str | None = None, schema: dict[str, Any] | None = None, allowExternalLinks: bool | None = None, enableWebSearch: bool | None = None, includeSubdomains: bool | None = None, **kwargs) -> Any:
    """
Extract structured information from web pages using LLM capabilities. Supports both cloud AI and self-hosted LLM extraction.

**Best for:** Extracting specific structured data like prices, names, details from web pages.
**Not recommended for:** When you need the full content of a page (use scrape); when you're not looking for specific structured data.
**Arguments:**
- urls: Array of URLs to extract information from
- prompt: Custom prompt for the LLM extraction
- schema: JSON schema for structured data extraction
- allowExternalLinks: Allow extraction from external links
- enableWebSearch: Enable web search for additional context
- includeSubdomains: Include subdomains in extraction
**Prompt Example:** "Extract the product name, price, and description from these product pages."
**Usage Example:**
```json
{
  "name": "firecrawl_extract",
  "arguments": {
    "urls": ["https://example.com/page1", "https://example.com/page2"],
    "prompt": "Extract product information including name, price, and description",
    "schema": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "price": { "type": "number" },
        "description": { "type": "string" }
      },
      "required": ["name", "price"]
    },
    "allowExternalLinks": false,
    "enableWebSearch": false,
    "includeSubdomains": false
  }
}
```
**Returns:** Extracted structured data as defined by your schema.


    Args:
        urls: 
        prompt: 
        schema: 
        allowExternalLinks: 
        enableWebSearch: 
        includeSubdomains: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if urls is not None:
        mcp_args["urls"] = urls
    if prompt is not None:
        mcp_args["prompt"] = prompt
    if schema is not None:
        mcp_args["schema"] = schema
    if allowExternalLinks is not None:
        mcp_args["allowExternalLinks"] = allowExternalLinks
    if enableWebSearch is not None:
        mcp_args["enableWebSearch"] = enableWebSearch
    if includeSubdomains is not None:
        mcp_args["includeSubdomains"] = includeSubdomains

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_extract",
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


def firecrawl_agent(prompt: str | Any = None, urls: list[str] | None = None, schema: dict[str, Any] | None = None, **kwargs) -> Any:
    """
Autonomous web research agent. This is a separate AI agent layer that independently browses the internet, searches for information, navigates through pages, and extracts structured data based on your query. You describe what you need, and the agent figures out where to find it.

**How it works:** The agent performs web searches, follows links, reads pages, and gathers data autonomously. This runs **asynchronously** - it returns a job ID immediately, and you poll `firecrawl_agent_status` to check when complete and retrieve results.

**IMPORTANT - Async workflow with patient polling:**
1. Call `firecrawl_agent` with your prompt/schema → returns job ID immediately
2. Poll `firecrawl_agent_status` with the job ID to check progress
3. **Keep polling for at least 2-3 minutes** - agent research typically takes 1-5 minutes for complex queries
4. Poll every 15-30 seconds until status is "completed" or "failed"
5. Do NOT give up after just a few polling attempts - the agent needs time to research

**Expected wait times:**
- Simple queries with provided URLs: 30 seconds - 1 minute
- Complex research across multiple sites: 2-5 minutes
- Deep research tasks: 5+ minutes

**Best for:** Complex research tasks where you don't know the exact URLs; multi-source data gathering; finding information scattered across the web; extracting data from JavaScript-heavy SPAs that fail with regular scrape.
**Not recommended for:** Simple single-page scraping where you know the URL (use scrape with JSON format instead - faster and cheaper).

**Arguments:**
- prompt: Natural language description of the data you want (required, max 10,000 characters)
- urls: Optional array of URLs to focus the agent on specific pages
- schema: Optional JSON schema for structured output

**Prompt Example:** "Find the founders of Firecrawl and their backgrounds"
**Usage Example (start agent, then poll patiently for results):**
```json
{
  "name": "firecrawl_agent",
  "arguments": {
    "prompt": "Find the top 5 AI startups founded in 2024 and their funding amounts",
    "schema": {
      "type": "object",
      "properties": {
        "startups": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "name": { "type": "string" },
              "funding": { "type": "string" },
              "founded": { "type": "string" }
            }
          }
        }
      }
    }
  }
}
```
Then poll with `firecrawl_agent_status` every 15-30 seconds for at least 2-3 minutes.

**Usage Example (with URLs - agent focuses on specific pages):**
```json
{
  "name": "firecrawl_agent",
  "arguments": {
    "urls": ["https://docs.firecrawl.dev", "https://firecrawl.dev/pricing"],
    "prompt": "Compare the features and pricing information from these pages"
  }
}
```
**Returns:** Job ID for status checking. Use `firecrawl_agent_status` to poll for results.


    Args:
        prompt: 
        urls: 
        schema: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if prompt is not None:
        mcp_args["prompt"] = prompt
    if urls is not None:
        mcp_args["urls"] = urls
    if schema is not None:
        mcp_args["schema"] = schema

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_agent",
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


def firecrawl_agent_status(id: str | Any = None, **kwargs) -> Any:
    """
Check the status of an agent job and retrieve results when complete. Use this to poll for results after starting an agent with `firecrawl_agent`.

**IMPORTANT - Be patient with polling:**
- Poll every 15-30 seconds
- **Keep polling for at least 2-3 minutes** before considering the request failed
- Complex research can take 5+ minutes - do not give up early
- Only stop polling when status is "completed" or "failed"

**Usage Example:**
```json
{
  "name": "firecrawl_agent_status",
  "arguments": {
    "id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```
**Possible statuses:**
- processing: Agent is still researching - keep polling, do not give up
- completed: Research finished - response includes the extracted data
- failed: An error occurred (only stop polling on this status)

**Returns:** Status, progress, and results (if completed) of the agent job.


    Args:
        id: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if id is not None:
        mcp_args["id"] = id

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_agent_status",
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


def firecrawl_browser_create(ttl: float | None = None, activityTtl: float | None = None, streamWebView: bool | None = None, **kwargs) -> Any:
    """
Create a persistent browser session for code execution via CDP (Chrome DevTools Protocol).

**Best for:** Running code (Python/JS) that interacts with a live browser page, multi-step browser automation, persistent sessions that survive across multiple tool calls.
**Not recommended for:** Simple page scraping (use firecrawl_scrape instead).

**Arguments:**
- ttl: Total session lifetime in seconds (30-3600, optional)
- activityTtl: Idle timeout in seconds (10-3600, optional)
- streamWebView: Whether to enable live view streaming (optional)

**Usage Example:**
```json
{
  "name": "firecrawl_browser_create",
  "arguments": {}
}
```
**Returns:** Session ID, CDP URL, and live view URL.


    Args:
        ttl: 
        activityTtl: 
        streamWebView: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if ttl is not None:
        mcp_args["ttl"] = ttl
    if activityTtl is not None:
        mcp_args["activityTtl"] = activityTtl
    if streamWebView is not None:
        mcp_args["streamWebView"] = streamWebView

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_browser_create",
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


def firecrawl_browser_execute(sessionId: str | Any = None, code: str | Any = None, language: str | None = None, **kwargs) -> Any:
    """
Execute code in a browser session. Supports agent-browser commands (bash), Python, or JavaScript.

**Best for:** Browser automation, navigating pages, clicking elements, extracting data, multi-step browser workflows.
**Requires:** An active browser session (create one with firecrawl_browser_create first).

**Arguments:**
- sessionId: The browser session ID (required)
- code: The code to execute (required)
- language: "bash", "python", or "node" (optional, defaults to "bash")

**Recommended: Use bash with agent-browser commands** (pre-installed in every sandbox):
```json
{
  "name": "firecrawl_browser_execute",
  "arguments": {
    "sessionId": "session-id-here",
    "code": "agent-browser open https://example.com",
    "language": "bash"
  }
}
```

**Common agent-browser commands:**
- `agent-browser open <url>` — Navigate to URL
- `agent-browser snapshot` — Get accessibility tree with clickable refs (for AI)
- `agent-browser snapshot -i -c` — Interactive elements only, compact
- `agent-browser click @e5` — Click element by ref from snapshot
- `agent-browser type @e3 "text"` — Type into element
- `agent-browser fill @e3 "text"` — Clear and fill element
- `agent-browser get text @e1` — Get text content
- `agent-browser get title` — Get page title
- `agent-browser get url` — Get current URL
- `agent-browser screenshot [path]` — Take screenshot
- `agent-browser scroll down` — Scroll page
- `agent-browser wait 2000` — Wait 2 seconds
- `agent-browser --help` — Full command reference

**For Playwright scripting, use Python** (has proper async/await support):
```json
{
  "name": "firecrawl_browser_execute",
  "arguments": {
    "sessionId": "session-id-here",
    "code": "await page.goto('https://example.com')\ntitle = await page.title()\nprint(title)",
    "language": "python"
  }
}
```

**Note:** Prefer bash (agent-browser) or Python.
**Returns:** Execution result including stdout, stderr, and exit code.


    Args:
        sessionId: 
        code: 
        language: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if sessionId is not None:
        mcp_args["sessionId"] = sessionId
    if code is not None:
        mcp_args["code"] = code
    if language is not None:
        mcp_args["language"] = language

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_browser_execute",
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


def firecrawl_browser_delete(sessionId: str | Any = None, **kwargs) -> Any:
    """
Destroy a browser session.

**Usage Example:**
```json
{
  "name": "firecrawl_browser_delete",
  "arguments": {
    "sessionId": "session-id-here"
  }
}
```
**Returns:** Success confirmation.


    Args:
        sessionId: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if sessionId is not None:
        mcp_args["sessionId"] = sessionId

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_browser_delete",
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


def firecrawl_browser_list(status: str | None = None, **kwargs) -> Any:
    """
List browser sessions, optionally filtered by status.

**Usage Example:**
```json
{
  "name": "firecrawl_browser_list",
  "arguments": {
    "status": "active"
  }
}
```
**Returns:** Array of browser sessions.


    Args:
        status: 

    Returns:
        Tool execution result
    """
    from graph_rlm.backend.src.mcp_integration.runtime import call_mcp_tool
    from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result
    import asyncio

    # Build parameters dict
    mcp_args = {}
    if status is not None:
        mcp_args["status"] = status

    async def _async_call():
        return await call_mcp_tool(
            server_name="mcp-server-firecrawl",
            tool_name="firecrawl_browser_list",
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
    return ['firecrawl_scrape', 'firecrawl_map', 'firecrawl_search', 'firecrawl_crawl', 'firecrawl_check_crawl_status', 'firecrawl_extract', 'firecrawl_agent', 'firecrawl_agent_status', 'firecrawl_browser_create', 'firecrawl_browser_execute', 'firecrawl_browser_delete', 'firecrawl_browser_list']
