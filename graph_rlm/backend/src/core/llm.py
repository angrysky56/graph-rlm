"""
LLM Service Layer for Graph-RLM.
Provides a unified interface for multiple LLM providers and embedding models.
"""

import asyncio
import json
from contextlib import suppress
from typing import Any, Dict, List, Optional

import httpx

from .config import settings
from .logger import get_logger
from .trace import trace_action

logger = get_logger("graph_rlm.llm")


class LLMService:
    """
    Unified LLM client supporting OpenRouter, Ollama, and OpenAI-compatible endpoints.
    Uses a persistent httpx.AsyncClient for connection pooling and robust timeout management.
    """

    DEFAULT_CONNECT_TIMEOUT = 5.0
    DEFAULT_READ_TIMEOUT = 60.0
    DEFAULT_WRITE_TIMEOUT = 60.0
    DEFAULT_POOL_TIMEOUT = 5.0

    def __init__(self):
        logger.info("LLMService initialized.")
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = httpx.Timeout(
            connect=self.DEFAULT_CONNECT_TIMEOUT,
            read=self.DEFAULT_READ_TIMEOUT,
            write=self.DEFAULT_WRITE_TIMEOUT,
            pool=self.DEFAULT_POOL_TIMEOUT,
        )
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Retrieves or initializes the shared httpx.AsyncClient.
        """
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self):
        """
        Gracefully closes the persistent httpx client.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def refresh(self):
        """
        Refreshes the LLM service configuration.
        Closes the existing client so the next request initializes a new one with updated settings.
        """
        logger.info("Refreshing LLM Service configuration...")
        await self.aclose()

    @property
    def provider(self) -> str:
        """
        Returns the configured LLM provider.
        """
        return settings.LLM_PROVIDER

    @property
    def config(self) -> dict:
        """
        Returns the LLM configuration from settings.
        """
        return settings.get_llm_config()

    def _get_headers(self) -> Dict[str, str]:
        """
        Constructs the HTTP headers for LLM API requests.
        """
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("api_key")
        if api_key and api_key != "lm-studio":
            headers["Authorization"] = f"Bearer {api_key}"

        # OpenRouter specific headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/angrysky56/graph-rlm"
            headers["X-Title"] = settings.PROJECT_NAME

        return headers

    def _get_endpoint(self, path: str = "chat/completions") -> str:
        """
        Resolves the full API endpoint URL for the current provider and path.
        """
        base = self.config.get("base_url", "").rstrip("/")
        if self.provider == "ollama" and path == "chat/completions":
            return f"{base}/api/chat"
        elif self.provider == "ollama" and path == "embeddings":
            return f"{base}/api/embeddings"
        return f"{base}/{path}"

    def _format_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        model_override: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format request body based on provider quirks."""
        model = model_override or self.config.get("model")

        if self.provider == "ollama":
            request = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": 0.7},  # Default
            }
            if stop:
                request["stop"] = stop
            return request
        else:
            # Standard OpenAI format
            body: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": stream,
            }
            if stop:
                body["stop"] = stop
            return body

    async def generate(
        self,
        prompt: Any,
        system: Optional[str] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        on_usage: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Unified generation interface. Supports both string prompts and message lists.
        Handles both streaming and non-streaming modes.
        """
        # Note: prompt can be str or List[Dict] (messages)
        # We need to handle both since agent passes messages directly sometimes

        messages = []
        if isinstance(prompt, list):
            messages = prompt
        else:
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        if stream:
            return self._generate_stream(messages, stop=stop)
        else:
            return await self._generate_async(
                messages, on_usage=on_usage, model_override=model, stop=stop
            )

    async def _generate_stream(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None
    ):
        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(messages, stream=True, stop=stop)

        # Trace Outgoing
        last_msg = messages[-1]["content"] if messages else "EMPTY"
        trace_action(
            "LLM",
            f"STREAM ({self.provider}/{body['model']})",
            result=f"PROMPT: {last_msg[:200]}...",
            tag="LLM",
        )

        client = await self._get_client()
        try:
            async with client.stream(
                "POST", endpoint, headers=headers, json=body
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error("LLM Stream HTTP error: %s", e)
                    yield f"Error: {str(e)}"
                    return
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    logger.debug("Stream Line: %s", line)
                    if line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            content = ""
                            if self.provider == "ollama":
                                content = chunk.get("message", {}).get("content", "")
                            else:
                                choices = chunk.get("choices", [])
                                if choices:
                                    content = (
                                        choices[0].get("delta", {}).get("content", "")
                                    )
                            if content:
                                yield content
                        except (
                            Exception
                        ) as e:  # pylint: disable=broad-except # noqa: BLE001
                            logger.warning("Stream Parse Error: %s", e)
        except httpx.TimeoutException as e:
            logger.error("LLM Stream Timeout: %s", e)
            yield f"Error: Timeout occurred: {str(e)}"
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("LLM Stream Error: %s", e)
            yield f"Error: {str(e)}"

    async def _generate_async(
        self,
        messages: List[Dict[str, str]],
        on_usage: Optional[Any] = None,
        model_override: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(
            messages, stream=False, model_override=model_override, stop=stop
        )

        # Trace Outgoing
        last_msg = messages[-1]["content"] if messages else "EMPTY"
        trace_action(
            "LLM",
            f"ASYNC ({self.provider}/{body['model']})",
            result=f"PROMPT: {last_msg[:200]}...",
            tag="LLM",
        )

        client = await self._get_client()
        try:
            response = await client.post(endpoint, headers=headers, json=body)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("LLM Async HTTP error: %s", e)
                with suppress(Exception):
                    # Log the full response body for debugging
                    err_body = e.response.text
                    logger.error("Provider Response Body: %s", err_body)

                with suppress(Exception):
                    # Log snippet of request body to check for bloat/format issues
                    req_snippet = json.dumps(body)[:2000]
                    logger.error(
                        "Request Body Snippet (First 2000 chars): %s", req_snippet
                    )

                return f"Error: {str(e)}"
            data = response.json()

            if self.provider == "ollama":
                return data.get("message", {}).get("content", "")
            else:
                # Robust Error Parsing (OpenRouter/OpenAI)
                if "error" in data:
                    error_obj = data["error"]
                    # If error_obj is just a string, wrap it for safety
                    if not isinstance(error_obj, dict):
                        error_obj = {"message": str(error_obj)}
                    error_msg = error_obj.get("message", "Unknown error")
                    error_code = error_obj.get("code", "unknown")
                    error_meta = error_obj.get("metadata", {})
                    full_err = f"Provider Error (Code: {error_code}): {error_msg}"
                    if error_meta:
                        full_err += f"\nMetadata: {json.dumps(error_meta)}"
                    logger.error("LLM Error Object Detected: %s", full_err)
                    trace_action("LLM", "ERROR", result=full_err, tag="ERROR")
                    return f"Error: {full_err}"
                message = data.get("choices", [{}])[0].get("message", {})
                res = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    # Polyfill: Convert tool call to Python code for Agent
                    # We assume the agent can handle standard Python tool invocations
                    codes = []
                    for tc in tool_calls:
                        try:
                            func = tc.get("function", {})
                            name = func.get("name", "unknown_tool")
                            args_str = func.get("arguments", "{}")
                            # Normalize args (sometimes it's a dict, sometimes string)
                            if isinstance(args_str, str):
                                args = json.loads(args_str)
                            else:
                                args = args_str

                            # Construct kwargs string
                            kwargs_str = ", ".join(
                                f"{k}={repr(v)}" for k, v in args.items()
                            )
                            codes.append(
                                f"# Model triggered native tool: {name}\nval = {name}({kwargs_str})"
                            )
                        except (
                            Exception
                        ) as e:  # pylint: disable=broad-except # noqa: BLE001
                            logger.error("Failed to polyfill tool call: %s", e)

                    if codes:
                        # Append to existing content if any (reasoning might be there)
                        tool_code = "\n".join(codes)
                        if res:
                            res += f"\n\n{tool_code}"
                        else:
                            res = tool_code

                        trace_action("LLM", "TOOL_POLYFILL", result=res, tag="LLM")

                if not res:
                    raw_data = json.dumps(data)
                    logger.warning(
                        "Empty response from provider. Full Data: %s", raw_data[:1000]
                    )
                    trace_action(
                        "LLM",
                        "WARNING",
                        result=f"Empty response from provider. Full Data: {raw_data[:1000]}",
                        tag="ERROR",
                    )
                else:
                    trace_action("LLM", "RESPONSE", result=res, tag="LLM")

                if on_usage and "usage" in data:
                    try:
                        on_usage(data["usage"])
                    except (
                        Exception
                    ) as e:  # pylint: disable=broad-except # noqa: BLE001
                        logger.warning("Failed to execute usage callback: %s", e)

                return res
        except httpx.TimeoutException as e:
            logger.error("LLM Async Timeout: %s", e)
            return f"Error: Timeout occurred: {str(e)}"
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("LLM Async Error: %s", e)
            return f"Error: {str(e)}"

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings. Supports Ollama (nomic/llama3) and OpenAI/OpenRouter formats.
        """
        endpoint = (
            self._get_endpoint("embeddings")
            if self.provider == "ollama"
            else self._get_endpoint("embeddings")
        )
        headers = self._get_headers()

        # Determine embedding model
        model = self.config.get("embedding_model")
        if self.provider == "ollama":
            body = {"model": model, "prompt": text}
        else:
            # OpenAI format
            body = {"model": model, "input": text}

        trace_action("LLM", f"EMBEDDING ({model})", result=text, tag="LLM")

        client = await self._get_client()
        try:
            response = await client.post(endpoint, headers=headers, json=body)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("Embedding HTTP error: %s", e)
                return []
            data = response.json()
            if self.provider == "ollama":
                return data.get("embedding", [])
            else:
                return data.get("data", [{}])[0].get("embedding", [])
        except httpx.TimeoutException as e:
            logger.error("Embedding Timeout: %s", e)
            return []
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("Embedding Error: %s", e)
            return []

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory. Primarily for Ollama.
        """
        if self.provider != "ollama":
            return True  # No-op for cloud providers

        endpoint = self._get_endpoint("chat/completions")  # /api/chat
        # To unload, send empty prompt with keep_alive=0
        body = {"model": model_name, "keep_alive": 0}
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(endpoint, json=body)
                logger.info("Unloaded model %s", model_name)
                return True
        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("Failed to unload model: %s", e)
            return False

    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Dynamically fetches and categorizes models from the current or specified provider.
        """
        target_provider = provider or self.provider

        # Use simple override config or default
        if provider:
            # Create a temporary config view
            temp_config = settings.get_config_for_provider(provider)
            base = temp_config.get("base_url", "").rstrip("/")
            api_key = temp_config.get("api_key")

            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "lm-studio":
                headers["Authorization"] = f"Bearer {api_key}"
            if provider == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/angrysky56/graph-rlm"
                headers["X-Title"] = settings.PROJECT_NAME
        else:
            headers = self._get_headers()
            base = self.config.get("base_url", "").rstrip("/")

        try:
            url = ""
            if target_provider == "ollama":
                url = f"{base}/api/tags"
            else:
                url = f"{base}/models"

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                models = []
                if target_provider == "ollama":
                    for m in data.get("models", []):
                        name = m.get("name")

                        is_embedding = (
                            "embed" in name or "nomic" in name or "bert" in name
                        )

                        models.append(
                            {
                                "id": name,
                                "name": name,
                                "context_length": "",
                                "pricing": {"prompt": "0", "completion": "0"},
                                "supports_tools": "llama3" in name
                                or "mistral" in name
                                or "gemma" in name
                                or "qwen" in name,
                                "type": "embedding" if is_embedding else "chat",
                                "provider": "ollama",
                            }
                        )
                else:
                    # OpenRouter/OpenAI format
                    # 1. Fetch Chat Models
                    raw_list = data.get("data", [])

                    # 2. Fetch Embedding Models (OpenRouter Specific)
                    if target_provider == "openrouter":
                        try:
                            # Usually https://openrouter.ai/api/v1/embeddings/models
                            embed_url = f"{base}/embeddings/models"
                            with httpx.Client(timeout=5.0) as client:
                                resp_emb = client.get(embed_url, headers=headers)
                                if resp_emb.status_code == 200:
                                    emb_data = resp_emb.json().get("data", [])
                                    # Tag them explicitly for logic downstream
                                    for em in emb_data:
                                        em["_is_embedding_endpoint"] = True
                                    raw_list.extend(emb_data)
                        except (
                            Exception
                        ) as e:  # pylint: disable=broad-except # noqa: BLE001
                            logger.warning(
                                "Failed to fetch separate embedding models: %s", e
                            )

                    for m in raw_list:
                        m_id = m.get("id")
                        name = m.get("name") or m_id

                        # Heuristics for Embeddings (Robust)
                        # Check endpoint tag, ID pattern, or context length
                        is_embedding = (
                            m.get("_is_embedding_endpoint", False)
                            or "embed" in m_id.lower()
                            or "nomic" in m_id.lower()
                            or "text-embedding" in m_id.lower()
                        )

                        # Check tool support via 'supported_parameters' (OpenRouter specific)
                        supported_params = m.get("supported_parameters", [])
                        supports_tools = "tools" in supported_params

                        # If standard OpenAI, assume tools for gpt-4/3.5
                        if target_provider == "openai":
                            supports_tools = "gpt" in m_id

                        pricing = m.get("pricing", {})
                        ctx = m.get("context_length", 4096)

                        # Architecture (for tokenizer info etc)

                        models.append(
                            {
                                "id": m_id,
                                "name": name,
                                "context_length": ctx,
                                "pricing": {
                                    "prompt": pricing.get("prompt", "0"),
                                    "completion": pricing.get("completion", "0"),
                                },
                                "supports_tools": supports_tools,
                                "type": "embedding" if is_embedding else "chat",
                                "provider": (
                                    m_id.split("/")[0]
                                    if "/" in m_id
                                    else target_provider
                                ),
                            }
                        )

                return models

        except Exception as e:  # pylint: disable=broad-except # noqa: BLE001
            logger.error("List Models Error: %s", e)
            return []


llm = LLMService()
