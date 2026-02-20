"""
LLM Service Layer for Graph-RLM.
Provides a unified interface for multiple LLM providers and embedding models.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx

# Pydantic AI Integration
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

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
    DEFAULT_READ_TIMEOUT = 120.0
    DEFAULT_WRITE_TIMEOUT = 1200.0
    DEFAULT_POOL_TIMEOUT = 5.0

    def __init__(self):
        logger.info("LLMService initialized.")
        # Store clients and locks per event loop to prevent loop-match deadlocks
        # {loop: (client, lock)}
        self._loop_resources: Dict[
            asyncio.AbstractEventLoop, tuple[httpx.AsyncClient, asyncio.Lock]
        ] = {}
        self._timeout = httpx.Timeout(
            connect=self.DEFAULT_CONNECT_TIMEOUT,
            read=self.DEFAULT_READ_TIMEOUT,
            write=self.DEFAULT_WRITE_TIMEOUT,
            pool=self.DEFAULT_POOL_TIMEOUT,
        )

    async def _get_client_and_lock(self) -> tuple[httpx.AsyncClient, asyncio.Lock]:
        """
        Retrieves or initializes the httpx.AsyncClient and asyncio.Lock for the current loop.
        """
        loop = asyncio.get_running_loop()
        if loop not in self._loop_resources:
            # Atomic creation within the current loop
            self._loop_resources[loop] = (
                httpx.AsyncClient(timeout=self._timeout),
                asyncio.Lock(),
            )
        return self._loop_resources[loop]

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Retrieves the client for the current loop, initializing if necessary.
        """
        client, _ = await self._get_client_and_lock()
        return client

    async def aclose(self):
        """
        Gracefully closes all persistent httpx clients across all loops.
        """
        for client, _ in self._loop_resources.values():
            await client.aclose()
        self._loop_resources.clear()

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
        if self.provider == "ollama" and path == "embeddings":
            return f"{base}/api/embeddings"
        return f"{base}/{path}"

    def _format_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        model_override: Optional[str] = None,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
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
            if max_tokens:
                request["options"]["num_predict"] = max_tokens
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
            if max_tokens:
                body["max_tokens"] = max_tokens
            return body

    async def generate(
        self,
        prompt: Any,
        system: Optional[str] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        on_usage: Optional[Any] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Generates a response from the LLM.

        Args:
            prompt: Input string or list of message dictionaries.
            system: Optional system prompt.
            stream: Whether to return a stream of chunks.
            stop: Optional stop sequences.
            on_usage: Callback for usage statistics.
            model: Optional model override.

        Returns:
            The generated text or an async iterator if streaming is enabled.
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
            return self._generate_stream(messages, stop=stop, on_usage=on_usage)
        else:
            return await self._generate_async(
                messages,
                on_usage=on_usage,
                model_override=model,
                stop=stop,
                max_tokens=max_tokens,
            )

    async def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        on_usage: Optional[Any] = None,
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
                    logger.error(
                        "LLM Stream HTTP error for endpoint %s: %s",
                        endpoint,
                        e,
                        exc_info=True,
                    )
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

                            # [Usage Extraction] Check for usage in chunk (OpenRouter/OpenAI standard)
                            if on_usage and "usage" in chunk:
                                try:
                                    on_usage(chunk["usage"])
                                except (
                                    AttributeError,
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                ) as e:
                                    logger.warning(
                                        "Failed to process usage callback: %s", e
                                    )

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
                            json.JSONDecodeError,
                            KeyError,
                            IndexError,
                            TypeError,
                        ) as e:
                            logger.warning("Stream Parse Error: %s", e)
        except httpx.TimeoutException as e:
            logger.error(
                "LLM Stream Timeout for endpoint %s: %s", endpoint, e, exc_info=True
            )
            yield f"Error: Timeout occurred: {str(e)}"
        except httpx.RequestError as e:
            logger.error(
                "LLM Stream Request Error for endpoint %s: %s",
                endpoint,
                e,
                exc_info=True,
            )
            yield f"Error: Request failed: {str(e)}"

    def _handle_async_http_error(
        self, e: httpx.HTTPStatusError, body: Dict[str, Any], last_msg: str
    ) -> str:
        logger.error("LLM Async HTTP error: %s", e, exc_info=True)
        try:
            # Log the full response body for debugging
            err_body = e.response.text
            logger.error("Provider Response Body: %s", err_body)
        except (AttributeError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to extract error body: %s", exc)

        try:
            # Log snippet of request body to check for bloat/format issues
            messages_count = len(body.get("messages", []))
            model_name = body.get("model", "unknown")
            req_snippet = json.dumps(
                {
                    "model": model_name,
                    "messages_count": messages_count,
                    "sample_prompt": last_msg[:500],
                }
            )
            logger.error("Request Metadata: %s", req_snippet)
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Failed to log request metadata: %s", exc)

        return f"Error: {str(e)}"

    def _validate_provider_error(self, data: Dict[str, Any]) -> Optional[str]:
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

            # [Resilience] Handle Malformed Function Calls
            if "MALFORMED_FUNCTION_CALL" in error_msg or "native_finish_reason" in str(
                error_obj
            ):
                return "SYSTEM ERROR: The model attempted a native function call which is malformed. Please use Python code blocks only."

            return f"Error: {full_err}"
        return None

    def _validate_choice_error(self, choice: Dict[str, Any]) -> Optional[str]:
        finish_reason = choice.get("finish_reason")
        native_finish_reason = choice.get("native_finish_reason")

        if (
            finish_reason == "error"
            or native_finish_reason == "MALFORMED_FUNCTION_CALL"
        ):
            err_msg = (
                "SYSTEM ERROR: The model attempted a native function call which is malformed. "
                "Please use Python code blocks only instead of attempting native tools."
            )
            logger.error(
                "Choice-Level Error Detected: %s (Native: %s)",
                finish_reason,
                native_finish_reason,
            )
            trace_action(
                "LLM",
                "CHOICE_ERROR",
                result=f"{finish_reason}/{native_finish_reason}",
                tag="ERROR",
            )
            return err_msg
        return None

    def _polyfill_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> str:
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
                kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                codes.append(
                    f"# Model triggered native tool: {name}\nval = {name}({kwargs_str})"
                )
            except (
                json.JSONDecodeError,
                KeyError,
                AttributeError,
                TypeError,
            ) as e:
                logger.error(
                    "Failed to polyfill tool call (data/format error): %s",
                    e,
                )
        return "\n".join(codes)

    async def _generate_async(
        self,
        messages: List[Dict[str, str]],
        on_usage: Optional[Any] = None,
        model_override: Optional[str] = None,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(
            messages,
            stream=False,
            model_override=model_override,
            stop=stop,
            max_tokens=max_tokens,
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
                return self._handle_async_http_error(e, body, last_msg)

            data = response.json()

            # [Usage Extraction]
            if on_usage and "usage" in data:
                try:
                    on_usage(data["usage"])
                except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                    logger.warning("Failed to process usage callback in async: %s", e)

            if self.provider == "ollama":
                return data.get("message", {}).get("content", "")

            # Robust Error Parsing (OpenRouter/OpenAI)
            if error_msg := self._validate_provider_error(data):
                return error_msg

            choices = data.get("choices", [])
            if not choices:
                return (
                    ""
                    if self.provider == "ollama"
                    else "Error: Empty choices returned by provider."
                )

            choice = choices[0]
            message = choice.get("message", {})
            res = message.get("content", "")
            tool_calls = message.get("tool_calls", [])

            # [Resilience] Handle Choice-Level Errors
            if choice_error := self._validate_choice_error(choice):
                return choice_error

            if tool_calls:
                tool_code = self._polyfill_tool_calls(tool_calls)
                if tool_code:
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

            return res
        except httpx.TimeoutException as e:
            logger.error(
                "LLM Async Timeout for endpoint %s: %s", endpoint, e, exc_info=True
            )
            return f"Error: Timeout occurred: {str(e)}"
        except httpx.RequestError as e:
            logger.error(
                "LLM Async Request Error for endpoint %s: %s",
                endpoint,
                e,
                exc_info=True,
            )
            return f"Error: Request failed: {str(e)}"

    async def generate_structured(
        self,
        prompt: str,
        output_type: Any,
        system: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Any:
        """
        Generate a structured, type-safe response using Pydantic AI.
        """
        target_model = model or self.config.get("model")
        if not target_model:
            raise ValueError("No model specified for structured generation.")

        # Ensure model_name is a string for Pydantic AI
        model_name = str(target_model)

        # Initialize the appropriate Pydantic AI model based on provider
        pydantic_model = None
        if self.provider == "openrouter":
            provider = OpenRouterProvider(
                api_key=self.config.get("api_key"),
                app_url="https://github.com/angrysky56/graph-rlm",
                app_title="Graph-RLM",
            )
            pydantic_model = OpenAIChatModel(model_name, provider=provider)
        elif self.provider == "openai":
            provider = OpenAIProvider(
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key"),
            )
            pydantic_model = OpenAIChatModel(model_name, provider=provider)
        else:
            # Fallback to generic OpenAIProvider for Ollama/LMStudio if they are OpenAI-compatible
            logger.warning(
                "Structured generation requested for provider %s. Falling back to generic OpenAIProvider.",
                self.provider,
            )
            provider = OpenAIProvider(
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key"),
            )
            pydantic_model = OpenAIChatModel(model_name, provider=provider)

        agent = PydanticAgent(
            pydantic_model, output_type=output_type, system_prompt=system or ""
        )

        trace_action(
            "LLM_STRUCTURED",
            f"GENERATE ({self.provider}/{target_model})",
            result=f"TYPE: {output_type.__name__}",
            tag="LLM",
        )

        try:
            result = await agent.run(prompt)
            return result.output
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.error("Pydantic AI Structured Generation Error: %s", e)
            trace_action("LLM_STRUCTURED", "ERROR", result=str(e), tag="ERROR")
            raise

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

        client, lock = await self._get_client_and_lock()
        async with lock:
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
            except (
                httpx.HTTPError,
                httpx.RequestError,
                json.JSONDecodeError,
                KeyError,
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as e:
                logger.error(
                    "Embedding Error for model %s: %s", model, e, exc_info=True
                )
                return []

    def compute_cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        import math

        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(a * a for a in v2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

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
        except (httpx.HTTPError, httpx.RequestError, AttributeError, RuntimeError) as e:
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
                            httpx.HTTPError,
                            httpx.RequestError,
                            json.JSONDecodeError,
                            KeyError,
                        ) as e:
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

        except (
            httpx.HTTPError,
            httpx.RequestError,
            json.JSONDecodeError,
            KeyError,
            AttributeError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            logger.error("List Models Error: %s", e)
            return []


llm = LLMService()
