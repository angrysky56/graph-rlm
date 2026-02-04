import asyncio
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

    DEFAULT_CONNECT_TIMEOUT = 10.0
    DEFAULT_READ_TIMEOUT = 1200.0
    DEFAULT_WRITE_TIMEOUT = 600.0
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
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def provider(self) -> str:
        return settings.LLM_PROVIDER

    @property
    def config(self) -> dict:
        return settings.get_llm_config()

    def _get_headers(self) -> Dict[str, str]:
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
        base = self.config.get("base_url", "").rstrip("/")
        if self.provider == "ollama" and path == "chat/completions":
            return f"{base}/api/chat"
        elif self.provider == "ollama" and path == "embeddings":
            return f"{base}/api/embeddings"
        return f"{base}/{path}"

    def _format_request(
        self, messages: List[Dict[str, str]], stream: bool = False
    ) -> Dict[str, Any]:
        """Format request body based on provider quirks."""
        model = self.config.get("model")

        if self.provider == "ollama":
            request = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": 0.7},  # Default
            }
            return request
        else:
            # Standard OpenAI format
            return {"model": model, "messages": messages, "stream": stream}

    async def generate(
        self,
        prompt: Any,
        system: Optional[str] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        on_usage: Optional[Any] = None,
    ) -> Any:
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
            return self._generate_stream(messages)
        else:
            return await self._generate_async(messages, on_usage=on_usage)

    async def _generate_stream(self, messages: List[Dict[str, str]]):
        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(messages, stream=True)

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
                    logger.error(f"LLM Stream HTTP error: {e}")
                    yield f"Error: {str(e)}"
                    return
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    logger.debug(f"Stream Line: {line}")
                    if line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            import json

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
                        except Exception as e:
                            logger.warning(f"Stream Parse Error: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"LLM Stream Timeout: {e}")
            yield f"Error: Timeout occurred: {str(e)}"
        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            yield f"Error: {str(e)}"

    async def _generate_async(
        self, messages: List[Dict[str, str]], on_usage: Optional[Any] = None
    ) -> str:
        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(messages, stream=False)

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
                logger.error(f"LLM Async HTTP error: {e}")
                return f"Error: {str(e)}"
            data = response.json()

            if self.provider == "ollama":
                return data.get("message", {}).get("content", "")
            else:
                # Robust Error Parsing (OpenRouter/OpenAI)
                if "error" in data:
                    import json

                    error_obj = data["error"]
                    error_msg = error_obj.get("message", "Unknown error")
                    error_code = error_obj.get("code", "unknown")
                    error_meta = error_obj.get("metadata", {})
                    full_err = f"Provider Error (Code: {error_code}): {error_msg}"
                    if error_meta:
                        full_err += f"\nMetadata: {json.dumps(error_meta)}"
                    logger.error(f"LLM Error Object Detected: {full_err}")
                    trace_action("LLM", "ERROR", result=full_err, tag="ERROR")
                    return f"Error: {full_err}"
                message = data.get("choices", [{}])[0].get("message", {})
                res = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    # Polyfill: Convert tool call to Python code for Agent
                    # We assume the agent can handle standard Python tool invocations
                    import json

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
                        except Exception as e:
                            logger.error(f"Failed to polyfill tool call: {e}")

                    if codes:
                        # Append to existing content if any (reasoning might be there)
                        tool_code = "\n".join(codes)
                        if res:
                            res += f"\n\n{tool_code}"
                        else:
                            res = tool_code

                        trace_action("LLM", "TOOL_POLYFILL", result=res, tag="LLM")

                if not res:
                    import json

                    raw_data = json.dumps(data)
                    logger.warning(
                        f"Empty response from provider. Full Data: {raw_data[:1000]}"
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
                    except Exception as e:
                        logger.warning(f"Failed to execute usage callback: {e}")

                return res
        except httpx.TimeoutException as e:
            logger.error(f"LLM Async Timeout: {e}")
            return f"Error: Timeout occurred: {str(e)}"
        except Exception as e:
            logger.error(f"LLM Async Error: {e}")
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
                logger.error(f"Embedding HTTP error: {e}")
                return []
            data = response.json()
            if self.provider == "ollama":
                return data.get("embedding", [])
            else:
                return data.get("data", [{}])[0].get("embedding", [])
        except httpx.TimeoutException as e:
            logger.error(f"Embedding Timeout: {e}")
            return []
        except Exception as e:
            logger.error(f"Embedding Error: {e}")
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
                logger.info(f"Unloaded model {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models available from the provider."""
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
                                "context_length": 1000000,
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
                            embed_url = f"{base}/embeddings/models"  # Usually https://openrouter.ai/api/v1/embeddings/models
                            with httpx.Client(timeout=5.0) as client:
                                resp_emb = client.get(embed_url, headers=headers)
                                if resp_emb.status_code == 200:
                                    emb_data = resp_emb.json().get("data", [])
                                    # Tag them explicitly so logic downstream knows they are embeddings
                                    for em in emb_data:
                                        em["_is_embedding_endpoint"] = True
                                    raw_list.extend(emb_data)
                        except Exception as e:
                            logger.warning(
                                f"Failed to fetch separate embedding models: {e}"
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

        except Exception as e:
            logger.error(f"List Models Error: {e}")
            return []


llm = LLMService()
