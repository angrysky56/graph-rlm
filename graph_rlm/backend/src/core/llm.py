
import httpx
from typing import Optional, Dict, Any, List
from .config import settings
from .logger import get_logger

logger = get_logger("graph_rlm.llm")


class LLMService:
    """
    Unified LLM client supporting OpenRouter, Ollama, and OpenAI-compatible endpoints.
    """
    def __init__(self):
        logger.info(f"LLMService initialized.")

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

    def _format_request(self, messages: List[Dict[str, str]], stream: bool = False) -> Dict[str, Any]:
        """Format request body based on provider quirks."""
        model = self.config.get("model")

        if self.provider == "ollama":
            request = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": 0.7} # Default
            }
            return request
        else:
            # Standard OpenAI format
            return {
                "model": model,
                "messages": messages,
                "stream": stream
            }

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Synchronous generation (blocking).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        endpoint = self._get_endpoint("chat/completions")
        headers = self._get_headers()
        body = self._format_request(messages, stream=False)

        try:
            # High timeout for reasoning models
            with httpx.Client(timeout=120.0) as client:
                response = client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

                if self.provider == "ollama":
                    return data.get("message", {}).get("content", "")
                else:
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            return f"Error: {str(e)}"

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings. Supports Ollama (nomic/llama3) and OpenAI/OpenRouter formats.
        """
        endpoint = self._get_endpoint("embeddings") if self.provider == "ollama" else self._get_endpoint("embeddings")
        headers = self._get_headers()

        # Determine embedding model
        model = self.config.get("embedding_model")
        if self.provider == "ollama":
            # For Ollama, we might want a specific embedding model if the main one is generative
            # But simpler to just use main model if it supports it, or hardcode fallback like 'nomic-embed-text' if widely available
            # Just use config model for now
            body = {"model": model, "prompt": text}
        else:
            # OpenAI format
            body = {"model": model, "input": text}

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()

                if self.provider == "ollama":
                    return data.get("embedding", [])
                else:
                    return data.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            logger.error(f"Embedding Error: {e}")
            return []

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory. Primarily for Ollama.
        """
        if self.provider != "ollama":
            return True # No-op for cloud providers

        endpoint = self._get_endpoint("chat/completions") # /api/chat
        # To unload, send empty prompt with keep_alive=0
        body = {
            "model": model_name,
            "keep_alive": 0
        }
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
                        details = m.get("details", {})
                        family = details.get("family", "")

                        is_embedding = "embed" in name or "nomic" in name or "bert" in name

                        models.append({
                            "id": name,
                            "name": name,
                            "context_length": 8192,
                            "pricing": {"prompt": "0", "completion": "0"},
                            "supports_tools": "llama3" in name or "mistral" in name or "gemma" in name or "qwen" in name,
                            "type": "embedding" if is_embedding else "chat",
                            "provider": "ollama"
                        })
                else:
                    # OpenRouter/OpenAI format
                    # 1. Fetch Chat Models
                    raw_list = data.get("data", [])

                    # 2. Fetch Embedding Models (OpenRouter Specific)
                    if target_provider == "openrouter":
                        try:
                            embed_url = f"{base}/embeddings/models" # Usually https://openrouter.ai/api/v1/embeddings/models
                            with httpx.Client(timeout=5.0) as client:
                                resp_emb = client.get(embed_url, headers=headers)
                                if resp_emb.status_code == 200:
                                    emb_data = resp_emb.json().get("data", [])
                                    # Tag them explicitly so logic downstream knows they are embeddings
                                    for em in emb_data:
                                        em["_is_embedding_endpoint"] = True
                                    raw_list.extend(emb_data)
                        except Exception as e:
                            logger.warning(f"Failed to fetch separate embedding models: {e}")

                    for m in raw_list:
                        m_id = m.get("id")
                        name = m.get("name") or m_id

                        # Heuristics for Embeddings (Robust)
                        # Check endpoint tag, ID pattern, or context length
                        is_embedding = (
                            m.get("_is_embedding_endpoint", False) or
                            "embed" in m_id.lower() or
                            "nomic" in m_id.lower() or
                            "text-embedding" in m_id.lower()
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
                        arch = m.get("architecture", {})

                        models.append({
                            "id": m_id,
                            "name": name,
                            "context_length": ctx,
                            "pricing": {
                                "prompt": pricing.get("prompt", "0"),
                                "completion": pricing.get("completion", "0")
                            },
                            "supports_tools": supports_tools,
                            "type": "embedding" if is_embedding else "chat",
                            "provider": m_id.split("/")[0] if "/" in m_id else target_provider
                        })

                return models

        except Exception as e:
            logger.error(f"List Models Error: {e}")
            return []

llm = LLMService()
