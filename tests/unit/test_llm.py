"""Unit tests for src.core.llm module.
Tests for LLMService including provider configurations, request formatting,
and asynchronous/structured generation.
"""

import json
from unittest import mock

import httpx
import pytest
from pydantic import BaseModel

from graph_rlm.backend.src.core.llm import LLMService


class MockResponse:
    """Mock httpx Response."""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "Error", request=mock.Mock(), response=self
            )


@pytest.fixture
def llm_service():
    """Fixture for LLMService instance."""
    service = LLMService()
    return service


@pytest.mark.asyncio
class TestLLMServiceInternal:
    """Test LLMService internal helper methods."""

    async def test_get_headers_openrouter(self, llm_service):
        """Test headers for OpenRouter provider."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openrouter"
            mock_settings.get_llm_config.return_value = {
                "api_key": "test_key",
                "base_url": "https://openrouter.ai/api/v1",
            }
            headers = llm_service._get_headers()
            assert headers["Content-Type"] == "application/json"
            assert headers["Authorization"] == "Bearer test_key"
            assert "HTTP-Referer" in headers
            assert headers["X-Title"] == mock_settings.PROJECT_NAME

    async def test_get_headers_ollama(self, llm_service):
        """Test headers for Ollama provider (no auth)."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {
                "api_key": "",
                "base_url": "http://localhost:11434",
            }
            headers = llm_service._get_headers()
            assert "Authorization" not in headers

    async def test_get_endpoint_ollama(self, llm_service):
        """Test Ollama specific endpoint resolution."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {
                "base_url": "http://localhost:11434",
            }
            assert llm_service._get_endpoint("chat/completions") == "http://localhost:11434/api/chat"
            assert llm_service._get_endpoint("embeddings") == "http://localhost:11434/api/embeddings"

    async def test_get_endpoint_standard(self, llm_service):
        """Test standard OpenAI-style endpoint resolution."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {
                "base_url": "https://api.openai.com/v1",
            }
            assert llm_service._get_endpoint("chat/completions") == "https://api.openai.com/v1/chat/completions"

    async def test_format_request_ollama(self, llm_service):
        """Test request formatting for Ollama."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {"model": "test-model"}
            messages = [{"role": "user", "content": "hi"}]
            request = llm_service._format_request(messages, stop=["\n"])
            assert request["model"] == "test-model"
            assert request["stream"] is False
            assert request["stop"] == ["\n"]
            assert request["options"]["num_predict"] == 16000

    async def test_format_request_openai(self, llm_service):
        """Test request formatting for OpenAI."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"model": "gpt-4"}
            messages = [{"role": "user", "content": "hi"}]
            request = llm_service._format_request(messages, temperature=0.5)
            assert request["model"] == "gpt-4"
            assert request["temperature"] == 0.5
            assert request["max_tokens"] == 16000


@pytest.mark.asyncio
class TestLLMServiceGeneration:
    """Test LLMService generation methods."""

    async def test_generate_async_success(self, llm_service):
        """Test successful async generation."""
        mock_data = {
            "choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop"}]
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "model": "m"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.return_value = MockResponse(mock_data)
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                result = await llm_service.generate("hi")
                assert result == "Hello world"
                mock_client.post.assert_called_once()

    async def test_generate_async_provider_error(self, llm_service):
        """Test provider error handling in async generation."""
        mock_data = {
            "error": {"message": "Invalid API Key", "code": "invalid_api_key"}
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "model": "m"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.return_value = MockResponse(mock_data)
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                result = await llm_service.generate("hi")
                assert "Provider Error" in result
                assert "Invalid API Key" in result

    async def test_generate_async_http_error(self, llm_service):
        """Test HTTP error handling in async generation."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "model": "m"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            # Create a response that will fail raise_for_status
            error_response = MockResponse({"error": "server down"}, status_code=500)
            mock_client.post.return_value = error_response
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                result = await llm_service.generate("hi")
                assert "Error" in result

    async def test_generate_async_timeout(self, llm_service):
        """Test timeout handling in async generation."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "model": "m"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                result = await llm_service.generate("hi")
                assert "Timeout occurred" in result

    async def test_generate_structured_success(self, llm_service):
        """Test successful structured generation."""
        class TestOutput(BaseModel):
            answer: str

        mock_result = mock.Mock()
        mock_result.output = TestOutput(answer="structured response")
        
        mock_agent_instance = mock.AsyncMock()
        mock_agent_instance.run.return_value = mock_result
        
        with mock.patch("graph_rlm.backend.src.core.llm.PydanticAgent", return_value=mock_agent_instance):
            with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
                mock_settings.LLM_PROVIDER = "openai"
                mock_settings.get_llm_config.return_value = {"model": "gpt-4"}
                
                result = await llm_service.generate_structured("hi", TestOutput)
                assert result.answer == "structured response"
                mock_agent_instance.run.assert_called_once_with("hi")

    async def test_generate_structured_error(self, llm_service):
        """Test error handling in structured generation."""
        class TestOutput(BaseModel):
            answer: str

        mock_agent_instance = mock.AsyncMock()
        mock_agent_instance.run.side_effect = ValueError("Validation error")
        
        with mock.patch("graph_rlm.backend.src.core.llm.PydanticAgent", return_value=mock_agent_instance):
            with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
                mock_settings.LLM_PROVIDER = "openai"
                mock_settings.get_llm_config.return_value = {"model": "gpt-4"}
                
                with pytest.raises(ValueError, match="Validation error"):
                    await llm_service.generate_structured("hi", TestOutput)

    async def test_refresh_closes_clients(self, llm_service):
        """Test that refresh closes clients."""
        mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
        loop = mock.Mock()
        loop.is_closed.return_value = False
        llm_service._loop_resources[loop] = (mock_client, mock.Mock())
        
        await llm_service.refresh()
        mock_client.aclose.assert_called_once()
        assert loop not in llm_service._loop_resources


@pytest.mark.asyncio
class TestLLMServiceStreaming:
    """Test LLMService streaming generation."""

    async def test_generate_stream_success(self, llm_service):
        """Test successful streaming generation."""
        # This is more complex to mock due to httpx.AsyncClient.stream context manager
        # and aiter_lines()
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "model": "m"}
            
            async def mock_aiter():
                lines = [
                    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                    'data: {"choices": [{"delta": {"content": " "}}]}',
                    'data: {"choices": [{"delta": {"content": "world"}}]}',
                    'data: [DONE]'
                ]
                for line in lines:
                    yield line
            
            mock_response = mock.AsyncMock()
            mock_response.raise_for_status.return_value = None
            # Use a regular Mock for aiter_lines so it returns the generator directly
            mock_response.aiter_lines = mock.Mock(return_value=mock_aiter())
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.stream.return_value.__aenter__.return_value = mock_response
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                chunks = []
                gen_wrapper = await llm_service.generate("hi", stream=True)
                async for chunk in gen_wrapper:
                    chunks.append(chunk)
                
                assert "".join(chunks) == "Hello world"

    async def test_generate_async_ollama_success(self, llm_service):
        """Test successful async generation with Ollama."""
        mock_data = {
            "message": {"content": "Ollama response"}
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {"base_url": "http://localhost:11434", "model": "m"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.return_value = MockResponse(mock_data)
            
            with mock.patch.object(llm_service, "_get_client", return_value=mock_client):
                result = await llm_service.generate("hi")
                assert result == "Ollama response"

    async def test_get_embedding_openai_success(self, llm_service):
        """Test successful embedding generation with OpenAI."""
        mock_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test", "embedding_model": "emb"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.return_value = MockResponse(mock_data)
            
            with mock.patch.object(llm_service, "_get_client_and_lock", return_value=(mock_client, mock.AsyncMock())):
                result = await llm_service.get_embedding("text")
                assert result == [0.1, 0.2, 0.3]

    async def test_get_embedding_ollama_success(self, llm_service):
        """Test successful embedding generation with Ollama."""
        mock_data = {
            "embedding": [0.4, 0.5, 0.6]
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {"base_url": "http://localhost:11434", "embedding_model": "emb"}
            
            mock_client = mock.AsyncMock(spec=httpx.AsyncClient)
            mock_client.post.return_value = MockResponse(mock_data)
            
            with mock.patch.object(llm_service, "_get_client_and_lock", return_value=(mock_client, mock.AsyncMock())):
                result = await llm_service.get_embedding("text")
                assert result == [0.4, 0.5, 0.6]

    def test_compute_cosine_similarity(self, llm_service):
        """Test cosine similarity calculation."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        assert llm_service.compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
        
        v3 = [0.0, 1.0]
        assert llm_service.compute_cosine_similarity(v1, v3) == pytest.approx(0.0)
        
        assert llm_service.compute_cosine_similarity([1.0], [2.0]) == pytest.approx(1.0)
        assert llm_service.compute_cosine_similarity([], [1.0]) == 0.0

    def test_unload_model_ollama(self, llm_service):
        """Test unloading model in Ollama."""
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {"base_url": "http://localhost:11434"}
            
            with mock.patch("httpx.Client") as mock_client_class:
                mock_client = mock.Mock()
                mock_client_class.return_value.__enter__.return_value = mock_client
                
                result = llm_service.unload_model("test-model")
                assert result is True
                mock_client.post.assert_called_once()

    def test_list_models_ollama(self, llm_service):
        """Test listing models in Ollama."""
        mock_data = {
            "models": [{"name": "llama3:latest"}, {"name": "nomic-embed-text:latest"}]
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.get_llm_config.return_value = {"base_url": "http://localhost:11434"}
            
            with mock.patch("httpx.Client") as mock_client_class:
                mock_client = mock.Mock()
                mock_client.get.return_value = MockResponse(mock_data)
                mock_client_class.return_value.__enter__.return_value = mock_client
                
                models = llm_service.list_models()
                assert len(models) == 2
                assert models[0]["id"] == "llama3:latest"
                assert models[1]["type"] == "embedding"

    def test_list_models_openai(self, llm_service):
        """Test listing models in OpenAI/OpenRouter."""
        mock_data = {
            "data": [{"id": "gpt-4", "name": "GPT-4"}, {"id": "text-embedding-3", "supported_parameters": []}]
        }
        
        with mock.patch("graph_rlm.backend.src.core.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.get_llm_config.return_value = {"base_url": "http://test"}
            
            with mock.patch("httpx.Client") as mock_client_class:
                mock_client = mock.Mock()
                mock_client.get.return_value = MockResponse(mock_data)
                mock_client_class.return_value.__enter__.return_value = mock_client
                
                models = llm_service.list_models()
                assert len(models) == 2
                assert models[0]["id"] == "gpt-4"
                assert models[1]["type"] == "embedding"

    def test_polyfill_tool_calls(self, llm_service):
        """Test polyfilling native tool calls to Python code."""
        tool_calls = [
            {
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}'
                }
            }
        ]
        result = llm_service._polyfill_tool_calls(tool_calls)
        assert "search(query='test')" in result
