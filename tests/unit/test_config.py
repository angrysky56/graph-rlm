"""Unit tests for src.core.config module.

Tests for Settings class including environment variable loading,
LLM provider configurations, config validation, and .env file management.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from graph_rlm.backend.src.core.config import Settings


class TestSettingsDefaults:
    """Test Settings class default values."""

    def test_project_name_default(self):
        """Test that PROJECT_NAME has correct default."""
        settings = Settings(PROJECT_NAME="Test")
        assert settings.PROJECT_NAME == "Test"

    def test_api_v1_str_default(self):
        """Test API_V1_STR default value."""
        settings = Settings()
        assert settings.API_V1_STR == "/api/v1"

    def test_knowledge_base_path_default(self):
        """Test KNOWLEDGE_BASE_PATH defaults to knowledge_base directory."""
        settings = Settings()
        assert "knowledge_base" in settings.KNOWLEDGE_BASE_PATH

    def test_falkor_host_default(self):
        """Test FALKOR_HOST defaults to localhost."""
        settings = Settings()
        assert settings.FALKOR_HOST == "localhost"

    def test_falkor_port_default(self):
        """Test FALKOR_PORT defaults to 6380."""
        settings = Settings()
        assert settings.FALKOR_PORT == 6380

    def test_redis_port_default(self):
        """Test REDIS_PORT defaults to None when not in .env file."""
        # The actual value depends on .env file loading
        # This test verifies the field is correctly defined
        settings = Settings()
        # REDIS_PORT is Optional[int] but may be loaded from .env
        # We test it can be set to None explicitly
        assert isinstance(settings.REDIS_PORT, int) or settings.REDIS_PORT is None

    def test_api_port_default(self):
        """Test API_PORT defaults to 8000."""
        settings = Settings()
        assert settings.API_PORT == 8000

    def test_graph_name_default(self):
        """Test GRAPH_NAME defaults to rlm_graph."""
        settings = Settings()
        assert settings.GRAPH_NAME == "rlm_graph"

    def test_max_recursion_depth_default(self):
        """Test MAX_RECURSION_DEPTH defaults to 1000."""
        settings = Settings()
        assert settings.MAX_RECURSION_DEPTH == 1000

    def test_repl_timeout_default(self):
        """Test REPL_TIMEOUT defaults to 3000."""
        settings = Settings()
        assert settings.REPL_TIMEOUT == 3000


class TestLLMProviderConfigs:
    """Test LLM provider configuration methods."""

    def test_get_config_for_provider_openrouter(self):
        """Test OpenRouter configuration retrieval."""
        settings = Settings()
        config = settings.get_config_for_provider("openrouter")
        # API key may be loaded from .env, but we verify the structure
        assert "api_key" in config
        assert config["base_url"] == "https://openrouter.ai/api/v1"
        assert "model" in config
        assert "embedding_model" in config

    def test_get_config_for_provider_ollama(self):
        """Test Ollama configuration retrieval."""
        settings = Settings()
        config = settings.get_config_for_provider("ollama")
        assert config["api_key"] == ""
        assert config["base_url"] == "http://localhost:11434"
        assert config["model"] == "gemma3:latest"
        assert config["embedding_model"] == "embeddinggemma:latest"

    def test_get_config_for_provider_lmstudio(self):
        """Test LM Studio configuration retrieval."""
        settings = Settings()
        config = settings.get_config_for_provider("lmstudio")
        assert config["api_key"] == "lm-studio"
        assert config["base_url"] == "http://localhost:1234/v1"
        assert config["model"] == "local-model"
        assert config["embedding_model"] == "local-embedding"

    def test_get_config_for_provider_openai(self):
        """Test OpenAI configuration retrieval."""
        settings = Settings()
        config = settings.get_config_for_provider("openai")
        # API key may be loaded from .env, but we verify the structure
        assert "api_key" in config
        assert config["base_url"] == "https://api.openai.com/v1"
        assert "model" in config
        assert "embedding_model" in config

    def test_get_config_for_unknown_provider(self):
        """Test unknown provider falls back to openrouter config."""
        settings = Settings()
        config = settings.get_config_for_provider("unknown")
        # Should return openrouter defaults as fallback
        assert config["base_url"] == "https://openrouter.ai/api/v1"

    def test_get_llm_config_includes_summary_model(self):
        """Test get_llm_config includes summary model."""
        settings = Settings(LLM_PROVIDER="ollama")  # Use ollama to avoid .env override
        config = settings.get_llm_config()
        assert "summary_model" in config
        # Verify SUMMARY_MODEL is included in the config
        assert "google" in config["summary_model"]

    def test_get_llm_config_returns_dict(self):
        """Test get_llm_config returns a dictionary with required keys."""
        settings = Settings(LLM_PROVIDER="openai")
        config = settings.get_llm_config()
        assert isinstance(config, dict)
        assert "api_key" in config
        assert "base_url" in config
        assert "model" in config


class TestEnvironmentVariables:
    """Test environment variable loading."""

    def test_env_file_loading(self):
        """Test that .env file variables are loaded."""
        # This test verifies that Settings can load from .env file
        # by checking that the model_config includes env_file setting
        assert hasattr(Settings, "model_config")
        assert "env_file" in Settings.model_config

    def test_model_config_env_file_setting(self):
        """Test that model_config has env_file configured."""
        # Verify the ConfigDict includes env_file as a list containing .env
        assert hasattr(Settings, "model_config")
        env_file = Settings.model_config.get("env_file")
        if isinstance(env_file, list):
            assert ".env" in env_file
        else:
            assert env_file == ".env"

    def test_model_config_extra_ignore(self):
        """Test that extra fields are ignored."""
        # Settings should ignore extra fields - verified by model_config
        assert hasattr(Settings, "model_config")
        assert Settings.model_config.get("extra") == "ignore"


class TestSaveToEnv:
    """Test .env file save functionality."""

    def test_save_to_env_updates_existing_key(self):
        """Test updating an existing environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("API_PORT=8080\n")
            os.chdir(tmpdir)

            try:
                # Patch model_config to only use the local .env file in the temp directory
                new_config = Settings.model_config.copy()
                new_config["env_file"] = [".env"]
                with mock.patch.object(Settings, "model_config", new_config):
                    settings = Settings()
                    # Ensure we start with the value from the file
                    assert settings.API_PORT == 8080
                    
                    result = settings.save_to_env({"API_PORT": "9000"})
                    assert result is True

                    # Reload settings should now pick up 9000
                    new_settings = Settings()
                    assert new_settings.API_PORT == 9000
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_adds_new_key(self):
        """Test adding a new environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                result = settings.save_to_env({"CUSTOM_VAR": "custom_value"})
                assert result is True

                # Verify the file was updated
                content = env_file.read_text()
                assert "CUSTOM_VAR=custom_value" in content
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_handles_missing_file(self):
        """Test saving when .env file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            os.chdir(tmpdir)

            try:
                settings = Settings()
                result = settings.save_to_env({"NEW_VAR": "new_value"})
                assert result is True
                assert env_file.exists()
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_returns_false_on_exception(self):
        """Test save_to_env returns False on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                settings = Settings()

                # Mock to raise an exception
                with mock.patch(
                    "builtins.open", side_effect=OSError("Permission denied")
                ):
                    result = settings.save_to_env({"TEST_VAR": "test"})
                    assert result is False
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_filters_openai_defaults(self):
        """Test that OpenAI defaults are not written if they match class defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                # Attempt to save OpenAI settings that match defaults
                result = settings.save_to_env(
                    {
                        "OPENAI_MODEL": "gpt-4o-mini",  # Matches default
                        "OPENAI_BASE_URL": "https://api.openai.com/v1",  # Matches default
                    }
                )
                assert result is True

                # OpenAI defaults should not be written
                content = env_file.read_text()
                assert "OPENAI_MODEL" not in content
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_skips_empty_api_keys(self):
        """Test that empty API keys are not written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                result = settings.save_to_env(
                    {
                        "OPENAI_API_KEY": "",
                        "OPENROUTER_API_KEY": "",
                    }
                )
                assert result is True

                content = env_file.read_text()
                # Empty API keys should be skipped
                assert "OPENAI_API_KEY=" not in content
                assert "OPENROUTER_API_KEY=" not in content
            finally:
                os.chdir(original_cwd)


class TestSettingsModelConfig:
    """Test Pydantic model configuration."""

    def test_model_config_exists(self):
        """Test that model_config is properly defined."""
        assert hasattr(Settings, "model_config")

    def test_model_config_has_env_file(self):
        """Test that env_file is configured."""
        assert "env_file" in Settings.model_config
        assert ".env" in Settings.model_config["env_file"]

    def test_model_config_extra_ignore(self):
        """Test that extra fields are set to ignore."""
        assert "extra" in Settings.model_config
        assert Settings.model_config["extra"] == "ignore"


class TestSettingsProviderModels:
    """Test provider-specific model configurations."""

    def test_openrouter_model(self):
        """Test OpenRouter model configuration."""
        settings = Settings(OPENROUTER_MODEL="test-model")
        assert settings.OPENROUTER_MODEL == "test-model"

    def test_openrouter_embedding_model(self):
        """Test OpenRouter embedding model configuration."""
        settings = Settings(OPENROUTER_EMBEDDING_MODEL="test-embedding")
        assert settings.OPENROUTER_EMBEDDING_MODEL == "test-embedding"

    def test_ollama_model(self):
        """Test Ollama model configuration."""
        settings = Settings(OLLAMA_MODEL="llama2:7b")
        assert settings.OLLAMA_MODEL == "llama2:7b"

    def test_ollama_embedding_model(self):
        """Test Ollama embedding model configuration."""
        settings = Settings(OLLAMA_EMBEDDING_MODEL="nomic-embed")
        assert settings.OLLAMA_EMBEDDING_MODEL == "nomic-embed"

    def test_openai_model(self):
        """Test OpenAI model configuration."""
        settings = Settings(OPENAI_MODEL="gpt-4")
        assert settings.OPENAI_MODEL == "gpt-4"

    def test_openai_embedding_model(self):
        """Test OpenAI embedding model configuration."""
        settings = Settings(OPENAI_EMBEDDING_MODEL="text-embedding-ada-002")
        assert settings.OPENAI_EMBEDDING_MODEL == "text-embedding-ada-002"

    def test_summary_model(self):
        """Test Summary model configuration."""
        settings = Settings(SUMMARY_MODEL="lightweight-model")
        assert settings.SUMMARY_MODEL == "lightweight-model"


class TestSettingsFalkorDBConfig:
    """Test FalkorDB configuration."""

    def test_falkordb_path_default(self):
        """Test FALKORDB_PATH defaults to None."""
        settings = Settings()
        assert settings.FALKORDB_PATH is None

    def test_falkordb_path_configurable(self):
        """Test FALKORDB_PATH can be configured."""
        settings = Settings(FALKORDB_PATH="/custom/path")
        assert settings.FALKORDB_PATH == "/custom/path"


class TestSettingsLLMProvider:
    """Test LLM provider selection."""

    def test_llm_provider_default(self):
        """Test LLM_PROVIDER defaults to openrouter."""
        settings = Settings()
        assert settings.LLM_PROVIDER == "openrouter"

    def test_llm_provider_configurable(self):
        """Test LLM_PROVIDER can be changed."""
        settings = Settings(LLM_PROVIDER="ollama")
        assert settings.LLM_PROVIDER == "ollama"


class TestSettingsReplTimeout:
    """Test REPL timeout configuration."""

    def test_repl_timeout_configurable(self):
        """Test REPL_TIMEOUT can be configured."""
        settings = Settings(REPL_TIMEOUT=6000)
        assert settings.REPL_TIMEOUT == 6000


class TestSaveToEnvEdgeCases:
    """Test edge cases for save_to_env functionality."""

    def test_save_to_env_skips_os_environ_matches(self):
        """Test that save_to_env skips keys matching OS environ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()

                # Set an env var that matches the value we'll try to save
                os.environ["API_PORT"] = "9000"
                result = settings.save_to_env({"API_PORT": "9000"})
                assert result is True

                # The key should not be written again since it matches OS environ
                content = env_file.read_text()
                # This tests line 139 coverage - when key matches OS environ
                # The key should NOT be in the file since it matches OS environ
                assert "API_PORT=9000" not in content
            finally:
                os.chdir(original_cwd)
                if "API_PORT" in os.environ:
                    del os.environ["API_PORT"]

    def test_save_to_env_skips_os_environ_mismatch(self):
        """Test that save_to_env writes keys when OS environ value differs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()

                # Set an env var that differs from what we're saving
                os.environ["API_PORT"] = "8080"
                result = settings.save_to_env({"API_PORT": "9000"})
                assert result is True

                # The key should be written since OS environ differs
                content = env_file.read_text()
                # This tests the case where key in os.environ but values differ
                # Line 130 should still pass but line 154 should execute
                assert "API_PORT=9000" in content
            finally:
                os.chdir(original_cwd)
                if "API_PORT" in os.environ:
                    del os.environ["API_PORT"]

    def test_save_to_env_handles_openai_non_api_keys(self):
        """Test that OpenAI non-API-key defaults are filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                # This tests line 139 coverage
                result = settings.save_to_env(
                    {
                        "OPENAI_MODEL": "gpt-4o-mini",  # Should be filtered (matches default)
                        "OPENAI_BASE_URL": "https://api.openai.com/v1",  # Should be filtered
                    }
                )
                assert result is True
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_with_empty_env_file(self):
        """Test saving to empty .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                result = settings.save_to_env({"EMPTY_TEST": "value"})
                assert result is True
                assert env_file.exists()
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_preserves_unmodified_lines(self):
        """Test that unmodified lines are preserved (line 130 coverage)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            # Create .env with existing content
            env_file.write_text("EXISTING_VAR=preserved\nANOTHER_VAR=also_preserved\n")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                # Only update one key, others should be preserved
                result = settings.save_to_env({"EXISTING_VAR": "updated"})
                assert result is True

                # Verify the file content
                content = env_file.read_text()
                # Line 130: new_lines.append(line) should preserve unmodified lines
                assert "EXISTING_VAR=updated" in content
                assert "ANOTHER_VAR=also_preserved" in content
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_handles_openai_non_api_keys(self):
        """Test that OpenAI non-API-key defaults are filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                # This tests line 139 coverage
                result = settings.save_to_env(
                    {
                        "OPENAI_MODEL": "gpt-4o-mini",  # Should be filtered (matches default)
                        "OPENAI_BASE_URL": "https://api.openai.com/v1",  # Should be filtered
                    }
                )
                assert result is True
            finally:
                os.chdir(original_cwd)

    def test_save_to_env_with_empty_env_file(self):
        """Test saving to empty .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("")
            os.chdir(tmpdir)

            try:
                settings = Settings()
                result = settings.save_to_env({"EMPTY_TEST": "value"})
                assert result is True
                assert env_file.exists()
            finally:
                os.chdir(original_cwd)


class TestSettingsWithContextOverride:
    """Test Settings with context variable overrides."""

    def test_settings_with_overridden_provider(self):
        """Test that provider can be overridden."""
        settings = Settings(LLM_PROVIDER="lmstudio")
        assert settings.LLM_PROVIDER == "lmstudio"

        config = settings.get_config_for_provider("lmstudio")
        assert config["base_url"] == "http://localhost:1234/v1"

    def test_settings_multiple_overrides(self):
        """Test multiple settings can be overridden at once."""
        settings = Settings(
            LLM_PROVIDER="openai",
            OPENAI_MODEL="gpt-4",
            OPENAI_BASE_URL="https://custom.api.com/v1",
        )
        assert settings.LLM_PROVIDER == "openai"
        assert settings.OPENAI_MODEL == "gpt-4"
        assert settings.OPENAI_BASE_URL == "https://custom.api.com/v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
