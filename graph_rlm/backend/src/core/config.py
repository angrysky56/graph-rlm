
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Graph-RLM"
    API_V1_STR: str = "/api/v1"

    # Database
    FALKOR_HOST: str = "localhost"
    FALKOR_PORT: int = 6380  # Default, or can use REDIS_PORT from env

    # Matching .env vars specifically
    REDIS_PORT: Optional[int] = None # Will read from .env
    API_PORT: int = 8000
    FALKORDB_PATH: Optional[str] = None
    LLM_PROVIDER: str = "ollama"

    GRAPH_NAME: str = "rlm_graph"

    # LLM Settings (Primary Provider)
    LLM_PROVIDER: str = "ollama"  # ollama, openrouter, lmstudio, openai

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "google/gemini-3-flash-preview"
    OPENROUTER_EMBEDDING_MODEL: str = "google/gemini-embedding-001"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3:latest"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # LM Studio / Local
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_MODEL: str = "local-model"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    def get_config_for_provider(self, provider: str) -> dict:
        """Returns the LLM configuration for a specific provider."""
        # Re-use the logic from get_llm_config but for any provider
        configs = {
            "openrouter": {
                "api_key": self.OPENROUTER_API_KEY,
                "base_url": self.OPENROUTER_BASE_URL,
                "model": self.OPENROUTER_MODEL,
                "embedding_model": self.OPENROUTER_EMBEDDING_MODEL
            },
            "ollama": {
                "api_key": "",
                "base_url": self.OLLAMA_BASE_URL,
                "model": self.OLLAMA_MODEL,
                "embedding_model": self.OLLAMA_EMBEDDING_MODEL,
            },
            "lmstudio": {
                "api_key": "lm-studio",
                "base_url": self.LMSTUDIO_BASE_URL,
                "model": self.LMSTUDIO_MODEL,
                "embedding_model": "local-embedding"
            },
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "model": self.OPENAI_MODEL,
                "embedding_model": self.OPENAI_EMBEDDING_MODEL
            },
        }
        return configs.get(provider, configs["ollama"])

    def get_llm_config(self) -> dict:
        """Returns the active LLM configuration based on LLM_PROVIDER."""
        return self.get_config_for_provider(self.LLM_PROVIDER)

    def save_to_env(self, config_updates: dict) -> bool:
        """
        Updates the .env file with new values and reloads the settings.
        """
        try:
            # 1. Read existing .env lines
            lines = []
            try:
                with open(".env", "r") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                pass

            # 2. Update or append keys
            updated_keys = set()
            new_lines = []

            for line in lines:
                key_part = line.split("=", 1)[0].strip()
                if key_part in config_updates:
                    # Replace this line
                    new_lines.append(f"{key_part}={config_updates[key_part]}\n")
                    updated_keys.add(key_part)
                else:
                    new_lines.append(line)

            # 3. Append missing keys (carefully)
            import os
            for key, value in config_updates.items():
                if key not in updated_keys:
                    # Skip writing keys that are already in OS environ and user didn't explicitly change
                    if key in os.environ and os.environ[key] == str(value):
                        continue

                    # Strict Filter: Do not write OpenAI defaults if they are not active or explicitly set
                    # If the key is OPENAI_... and the value matches the class default, skip it
                    if key.startswith("OPENAI_") and not key.endswith("API_KEY"):
                         # Check if it matches default
                         default_val = getattr(self, key, None)
                         if str(value) == str(default_val):
                             continue

                    # Also skip empty keys if they are purely optional/runtime
                    if value == "" and key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                       continue

                    new_lines.append(f"{key}={value}\n")

            # 4. Write back
            with open(".env", "w") as f:
                f.writelines(new_lines)

            # 5. Hot Reload in-memory (Partial)
            # Pydantic doesn't fully auto-reload from file instantly, so we update the instance dict
            for key, value in config_updates.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            return True
        except Exception as e:
            print(f"Error saving to .env: {e}")
            return False

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
