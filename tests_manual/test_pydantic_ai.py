import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()


class Result(BaseModel):
    summary: str
    confidence: float


async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Official OpenRouter provider
    provider = OpenRouterProvider(
        api_key=api_key,
        app_url="https://github.com/angrysky56/graph-rlm",
        app_title="Graph-RLM",
    )

    model = OpenAIChatModel("anthropic/claude-3.5-sonnet", provider=provider)

    # In 1.59.x, it's output_type, not result_type
    agent = Agent(model, output_type=Result)

    result = await agent.run("What is the capital of France?")
    print(f"Result Type: {type(result.output)}")
    print(f"Result Data: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
