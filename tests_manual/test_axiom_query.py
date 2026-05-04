
import asyncio

from graph_rlm.backend.src.core.llm import LLMService


async def test_axiom_query_generation():
    llm = LLMService()

    tasks = [
        "Write a python script to scrape a website.",
        "Calculate the orbital trajectory of Mars using LaTeX.",
        "Refactor the database schema for infinite scaling.",
    ]

    system_prompt = (
        "You are the Governance Module of the Agent. "
        "Your goal is to translate a USER TASK into a SEARCH QUERY for retrieving relevant Axioms (Validation Rules). "
        "Axioms are Python files that validate specific behaviors (e.g., 'file persistence', 'math safety', 'python syntax', 'epistemic integrity'). "
        "Return ONLY the search query, focused on the types of validation needed."
    )

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        response = await llm.generate(task, system=system_prompt)
        print(f"Generated Query: {response}")

if __name__ == "__main__":
    asyncio.run(test_axiom_query_generation())
