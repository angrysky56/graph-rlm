"""
Semantic Summarizer for Graph-RLM.
Re-integrates the legacy Hierarchical Chain-of-Density (CoD) logic to generate
dense, directive gists of agent interactions.
"""

import json
import logging
import re
from typing import Optional

from .circuit import generate_correlation_id, get_correlation_id
from .config import settings
from .services.circuit import protected_llm_generate

logger = logging.getLogger("graph_rlm.summarizer")


async def summarize_event(
    prompt_text: str,
    result_text: str,
    model: Optional[str] = None,
) -> str:
    """
    Generates a dense semantic gist of an interaction.
    Uses Hierarchical Chain-of-Density to maximize information per token.
    """
    try:
        # Combine prompt and result into a single context block
        full_text = (
            f"USER/AGENT PROMPT:\n{prompt_text}\n\nRESULT/OUTPUT:\n{result_text}"
        )

        context = full_text[:150000]

        # 1. Hierarchical Chain-of-Density Prompt
        # This prompt asks the model to iteratively refine the summary for density.
        cod_prompt = f"""Summarize the following interaction into a DENSE SEMANTIC GIST.
You are the Agent's "Attention Schema" - your goal is to prevent information amnesia.

Focus on identifying and retaining:
- EXACT IDENTIFIERS: UUIDs, variable names, file paths, function signatures.
- ATOMIC OUTCOMES: "Created X", "Read Y", "Error: [Specific Code]", "Fixed Z".
- COGNITIVE CONTEXT: "Branching to solve X", "Refinement of Y", "Validating Z".
- GROUNDING DATA: Specific values or data points found in the output.

Format: JSON list of objects [{{"Summary": "...", "Denser_Summary": "..."}}]
The list should contain 2 iterations. The "Denser_Summary" MUST contain more specific details
(like UUIDs or paths) than the "Summary" while remaining under 1000 characters.

Input:
---
{context}
---"""

        response = await protected_llm_generate(
            cod_prompt,
            model=model or settings.SUMMARY_MODEL,
            correlation_id=get_correlation_id() or generate_correlation_id(),
        )

        if not response:
            return _fallback_summary(result_text)

        # 2. Extract the densest summary from the JSON response
        try:
            # Handle potential markdown code blocks in the response
            json_str = response.strip()
            match = re.search(r"```(?:json)?\n(.*?)\n```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            # If no code block, try to find the start and end of a JSON list
            elif json_str.find("[") != -1 and json_str.rfind("]") != -1:
                json_str = json_str[json_str.find("[") : json_str.rfind("]") + 1]

            data = json.loads(json_str)
            if isinstance(data, list) and len(data) > 0:
                # Return the last (densest) summary in the list
                final_summary = data[-1].get("Denser_Summary") or data[-1].get(
                    "Summary"
                )
                if final_summary:
                    return final_summary.strip()
            elif isinstance(data, dict):
                final_summary = data.get("Denser_Summary") or data.get("Summary")
                if final_summary:
                    return final_summary.strip()

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as je:
            logger.debug(
                "JSON parsing of semantic summary failed: %s. Falling back to raw response.",
                je,
            )
            # Fallback: if it's not JSON but length looks like a summary, use it
            if len(response) > 20 and "{" not in response:
                return response.strip()[:500]

        return response.strip()[:500]

    except (RuntimeError, ValueError, AttributeError, Exception) as e:
        logger.warning("Summary generation failed: %s", e)
        return _fallback_summary(result_text)


def _fallback_summary(text: str) -> str:
    """Static fallback for when LLM summarization fails."""
    clean = text.strip()
    if not clean:
        return "Empty Output"
    # Basic truncation with ellipsis
    return clean[:100] + "..." if len(clean) > 100 else clean
