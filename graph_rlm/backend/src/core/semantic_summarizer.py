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

        # truncated text to avoid token overflow in summary model
        # lightweight models usually have smaller context or we want speed
        context = full_text[:8000]

        # 1. Hierarchical Chain-of-Density Prompt
        # This prompt asks the model to iteratively refine the summary for density.
        cod_prompt = f"""Summarize the following interaction into a DENSE SEMANTIC GIST.
Focus on identifying:
- Exact identifiers (variable names, file paths, UUIDs, function names)
- Core result of the operation (success/fail/error code)
- Current cognitive state (waiting for input/branching/finished)

Format: JSON list of objects [{{"Summary": "...", "Denser_Summary": "..."}}]
The list should contain 2-3 iterations, with each "Denser_Summary" being progressively
more information-dense but of similar length to the previous iteration.

Input:
---
{context}
---"""

        response = await protected_llm_generate(
            cod_prompt,
            model=model or "google/gemini-2.0-flash-lite",
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
