import asyncio
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_rlm.backend.src.core.scratchpad_builder import ScratchpadBuilder


async def test_cod_summarization():
    print("Testing Chain of Density (CoD) summarization...")
    builder = ScratchpadBuilder()

    # Sample long text (>2000 chars) to trigger summarization
    content = """
    Agent Execution Trace for Round 42:
    The agent started by initializing the environment and checking the database connectivity.
    It then proceeded to execute a series of complex REPL commands to analyze the graph data.
    Step 1: MATCH (n:Thought) RETURN count(n) AS thought_count. Result: 1543.
    Step 2: Analysis of the sheaf score for the recent trajectory. The score was 0.82, indicating a potential loop.
    Step 3: RepE analysis flagged a high level of evasion (score: -0.45) in the last three steps.
    Step 4: Attempted to resolve the loop by clearing the context and re-injecting core axioms.
    Step 5: The dreamer module (v2.1) performed a cross-check of the subsistence state and found multiple materialized results of type 'code_artifact'.
    Step 6: Spectral energy deviation was calculated as 0.65, confirming the drift in the agent's focus.
    Step 7: The agent successfully materialized a new skill 'graph_traversal_optimized' and stored it in the skill storage (FalkorDB).
    Step 8: Final validation passed with an oMCD score of 0.95.
    ... [Repeated trace data to exceed 2000 characters] ...
    """ * 10

    print(f"Input length: {len(content)} characters.")

    # We will test the _summarize_content method
    # Since it calls protected_llm_generate, we might need a real API key or mock it
    # For this verification, I'll mock protected_llm_generate to test the parsing logic first

    print("\n--- Phase 1: Testing Parsing Logic with Mocked Response ---")
    mock_json_response = """
    [
      {
        "Missing_Entities": "Thought count; sheaf score; RepE evasion",
        "Denser_Summary": "The agent analyzed 1543 Thoughts and identified a loop via a 0.82 sheaf score. RepE flagged significant evasion (-0.45) during REPL execution. Environmental initialization confirmed database connectivity before trajectory analysis. This initial summary captures the core diagnostic metrics and the agent's starting state within the round, establishing a baseline for subsequent densification and iterative refinement of the execution trace summary."
      },
      {
        "Missing_Entities": "dreamer v2.1; spectral energy; subsistence state",
        "Denser_Summary": "Agent round 42 began with database checks before analyzing 1543 Thoughts. A 0.82 sheaf score and -0.45 RepE evasion flagged trajectory loops. Dreamer v2.1 monitored subsistence states, identifying 'code_artifact' materialization amid 0.65 spectral energy deviation. These metrics confirmed drift, necessitating context clearing. This version integrates diagnostic tools and environmental monitoring, focusing on the interplay between cognitive metrics and materialized outputs for improved density and readability."
      },
      {
        "Missing_Entities": "graph_traversal_optimized; oMCD score; FalkorDB",
        "Denser_Summary": "Analyzing 1543 Thoughts, the agent identified loops via 0.82 sheaf scores and -0.45 RepE evasion. Dreamer v2.1 checked subsistence states while 0.65 spectral energy deviation signaled drift. After context clearing, the agent materialized the 'graph_traversal_optimized' skill in FalkorDB. The round concluded with a 0.95 oMCD score, validating the successful optimization. This summary fuses technical metrics with architectural outcomes, providing a high-density overview of the agent's progress and final resolution."
      },
      {
        "Missing_Entities": "REPL commands; core axioms; trajectory analysis",
        "Denser_Summary": "Agent round 42 executed complex REPL commands, analyzing 1543 Thoughts. Diagnostic sheaf (0.82) and RepE evasion (-0.45) metrics flagged trajectory loops, corrected via core axiom re-injection. Dreamer v2.1 monitored subsistence amid 0.65 spectral energy drift, ultimately materializing the 'graph_traversal_optimized' skill in FalkorDB. Validated by a 0.95 oMCD score, the process resolved focus deviations. This final summary maintains strict word counts while maximizing entity density across diagnostic, materialization, and validation phases."
      }
    ]
    """

    # Patch protected_llm_generate
    import graph_rlm.backend.src.core.scratchpad_builder as spb

    async def mock_generate(*args, **kwargs):
        return mock_json_response

    original_generate = spb.protected_llm_generate
    spb.protected_llm_generate = mock_generate

    try:
        summary = await builder._summarize_content(content, "Test Trace")
        print("\nGenerated Summary (from Mock):")
        print(f"'{summary}'")

        # Verify length (approx 70-90 words as per instructions)
        word_count = len(summary.split())
        print(f"Word count: {word_count}")

        # Verify it's the last iteration
        assert "final summary" in summary.lower()
        assert "oMCD score" in summary
        print("✅ Parsing logic verified.")

    finally:
        spb.protected_llm_generate = original_generate

    print("\n--- Phase 2: Testing Fallback Logic (Non-JSON response) ---")
    async def mock_raw_generate(*args, **kwargs):
        return "This is a plain text fallback summary that should be returned if JSON parsing fails."

    spb.protected_llm_generate = mock_raw_generate
    try:
        summary = await builder._summarize_content(content, "Test Trace")
        print(f"Fallback Summary: '{summary}'")
        assert "plain text fallback" in summary
        print("✅ Fallback logic verified.")
    finally:
        spb.protected_llm_generate = original_generate

if __name__ == "__main__":
    asyncio.run(test_cod_summarization())
