import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import logging
import uuid

# from graph_rlm.backend.src.core.agent import RLMInterface # RLMInterface is not used in this file
from graph_rlm.backend.src.core.config import settings
from graph_rlm.backend.src.core.db import db
from graph_rlm.backend.src.core.scratchpad_builder import scratchpad_builder
from graph_rlm.backend.src.core.thimac_memory import (
    ThimacIntention,
    ThimacLevel,
    ThimacMemory,
    ThimacOperation,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_semantics")


async def verify_semantic_grounding():
    logger.info("Starting Semantic Grounding Verification...")

    # 1. Initialize Mock Session
    session_id = f"verify-{uuid.uuid4().hex[:8]}"
    logical_id = "V1:S1"

    # 2. Simulate a thought with high semantic value
    prompt = "Research the legacy summarizer in git history and recover the logic."
    result = """
    Found commit 123f7cd0 with the following logic:
    def _summarize_content(self, task, gestalt):
        # ... logic for Chain-of-Density ...
        return dense_summary
    Successfully recovered the core CoD prompt and parsing logic.
    """

    # 3. Test Semantic Summarizer Directly
    from graph_rlm.backend.src.core.semantic_summarizer import summarize_event

    logger.info("Testing semantic_summarizer.summarize_event...")
    gist = await summarize_event(prompt, result)
    logger.info(f"Generated Gist: {gist}")
    assert gist and len(gist) > 0, "Gist generation failed"

    # 4. Test Thimac Ingestion with Gist
    logger.info("Testing Thimac ingestion with gist...")
    thimac = ThimacMemory()
    thought_data = {
        "id": str(uuid.uuid4()),
        "prompt": prompt,
        "status": "success",
        "result": result,
        "created_at": 1600000000000,
        "turn_id": 1,
        "step_id": 1,
    }
    event = thimac.ingest_thought(thought_data, semantic_gist=gist)
    assert event.semantic_gist == gist, "Thimac failed to store semantic_gist"

    gestaltLines = thimac.get_gestalt_string()
    logger.info("Gestalt Output:")
    print(gestaltLines)
    assert "Recent Directive Gists" in gestaltLines, "Gestalt missing recent directives"
    assert gist in gestaltLines, "Gestalt missing the specific gist"

    # 5. Test Database Persistence
    logger.info("Testing DB persistence...")
    thought_id = str(uuid.uuid4())
    db.create_thought_node(
        thought_id=thought_id,
        prompt=prompt,
        session_id=session_id,
        status="success",
        result=result,
        semantic_gist=gist,
        step_id=1,
        turn_id=1,
        round_id="round-1",
    )

    # Retrieve it back
    from graph_rlm.backend.src.core.context_index import context_index

    thoughts = context_index.get_session_thoughts(session_id)
    assert len(thoughts) > 0, "Thought not persisted"
    assert (
        thoughts[0].get("semantic_gist") == gist
    ), "DB failed to retrieve semantic_gist"

    # 6. Test Scratchpad Rendering
    logger.info("Testing Scratchpad rendering...")
    pad = await scratchpad_builder.build_scratchpad(
        task="Verify semantic grounding",
        session_id=session_id,
        root_session_id=session_id,
        current_round_id="round-1",
    )
    logger.info("Scratchpad Output Snippet:")
    print(pad[:500])
    assert gist in pad, "Scratchpad missing semantic gist in table"

    logger.info("Verification SUCCESSFUL!")


if __name__ == "__main__":
    asyncio.run(verify_semantic_grounding())
