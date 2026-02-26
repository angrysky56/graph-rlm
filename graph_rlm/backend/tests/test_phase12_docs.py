import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add repo root to path
project_root = Path(__file__).parent.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Define a mock for protected_llm_with_fallback that returns 3 blocks
async def mock_llm_response(**kwargs):
    content = """
Here is the codified knowledge.

```python
def check_arxiv_grounding(paper_id: str) -> bool:
    \"\"\"Checks if a paper is properly grounded in the knowledge base.\"\"\"
    return True
```

```python
# No healing script needed
pass
```

```markdown
# check_arxiv_grounding

## Rationale
Ensures that we don't hallucinate paper contents by enforcing a grounding check.

## Capabilities
- Validates Arxiv IDs.
- Verifies local cache existence.

## Usage
`check_arxiv_grounding("2301.12345")`

## Common Pitfalls
- Moving files manually will break this check.
```
"""
    return content, False


class TestHighFidelityDocs(unittest.IsolatedAsyncioTestCase):

    @patch(
        "graph_rlm.backend.src.core.dream.protected_llm_with_fallback",
        side_effect=mock_llm_response,
    )
    @patch("graph_rlm.backend.src.core.dream.get_axioms_manager")
    async def test_codification_with_docs(self, mock_get_mgr, mock_llm):
        from graph_rlm.backend.src.core.dream import Dreamer

        # Mock AxiomsManager
        mock_mgr = AsyncMock()
        mock_get_mgr.return_value = mock_mgr

        dreamer = Dreamer()

        # Test _codify_axiom
        knowledge = {
            "name": "ArxivGrounding",
            "type": "validator",
            "description": "Check arxiv docs",
        }
        axiom_code, healing_code, doc_text = await dreamer._codify_axiom(
            knowledge, "Research"
        )

        print(f"Captured Doc Text: {doc_text[:100]}...")
        self.assertIn("Rationale", doc_text)
        self.assertIn("Capabilities", doc_text)
        self.assertIn("Common Pitfalls", doc_text)

        # Test _save_axiom propagation
        # Note: _save_axiom internally calls axioms_mgr.save_axiom
        await dreamer._save_axiom(
            code=axiom_code,
            description="Test desc",
            domain="Research",
            markdown_body=doc_text,
        )

        # Verify markdown_body was passed to save_axiom
        call_args = mock_mgr.save_axiom.call_args[1]
        self.assertEqual(call_args["markdown_body"], doc_text)
        print("Verification: markdown_body correctly propagated to AxiomsManager.")

    async def test_real_save_axiom_file_content(self):
        from graph_rlm.backend.src.mcp_integration.skill_storage import AxiomsManager

        test_axioms_dir = project_root / "graph_rlm" / "backend" / "axioms_test_docs"
        test_axioms_dir.mkdir(parents=True, exist_ok=True)

        mgr = AxiomsManager(test_axioms_dir)

        doc_body = "# Test Doc\n\nThis is a high-fidelity documentation test.\n\n## Section\nContent."
        code = "def test_doc_axiom():\n    return True"

        try:
            name = await mgr.save_axiom(
                name="test-doc-axiom",
                code=code,
                description="Test documentation persistence",
                markdown_body=doc_body,
            )

            skill_md_path = test_axioms_dir / name / "SKILL.md"
            content = skill_md_path.read_text()

            print(f"Generated SKILL.md content:\n{content}")

            self.assertIn(doc_body, content)
            self.assertIn("origin: dreamer", content)

        finally:
            import shutil

            shutil.rmtree(test_axioms_dir)


if __name__ == "__main__":
    unittest.main()
