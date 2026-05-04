import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_rlm.backend.src.slac_framework import main as slac_main


async def test_slac_unified():
    print("Testing Unified SLAC Framework...")

    test_cases = [
        {
            "name": "Initial Concept",
            "args": {
                "concept_data": {
                    "truth": 0.5,
                    "flaws": [],
                    "improvements": [],
                    "stage": "C",
                    "text": "The project aims to unify SLAC fragments."
                },
                "alpha": 0.5,
                "beta": 0.5
            }
        },
        {
            "name": "Advanced Scrutiny with Future Logic",
            "args": {
                "concept_data": {
                    "truth": 0.7,
                    "flaws": [
                        {"desc": "Potential fragmentation risk (F)", "impact": 0.5},
                        {"desc": "Missing prior axioms", "impact": 0.2}
                    ],
                    "improvements": [
                        {"desc": "Consolidated into slac_framework.py", "impact": 0.8}
                    ],
                    "stage": "S",
                    "text": "The system will always (G) resolve fragmentation."
                },
                "alpha": 0.4,
                "beta": 0.6
            }
        }
    ]

    for case in test_cases:
        print(f"\n--- Case: {case['name']} ---")
        result = await slac_main(case['args'])
        print(result["meter"])
        print(f"Status: {result['status']}")
        print(f"Temporal Audit: {result['temporal_audit']}")

        # Simple assertions
        assert "at_score" in result
        assert "meter" in result
        assert "temporal_audit" in result

if __name__ == "__main__":
    asyncio.run(test_slac_unified())
