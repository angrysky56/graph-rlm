#!/usr/bin/env python3
"""
Test script to verify the report data pipeline fix.
Tests that kernel results can be retrieved from DB and used for report generation.
"""

import sys

sys.path.insert(0, "/home/ty/Repositories/ai_workspace/graph-rlm/graph_rlm/backend/src")

from core.db import db


def test_kernel_results_retrieval():
    """Test that get_kernel_results() works."""
    print("Testing kernel results retrieval...")

    # Test with a sample session ID
    test_session = "test-session-123"

    # Call the new function
    results = db.get_kernel_results(test_session)

    # Verify structure
    assert isinstance(results, dict), "Results should be a dictionary"
    assert "status" in results, "Results should have status field"
    assert "sheaf_scores" in results, "Results should have sheaf_scores field"
    assert "spectral_energies" in results, "Results should have spectral_energies field"
    assert "h0_ranks" in results, "Results should have h0_ranks field"
    assert "avg_sheaf_score" in results, "Results should have avg_sheaf_score field"
    assert (
        "avg_spectral_energy" in results
    ), "Results should have avg_spectral_energy field"
    assert "avg_h0_rank" in results, "Results should have avg_h0_rank field"
    assert "kernel_basis" in results, "Results should have kernel_basis field"

    print("✓ get_kernel_results() returns correct structure")


def test_session_report_data():
    """Test that get_session_report_data() works."""
    print("\nTesting session report data generation...")

    test_session = "test-session-456"

    # Call the new function
    report_data = db.get_session_report_data(test_session)

    # Verify structure
    assert isinstance(report_data, dict), "Report data should be a dictionary"
    assert "session_id" in report_data, "Report data should have session_id"
    assert "kernel_results" in report_data, "Report data should have kernel_results"
    assert "thought_count" in report_data, "Report data should have thought_count"
    assert "paper_title" in report_data, "Report data should have paper_title"
    assert "timestamp" in report_data, "Report data should have timestamp"

    print("✓ get_session_report_data() returns correct structure")


def test_template_population():
    """Test that report data can be used to populate templates."""
    print("\nTesting template population...")

    # Simulate report generation workflow
    session_id = "demo-session-789"
    report_data = db.get_session_report_data(session_id)

    # Simulate template variables
    template_vars = {
        "paper_title": report_data["paper_title"],
        "session_id": report_data["session_id"],
        "kernel_basis": report_data["kernel_results"]["kernel_basis"],
        "avg_sheaf_score": report_data["kernel_results"]["avg_sheaf_score"],
        "avg_spectral_energy": report_data["kernel_results"]["avg_spectral_energy"],
        "avg_h0_rank": report_data["kernel_results"]["avg_h0_rank"],
        "thought_count": report_data["thought_count"],
        "timestamp": report_data["timestamp"],
    }

    print("✓ Template variables generated successfully:")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Report Data Pipeline Fix")
    print("=" * 60)

    try:
        test_kernel_results_retrieval()
        test_session_report_data()
        test_template_population()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Report data pipeline is working!")
        print("=" * 60)

        print("\nUsage example for agents:")
        print("```python")
        print("# Get kernel computation results")
        print("kernel_data = await rlm.get_kernel_results()")
        print("print(f'Average sheaf score: {kernel_data[\"avg_sheaf_score\"]}')")
        print("")
        print("# Generate complete report data")
        print("report_data = await rlm.generate_report_data('My Analysis')")
        print("print(f'Paper: {report_data[\"paper_title\"]}')")
        print("```")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
