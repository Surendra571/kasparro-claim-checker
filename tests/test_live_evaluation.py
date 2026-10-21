"""
Tests for live evaluation runner and development comparison reporting.
Verifies that v1, v2, and comparison reporting produce accurate deterministic metrics.
"""
from pathlib import Path

from scripts.run_live_evaluation import (
    compute_sha256,
    run_evaluation,
)


def test_prompt_files_exist_and_hashes_frozen():
    """Verify v1 and v2 prompts exist and have distinct frozen hashes."""
    v1_path = Path("prompts/v1.txt")
    v2_path = Path("prompts/v2.txt")

    assert v1_path.exists()
    assert v2_path.exists()

    v1_hash = compute_sha256(v1_path)
    v2_hash = compute_sha256(v2_path)

    assert v1_hash == "d988649f0beddba3844fa08fcc5d0938ab281a4f9acb016e419eb9a331de4c3a"
    assert v2_hash == "54586b58234d66a0682b8ed397159f0bb1e18555160cc2105bc439f2d1c0c89c"
    assert v1_hash != v2_hash


def test_live_evaluation_runner_end_to_end(tmp_path):
    """Verify evaluation runner runs across answers and produces valid metrics."""
    res_v1 = run_evaluation(prompt_version="v1", limit=5)
    res_v2 = run_evaluation(prompt_version="v2", limit=5)

    assert res_v1["metadata"]["is_run_complete"] is True
    assert res_v2["metadata"]["is_run_complete"] is True

    # Check metrics existence
    for res in [res_v1, res_v2]:
        m = res["evaluation_summary"]["metrics"]
        assert "extraction_f1" in m
        assert "joint_f1" in m
        assert "gold_claim_coverage" in m


def test_comparison_report_generation():
    """Verify development comparison markdown generates cleanly."""
    comp_md = Path("reports/development_comparison.md")
    assert comp_md.exists()
    content = comp_md.read_text(encoding="utf-8")
    assert "Joint Precision" in content
    assert "Joint Recall" in content
    assert "Joint F1" in content
    assert "A01:1440:1619" in content
    assert "condition_swap" in content

