from oncolncai.demo import run_offline_agent_demo
from oncolncai.ui import build_demo_run_view


def test_demo_view_exposes_agent_process_without_fake_science() -> None:
    report = run_offline_agent_demo("lung adenocarcinoma")
    view = build_demo_run_view(
        "Which lncRNAs are promising biomarkers for lung adenocarcinoma?", report
    )

    assert ("run_de", "skipped") in view.plan_rows
    assert "TCGA-LUAD" in view.overview
    assert "Observation-driven replanning" in view.replanning
    assert "Not evaluated" in view.evidence
    assert "Not evaluated" in view.computed_results
    assert "No biomedical conclusion" in view.final_report
    assert "not a biomedical finding" in view.limitations
    assert all("Thought:" not in summary for _, _, summary in view.trace_rows)
