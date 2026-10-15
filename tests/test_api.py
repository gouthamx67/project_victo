# tests/test_api.py

from pathlib import Path

from fastapi.testclient import TestClient

import src.api as api
from src.analysis_framework import AnalysisResult
from src.evidence_trace import build_evidence_trace
from src.llm_contract import LLMNarrative
from src.report_config import ReportConfig
from src.report_pipeline import (
    PipelineSection,
    ReportPipelineResult,
    ReportStatus,
)
from src.report_review import make_initial_review_set
from src.review import ReviewStatus


client = TestClient(api.app)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "fixtures" / "synthetic_icsr.csv"


def _fake_result() -> ReportPipelineResult:
    config = ReportConfig.model_validate(
        {
            "report": {
                "id": "pader",
                "title": "Periodic Adverse Drug Experience Report",
                "version": "1.0.0",
                "regulatory_basis": {
                    "jurisdiction": "US",
                    "reference": "21 CFR 314.80(c)(2)",
                },
                "sections": [
                    {
                        "id": "executive_summary",
                        "title": "Executive Summary",
                        "order": 1,
                        "needs": ["case_count"],
                        "ai": False,
                        "instructions": "Deterministic test section.",
                        "style": {
                            "tone": "formal",
                            "length": "short",
                        },
                    }
                ],
            }
        }
    )

    analysis = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Unique safety-report cases.",
        population="case",
        supporting_ids=["CASE-001"],
        version="1.0.0",
    )

    context = api._build_section_context(
        config=config,
        section=config.report.sections[0],
        results=[analysis],
    )

    evidence = build_evidence_trace(
        section_id="executive_summary",
        metric_ids=["case_count"],
        analyses=[analysis],
    )

    section = PipelineSection(
        section_id="executive_summary",
        title="Executive Summary",
        ai_enabled=False,
        context=context,
        narrative=LLMNarrative(text="{case_count}"),
        narrative_template="{case_count}",
        evidence=evidence,
    )

    return ReportPipelineResult(
        config=config,
        normalized=None,
        analyses=[analysis],
        sections=[section],
        review_set=make_initial_review_set(config),
        status=ReportStatus.DRAFT,
    )


def _fake_write_run_artifacts(
    *,
    result,
    run_id,
    dataset_path,
    report_config_path,
    prompt_path,
    generation_config,
    template_path,
    output_root,
):
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    (run_directory / "report.html").write_text(
        "<html><body>DRAFT</body></html>",
        encoding="utf-8",
    )

    (run_directory / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (run_directory / "review.json").write_text(
        result.review_set.model_dump_json(),
        encoding="utf-8",
    )

    (run_directory / "analyses.json").write_text(
        "[]",
        encoding="utf-8",
    )

    (run_directory / "evidence.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (run_directory / "narratives.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "regulatory-safety-reporting-api"


def test_create_run(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    monkeypatch.setattr(
        api,
        "run_report_pipeline",
        lambda **kwargs: _fake_result(),
    )

    monkeypatch.setattr(
        api,
        "write_run_artifacts",
        _fake_write_run_artifacts,
    )

    response = client.post(
        "/runs",
        files={
            "file": (
                DATASET.name,
                DATASET.read_bytes(),
                "text/csv",
            )
        },
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["status"] == "DRAFT"
    assert "run_id" in payload


def test_get_run(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-001"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.get(f"/runs/{run_id}")

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == run_id
    assert payload["status"] == "DRAFT"


def test_get_report(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-002"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.get(f"/runs/{run_id}/report")

    assert response.status_code == 200
    assert "DRAFT" in response.text


def test_get_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-003"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.get(f"/runs/{run_id}/manifest")

    assert response.status_code == 200
    assert response.json() == {}


def test_get_review(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-004"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.get(f"/runs/{run_id}/review")

    assert response.status_code == 200

    payload = response.json()

    review = next(
        item
        for item in payload["reviews"]
        if item["section_id"] == "executive_summary"
    )

    assert review["status"] == ReviewStatus.UNREVIEWED


def test_get_evidence(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-evidence"

    run_directory = tmp_path / "outputs" / run_id
    run_directory.mkdir(parents=True)

    evidence = {
        "executive_summary": [
            {
                "metric_id": "case_count",
                "supporting_ids": ["CASE-001"],
            }
        ]
    }

    import json

    (run_directory / "evidence.json").write_text(
        json.dumps(evidence),
        encoding="utf-8",
    )

    response = client.get(f"/runs/{run_id}/evidence")

    assert response.status_code == 200
    assert response.json() == evidence


def test_review_action_approve(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-review-approve"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.post(
        f"/runs/{run_id}/review",
        json={
            "section_id": "executive_summary",
            "status": "APPROVED",
            "reviewer": "reviewer-1",
            "comment": "Approved.",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["section_id"] == "executive_summary"
    assert payload["status"] == "APPROVED"
    assert payload["reviewer"] == "reviewer-1"


def test_review_action_flag(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-review-flag"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.post(
        f"/runs/{run_id}/review",
        json={
            "section_id": "executive_summary",
            "status": "FLAGGED",
            "reviewer": "reviewer-1",
            "comment": "Needs review.",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["section_id"] == "executive_summary"
    assert payload["status"] == "FLAGGED"
    assert payload["reviewer"] == "reviewer-1"
    assert payload["comment"] == "Needs review."


def test_review_action_requires_comment_for_flag(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-review-flag-invalid"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.post(
        f"/runs/{run_id}/review",
        json={
            "section_id": "executive_summary",
            "status": "FLAGGED",
            "reviewer": "reviewer-1",
            "comment": "",
        },
    )

    assert response.status_code == 400


def test_finalize_rejects_unreviewed_sections(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    run_id = "run-finalize-unreviewed"

    _fake_write_run_artifacts(
        result=_fake_result(),
        run_id=run_id,
        dataset_path=tmp_path / "dataset.csv",
        report_config_path=tmp_path / "config.yaml",
        prompt_path=tmp_path / "prompt.txt",
        generation_config=None,
        template_path=tmp_path / "template.html",
        output_root=tmp_path / "outputs",
    )

    response = client.post(f"/runs/{run_id}/finalize")

    assert response.status_code == 409


def test_finalize_approved_run_returns_final(
    monkeypatch,
    tmp_path: Path,
):
    """
    Regression test: finalizing an approved run must reconstruct the
    persisted report artifacts (including section provenance) and render
    the FINAL report instead of raising.
    """

    import json

    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    result = _fake_result()
    run_id = "run-finalize-approved"

    monkeypatch.setattr(
        api,
        "_load_report_config",
        lambda: result.config,
    )

    run_directory = tmp_path / "outputs" / run_id
    run_directory.mkdir(parents=True)

    section = result.sections[0]

    (run_directory / "narratives.json").write_text(
        json.dumps(
            {
                section.section_id: {
                    "narrative": {"text": section.narrative_template},
                    "narrative_template": section.narrative_template,
                    "ai_enabled": False,
                    "provider_id": "deterministic",
                    "model_id": "not_applicable",
                    "grounding_status": "not_recorded",
                    "generation_source": "deterministic_template",
                    "rendered_prompt_hash": None,
                }
            }
        ),
        encoding="utf-8",
    )

    (run_directory / "analyses.json").write_text(
        json.dumps([analysis.model_dump() for analysis in result.analyses]),
        encoding="utf-8",
    )

    (run_directory / "evidence.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (run_directory / "review.json").write_text(
        result.review_set.model_dump_json(),
        encoding="utf-8",
    )

    approve = client.post(
        f"/runs/{run_id}/review",
        json={
            "section_id": section.section_id,
            "status": "APPROVED",
            "reviewer": "reviewer-1",
            "comment": "Approved.",
        },
    )

    assert approve.status_code == 200

    response = client.post(f"/runs/{run_id}/finalize")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "FINAL"

    report_html = (run_directory / "report.html").read_text(
        encoding="utf-8"
    )

    assert "FINAL — HUMAN REVIEW APPROVED" in report_html