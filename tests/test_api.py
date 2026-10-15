# tests/test_api.py

from pathlib import Path

import pytest
from fastapi import HTTPException
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
    assert "artifact_directory" not in payload
    assert str(tmp_path) not in response.text


def test_create_run_failure_cleans_partial_artifacts(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        api,
        "run_report_pipeline",
        lambda **kwargs: _fake_result(),
    )

    def fail_after_partial_write(**kwargs):
        partial_dir = kwargs["output_root"] / kwargs["run_id"]
        partial_dir.mkdir(parents=True)
        (partial_dir / "partial.json").write_text(
            "{}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Internal failure at {tmp_path}/private/path"
        )

    monkeypatch.setattr(
        api,
        "write_run_artifacts",
        fail_after_partial_write,
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

    assert response.status_code == 500
    assert "private/path" not in response.text
    assert str(tmp_path) not in response.text

    input_dir = output_root / "_inputs"
    assert not list(input_dir.glob("*.csv"))
    assert not [
        path for path in output_root.iterdir()
        if path.is_dir() and path.name != "_inputs"
    ]


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


def test_cleanup_logs_run_directory_removal_failure(
    monkeypatch,
    tmp_path: Path,
    caplog,
):
    dataset_path = tmp_path / "_inputs" / "upload.csv"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("synthetic upload", encoding="utf-8")

    run_directory = tmp_path / "outputs" / "run-test"
    run_directory.mkdir(parents=True)
    (run_directory / "partial.json").write_text(
        "{}",
        encoding="utf-8",
    )

    def fail_rmtree(_path):
        raise PermissionError("synthetic deletion failure")

    monkeypatch.setattr(api.shutil, "rmtree", fail_rmtree)

    api._cleanup_failed_run(dataset_path, run_directory)

    assert not dataset_path.exists()
    assert run_directory.exists()
    assert "Failed-run artifact cleanup failed (PermissionError)." in caplog.text
    assert "synthetic deletion failure" not in caplog.text


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "../../etc/passwd",
        "/tmp/outside",
        "nested/run",
        "nested\\\\run",
        "%2e%2e",
    ],
)
def test_run_directory_rejects_unsafe_path_components(
    monkeypatch,
    tmp_path: Path,
    run_id: str,
):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )

    with pytest.raises(HTTPException) as exc_info:
        api._run_directory(run_id)

    assert exc_info.value.status_code == 404


def test_create_run_rejects_oversized_upload(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        api,
        "API_OUTPUT_ROOT",
        tmp_path / "outputs",
    )
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/runs",
        files={
            "file": (
                "synthetic.csv",
                b"x" * 11,
                "text/csv",
            )
        },
    )

    assert response.status_code == 413
    assert not list(
        (tmp_path / "outputs" / "_inputs").glob("*.csv")
    )


def test_create_run_invalid_csv_returns_generic_400(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.post(
        "/runs",
        files={
            "file": (
                "invalid.csv",
                b"not,a,valid\n1,2,3\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded CSV could not be processed. "
        "Check its format and contents."
    )
    assert str(tmp_path) not in response.text
    assert not list((output_root / "_inputs").glob("*.csv"))


def test_create_run_internal_value_error_returns_500(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    def fail_internally(**kwargs):
        raise ValueError(f"Internal failure at {tmp_path}/private/path")

    monkeypatch.setattr(api, "run_report_pipeline", fail_internally)

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

    assert response.status_code == 500
    assert "private/path" not in response.text
    assert str(tmp_path) not in response.text
    assert not list((output_root / "_inputs").glob("*.csv"))


def test_run_directory_rejects_symlink_escape(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    outside = tmp_path / "outside"
    output_root.mkdir()
    outside.mkdir()
    (output_root / "safe-run").symlink_to(
        outside,
        target_is_directory=True,
    )
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    with pytest.raises(HTTPException) as exc_info:
        api._run_directory("safe-run")

    assert exc_info.value.status_code == 404


def test_run_directory_rejects_internal_symlink_alias(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    existing_run = output_root / "existing-run"
    existing_run.mkdir(parents=True)

    (output_root / "alias-run").symlink_to(
        existing_run,
        target_is_directory=True,
    )
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    with pytest.raises(HTTPException) as exc_info:
        api._run_directory("alias-run")

    assert exc_info.value.status_code == 404


def test_read_json_missing_artifact_returns_404(tmp_path: Path):
    with pytest.raises(HTTPException) as exc_info:
        api._read_json(tmp_path / "missing.json")

    assert exc_info.value.status_code == 404


def test_read_json_malformed_json_returns_controlled_500(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{invalid", encoding="utf-8")

    with pytest.raises(HTTPException) as exc_info:
        api._read_json(artifact)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Artifact 'artifact.json' is invalid JSON."


def test_read_json_invalid_utf8_returns_controlled_500(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(HTTPException) as exc_info:
        api._read_json(artifact)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Artifact 'artifact.json' is invalid JSON."


@pytest.mark.parametrize(
    ("filename", "content", "expected_detail"),
    [
        ("notes.txt", b"not csv", "Only CSV files are accepted."),
    ],
)
def test_create_run_rejects_invalid_filename(
    monkeypatch,
    tmp_path: Path,
    filename: str,
    content: bytes,
    expected_detail: str,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.post(
        "/runs",
        files={"file": (filename, content, "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
    assert not list((output_root / "_inputs").glob("*.csv"))


def test_create_run_rejects_empty_upload(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.post(
        "/runs",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded CSV file is empty."
    assert not list((output_root / "_inputs").glob("*.csv"))


def test_create_run_without_file_returns_400(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.post("/runs")

    assert response.status_code == 400
    assert response.json()["detail"] == "A CSV file is required."
    assert not list((output_root / "_inputs").glob("*.csv"))


def test_missing_run_returns_404(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", tmp_path / "outputs")

    response = client.get("/runs/missingrun123")

    assert response.status_code == 404


def test_report_endpoint_returns_404_when_artifact_missing(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    run_id = "missingreport123"
    (output_root / run_id).mkdir(parents=True)
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.get(f"/runs/{run_id}/report")

    assert response.status_code == 404
    assert response.json()["detail"] == "Report artifact was not found."


def test_manifest_endpoint_returns_controlled_500_for_corrupt_json(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    run_id = "badmanifest123"
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "manifest.json").write_text(
        "{invalid",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.get(f"/runs/{run_id}/manifest")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Artifact 'manifest.json' is invalid JSON."
    )
    assert str(tmp_path) not in response.text


def test_narratives_endpoint_rejects_non_object_json(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "outputs"
    run_id = "badnarratives123"
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "narratives.json").write_text(
        "[]",
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "API_OUTPUT_ROOT", output_root)

    response = client.get(f"/runs/{run_id}/narratives")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Narrative artifact must contain a JSON object."
    )
    assert str(tmp_path) not in response.text


def test_build_generation_config_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = api._build_generation_config()

    assert config.provider == "groq"
    assert config.model == "openai/gpt-oss-120b"
    assert config.temperature == 0.0
    assert config.max_tokens == 1000


def test_build_generation_config_normalizes_openai_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " OpenAI ")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")

    config = api._build_generation_config()

    assert config.provider == "openai"
    assert config.model == "unit-test-model"


def test_build_generation_config_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        api._build_generation_config()


def test_build_llm_provider_passes_config_to_factory(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    marker = object()
    received = {}

    def fake_factory(**kwargs):
        received.update(kwargs)
        return marker

    monkeypatch.setattr(api, "build_provider", fake_factory)

    assert api._build_llm_provider() is marker
    assert received == {
        "provider": "openai",
        "model": "unit-test-model",
        "max_tokens": 1000,
    }


def test_build_llm_provider_rejects_unconfigured_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(api, "build_provider", lambda **kwargs: None)

    with pytest.raises(ValueError, match="not configured"):
        api._build_llm_provider()
