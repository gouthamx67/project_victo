from __future__ import annotations

import json
from pathlib import Path

from src.llm_provider import LLMGenerationConfig
from src.report_pipeline import run_report_pipeline
from src.review import ReviewStatus
from src.run_artifacts import write_run_artifacts


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "fixtures" / "synthetic_icsr.csv"
CONFIG = ROOT / "configs" / "pader.yaml"
PROMPT = ROOT / "prompts" / "sections" / "default.txt"
TEMPLATE = ROOT / "prompts" / "report_template.html"


def _run_result():
    return run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        provider=None,
        use_llm=False,
    )


def _generation_config() -> LLMGenerationConfig:
    return LLMGenerationConfig(
        provider="none",
        model="none",
        temperature=0.0,
        max_tokens=1000,
    )


def test_writes_complete_run_bundle(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="test-run",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    assert run_directory == tmp_path / "test-run"

    assert sorted(path.name for path in run_directory.iterdir()) == [
        "analyses.json",
        "evidence.json",
        "manifest.json",
        "narratives.json",
        "report.html",
        "review.json",
    ]


def test_manifest_matches_run(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="manifest-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    manifest = json.loads(
        (run_directory / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["run_id"] == "manifest-test"
    assert manifest["dataset_path"] == str(DATASET)
    assert manifest["report_id"] == "pader"
    assert manifest["report_version"] == "1.0.0"
    assert manifest["analysis_registry_version"] == "1.0.0"
    assert manifest["provider"] == "none"
    assert manifest["model"] == "none"


def test_review_starts_unreviewed(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="review-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    review_payload = json.loads(
        (run_directory / "review.json").read_text(encoding="utf-8")
    )

    assert review_payload["reviews"]
    assert all(
        review["status"] == ReviewStatus.UNREVIEWED
        for review in review_payload["reviews"]
    )


def test_analyses_artifact_contains_metric_results(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="analysis-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    analyses = json.loads(
        (run_directory / "analyses.json").read_text(encoding="utf-8")
    )

    metric_ids = {
        analysis["metric_id"]
        for analysis in analyses
    }

    assert "case_count" in metric_ids
    assert "serious_case_count" in metric_ids
    assert "reaction_count" in metric_ids


def test_evidence_artifact_contains_all_sections(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="evidence-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    evidence = json.loads(
        (run_directory / "evidence.json").read_text(encoding="utf-8")
    )

    section_ids = {
        section["section_id"]
        for section in evidence
    }

    expected_section_ids = {
        section.section_id
        for section in result.sections
    }

    assert section_ids == expected_section_ids


def test_report_artifact_matches_draft_status(tmp_path: Path):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="report-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    report_html = (
        run_directory / "report.html"
    ).read_text(encoding="utf-8")

    assert 'meta name="report-status" content="DRAFT"' in report_html
    assert "DRAFT — HUMAN REVIEW REQUIRED" in report_html
    assert "FINAL — HUMAN REVIEW APPROVED" not in report_html


def test_narratives_artifact_contains_renderable_section_content(
    tmp_path: Path,
):
    result = _run_result()

    run_directory = write_run_artifacts(
        result=result,
        run_id="narrative-test",
        dataset_path=DATASET,
        report_config_path=CONFIG,
        prompt_path=PROMPT,
        generation_config=_generation_config(),
        template_path=TEMPLATE,
        output_root=tmp_path,
    )

    narratives = json.loads(
        (run_directory / "narratives.json").read_text(
            encoding="utf-8"
        )
    )

    expected_section_ids = {
        section.section_id
        for section in result.sections
    }

    assert set(narratives) == expected_section_ids

    for section_id, payload in narratives.items():
        assert set(payload) == {
            "narrative",
            "narrative_template",
            "ai_enabled",
            "provider_id",
            "model_id",
            "grounding_status",
            "generation_source",
            "rendered_prompt_hash",
        }

        assert isinstance(payload["ai_enabled"], bool)

        narrative = payload["narrative"]

        if isinstance(narrative, dict):
            assert set(narrative) == {"text"}
            assert isinstance(narrative["text"], str)
        else:
            assert isinstance(narrative, str)

        assert isinstance(payload["narrative_template"], str)