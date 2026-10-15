# tests/test_report_pipeline.py
import json
from dataclasses import replace

import pytest

from src.analysis_framework import AnalysisResult
from src.report_pipeline import (
    DatasetInputError,
    _analysis_to_metric_context,
)

from pathlib import Path

from src.report_pipeline import (
    ReportStatus,
    render_html_report,
    run_report_pipeline,
)
from src.review import ReviewStatus


DATASET = Path("tests/fixtures/synthetic_icsr.csv")

CONFIG = Path(
    "configs/pader.yaml"
)

PROMPT = Path(
    "prompts/sections/default.txt"
)

TEMPLATE = Path(
    "prompts/report_template.html"
)


def test_pipeline_result_starts_as_draft():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    assert result.status == ReportStatus.DRAFT


def test_pipeline_creates_unreviewed_review_for_every_section():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    section_ids = [
        section.section_id
        for section in result.sections
    ]

    review_ids = [
        review.section_id
        for review in result.review_set.reviews
    ]

    assert review_ids == section_ids

    assert all(
        review.status == ReviewStatus.UNREVIEWED
        for review in result.review_set.reviews
    )


def test_pipeline_review_count_matches_section_count():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    assert len(result.review_set.reviews) == len(
        result.sections
    )


def test_pipeline_starts_with_no_reviewer_information():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    assert all(
        review.reviewer is None
        for review in result.review_set.reviews
    )

    assert all(
        review.reviewed_at is None
        for review in result.review_set.reviews
    )


def test_rendered_report_contains_metric_evidence_links():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    html = render_html_report(
        result,
        TEMPLATE,
    )

    assert 'class="metric-evidence-link"' in html
    assert 'href="#evidence-case_count"' in html
    assert 'data-metric-id="case_count"' in html


def test_rendered_report_contains_matching_evidence_anchor():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    html = render_html_report(
        result,
        TEMPLATE,
    )

    assert 'id="evidence-case_count"' in html
    assert 'href="#evidence-case_count"' in html


def test_trend_method_note_added_when_narrative_omits_method():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    sections = [
        replace(
            section,
            narrative_template=(
                "Review flags are reported in "
                "{monthly_case_deviations}."
            ),
        )
        if section.section_id == "trend_review"
        else section
        for section in result.sections
    ]

    html = render_html_report(
        replace(result, sections=sections),
        TEMPLATE,
    )

    assert 'class="trend-method-note"' in html
    assert "1.5 times the eligible full-month mean" in html


def test_trend_method_note_skipped_when_template_explains_method():
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    html = render_html_report(
        result,
        TEMPLATE,
    )

    assert 'class="trend-method-note"' not in html
    assert "excluded from the full-month baseline" in html


def test_llm_context_does_not_include_supporting_case_ids():
    analysis = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of unique safety cases.",
        population="case",
        supporting_ids=[
            f"CASE-{index}"
            for index in range(5000)
        ],
        version="1.0.0",
    )

    context = _analysis_to_metric_context(analysis)

    assert context.metric_id == "case_count"
    assert context.value == 1024
    assert context.supporting_ids == []


def test_llm_prompt_size_does_not_scale_with_supporting_case_ids():
    analysis = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of unique safety cases.",
        population="case",
        supporting_ids=[
            f"CASE-{index}"
            for index in range(10000)
        ],
        version="1.0.0",
    )

    metric_context = _analysis_to_metric_context(analysis)

    assert metric_context.supporting_ids == []

    prompt_context = {
        "report_id": "pader",
        "report_version": "1.0.0",
        "section_id": "case_volume",
        "section_title": "Case Volume",
        "instructions": "Summarize the approved case volume metric.",
        "style": {
            "tone": "regulatory",
            "length": "concise",
        },
        "metrics": [metric_context.model_dump()],
    }

    serialized = json.dumps(
        prompt_context,
        indent=2,
        sort_keys=True,
    )

    assert len(serialized) < 5000
    assert "CASE-9999" not in serialized


def test_pipeline_invalid_dataset_raises_dataset_input_error(
    tmp_path: Path,
):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("unexpected\nvalue\n", encoding="utf-8")

    with pytest.raises(DatasetInputError):
        run_report_pipeline(
            csv_path=csv_path,
            config_path=CONFIG,
            prompt_path=PROMPT,
            use_llm=False,
        )


def test_pipeline_missing_dataset_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_report_pipeline(
            csv_path=tmp_path / "missing.csv",
            config_path=CONFIG,
            prompt_path=PROMPT,
            use_llm=False,
        )


def test_pipeline_wraps_ingestion_value_error(
    monkeypatch,
):
    import src.ingest

    def fail_ingestion(_path):
        raise ValueError("synthetic parser error")

    monkeypatch.setattr(
        src.ingest,
        "ingest_and_validate",
        fail_ingestion,
    )

    with pytest.raises(
        DatasetInputError,
        match="Uploaded CSV could not be parsed or validated",
    ) as exc_info:
        run_report_pipeline(
            csv_path=DATASET,
            config_path=CONFIG,
            prompt_path=PROMPT,
            use_llm=False,
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
