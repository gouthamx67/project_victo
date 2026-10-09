from pathlib import Path
from src.analysis_framework import AnalysisResult
from src.report_pipeline import _analysis_to_metric_context

from src.report_pipeline import (
    ReportStatus,
    render_html_report,
    run_report_pipeline,
)
from src.review import ReviewStatus


DATASET = Path(
    "data/Bisoprolol_icsr_sample_1068rows (1).csv"
)

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