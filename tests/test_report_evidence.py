from pathlib import Path

from src.evidence_trace import (
    EvidenceTrace,
    render_evidence_html,
)
from src.report_pipeline import (
    render_html_report,
    run_report_pipeline,
)


DATASET = Path(
    "data/Bisoprolol_icsr_sample_1068rows (1).csv"
)

PADER_CONFIG = Path(
    "configs/pader.yaml"
)

PSUR_CONFIG = Path(
    "configs/psur_lite.yaml"
)

PROMPT = Path(
    "prompts/sections/default.txt"
)

TEMPLATE = Path(
    "prompts/report_template.html"
)


def test_evidence_html_contains_metric_ids() -> None:
    trace = EvidenceTrace(
        section_id="executive_summary",
        metric_ids=[
            "case_count",
            "fatal_case_count",
        ],
        supporting_ids={
            "case_count": [
                "CASE-1",
                "CASE-2",
            ],
            "fatal_case_count": [
                "CASE-9",
            ],
        },
    )

    html = render_evidence_html(trace)

    assert "case_count" in html
    assert "fatal_case_count" in html
    assert "CASE-1" in html
    assert "CASE-2" in html
    assert "CASE-9" in html


def test_evidence_html_handles_empty_supporting_ids() -> None:
    trace = EvidenceTrace(
        section_id="example",
        metric_ids=["example_metric"],
        supporting_ids={
            "example_metric": [],
        },
    )

    html = render_evidence_html(trace)

    assert "example_metric" in html
    assert "Supporting IDs:</strong> None" in html


def test_evidence_html_escapes_html() -> None:
    trace = EvidenceTrace(
        section_id="example",
        metric_ids=["metric<script>"],
        supporting_ids={
            "metric<script>": [
                "CASE-<script>alert(1)</script>",
            ],
        },
    )

    html = render_evidence_html(trace)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_pipeline_sections_have_evidence() -> None:
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=PADER_CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    assert result.sections

    for section in result.sections:
        assert section.evidence.section_id == (
            section.section_id
        )

        assert section.evidence.metric_ids == [
            metric.metric_id
            for metric in section.context.metrics
        ]


def test_rendered_html_contains_evidence() -> None:
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=PADER_CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    html = render_html_report(
        result=result,
        template_path=TEMPLATE,
    )

    assert "Evidence Trace" in html
    assert "case_count" in html
    assert "Supporting IDs" in html


def test_psur_lite_html_also_contains_evidence() -> None:
    result = run_report_pipeline(
        csv_path=DATASET,
        config_path=PSUR_CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )

    html = render_html_report(
        result=result,
        template_path=TEMPLATE,
    )

    assert "Evidence Trace" in html
    assert "case_count" in html