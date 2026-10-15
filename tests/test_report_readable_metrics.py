from src.analysis_framework import AnalysisResult
from src.report_renderer import (
    format_metric_value,
    render_narrative,
)


def test_percentage_is_human_readable() -> None:
    result = AnalysisResult(
        metric_id="serious_case_percentage",
        value=99.90234375,
        definition="Percentage of serious cases.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    assert format_metric_value(
        result.metric_id,
        result.value,
    ) == "99.9%"


def test_monthly_case_counts_are_not_rendered_as_python_dicts() -> None:
    result = AnalysisResult(
        metric_id="monthly_case_counts",
        value=[
            {
                "reporting_month": "2024-12",
                "case_count": 21,
            },
            {
                "reporting_month": "2025-01",
                "case_count": 75,
            },
        ],
        definition="Monthly unique case counts.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    rendered = format_metric_value(
        result.metric_id,
        result.value,
    )

    assert rendered == (
        "2024-12: 21 cases; "
        "2025-01: 75 cases"
    )

    assert "'reporting_month'" not in rendered
    assert "'case_count'" not in rendered


def test_monthly_deviations_do_not_dump_every_record() -> None:
    result = AnalysisResult(
        metric_id="monthly_case_deviations",
        value=[
            {
                "reporting_month": "2024-12",
                "case_count": 21,
                "mean_monthly_case_count": 78.7,
                "deviation_threshold": 118.1,
                "deviation_flag": False,
            },
            {
                "reporting_month": "2025-10",
                "case_count": 102,
                "mean_monthly_case_count": 78.7,
                "deviation_threshold": 118.1,
                "deviation_flag": False,
            },
        ],
        definition="Monthly deviation review.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    rendered = format_metric_value(
        result.metric_id,
        result.value,
    )

    assert rendered == (
        "the absence of monthly deviations "
        "above the configured threshold"
    )

    assert "'reporting_month'" not in rendered
    assert "'deviation_flag'" not in rendered


def test_monthly_deviations_summarize_flagged_months() -> None:
    result = AnalysisResult(
        metric_id="monthly_case_deviations",
        value=[
            {
                "reporting_month": "2024-12",
                "case_count": 21,
                "mean_monthly_case_count": 78.7,
                "deviation_threshold": 118.1,
                "deviation_flag": False,
            },
            {
                "reporting_month": "2025-03",
                "case_count": 402,
                "mean_monthly_case_count": 78.7,
                "deviation_threshold": 118.1,
                "deviation_flag": True,
            },
        ],
        definition="Monthly deviation review.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    rendered = format_metric_value(
        result.metric_id,
        result.value,
    )

    assert rendered == (
        "monthly deviations flagged for human review in 2025-03 "
        "(descriptive flags, not confirmed signals)"
    )


def test_narrative_uses_readable_metric_values() -> None:
    results = [
        AnalysisResult(
            metric_id="case_count",
            value=1024,
            definition="Number of cases.",
            population="case",
            supporting_ids=[],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="serious_case_percentage",
            value=99.90234375,
            definition="Serious percentage.",
            population="case",
            supporting_ids=[],
            version="1.0.0",
        ),
    ]

    rendered = render_narrative(
        "There were {case_count} cases and "
        "{serious_case_percentage} were serious.",
        results,
    )

    assert rendered == (
        "There were 1,024 cases and "
        "99.9% were serious."
    )


def test_case_listing_is_not_dumped_into_narrative() -> None:
    result = AnalysisResult(
        metric_id="case_listing",
        value=[
            {
                "safetyreportid": "CASE-1",
            },
            {
                "safetyreportid": "CASE-2",
            },
        ],
        definition="Deterministic case listing.",
        population="case",
        supporting_ids=["CASE-1", "CASE-2"],
        version="1.0.0",
    )

    rendered = format_metric_value(
        result.metric_id,
        result.value,
    )

    assert rendered == (
        "2 case records are available "
        "in the deterministic case listing."
    )

    assert "CASE-1" not in rendered
    assert "CASE-2" not in rendered