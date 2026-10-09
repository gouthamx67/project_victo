import pytest

from src.analysis_framework import AnalysisResult
from src.evidence_trace import build_evidence_trace


def make_results() -> list[AnalysisResult]:
    return [
        AnalysisResult(
            metric_id="case_count",
            value=1024,
            definition="Number of cases.",
            population="case",
            supporting_ids=["CASE-1", "CASE-2"],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="fatal_case_count",
            value=3,
            definition="Number of fatal cases.",
            population="case",
            supporting_ids=["CASE-9", "CASE-10", "CASE-11"],
            version="1.0.0",
        ),
    ]


def test_build_evidence_trace() -> None:
    trace = build_evidence_trace(
        section_id="executive_summary",
        metric_ids=[
            "case_count",
            "fatal_case_count",
        ],
        analyses=make_results(),
    )

    assert trace.section_id == "executive_summary"
    assert trace.metric_ids == [
        "case_count",
        "fatal_case_count",
    ]

    assert trace.supporting_ids["case_count"] == [
        "CASE-1",
        "CASE-2",
    ]

    assert trace.supporting_ids["fatal_case_count"] == [
        "CASE-9",
        "CASE-10",
        "CASE-11",
    ]


def test_missing_metric_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Missing analyses",
    ):
        build_evidence_trace(
            section_id="executive_summary",
            metric_ids=["unknown_metric"],
            analyses=make_results(),
        )


def test_empty_supporting_ids_are_preserved() -> None:
    result = AnalysisResult(
        metric_id="example_metric",
        value=1,
        definition="Example.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    trace = build_evidence_trace(
        section_id="example",
        metric_ids=["example_metric"],
        analyses=[result],
    )

    assert trace.supporting_ids["example_metric"] == []