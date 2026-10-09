from __future__ import annotations

from src.analysis_framework import AnalysisResult
from eval.grounding_evaluation import (
    EvaluationCase,
    assert_suite_passes,
    evaluate_case,
    evaluate_suite,
)


def make_analyses() -> list[AnalysisResult]:
    return [
        AnalysisResult(
            metric_id="case_count",
            value=1024,
            definition="Number of unique safety report cases.",
            population="case",
            supporting_ids=[
                "CASE001",
                "CASE002",
            ],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="serious_case_percentage",
            value=99.9,
            definition="Percentage of cases classified as serious.",
            population="case",
            supporting_ids=[
                "CASE001",
                "CASE002",
            ],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="top_reactions",
            value=[
                {
                    "reaction": "Acute kidney injury",
                    "count": 22,
                }
            ],
            definition="Most frequent reported reactions.",
            population="reaction",
            supporting_ids=[
                "CASE001",
                "CASE002",
            ],
            version="1.0.0",
        ),
    ]


def test_valid_grounded_narrative_is_accepted():
    case = EvaluationCase(
        case_id="valid",
        description="Valid grounded narrative.",
        narrative=(
            "The dataset contains {case_count} cases, "
            "with {serious_case_percentage}% "
            "classified as serious."
        ),
        expected_valid=True,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is True


def test_approved_nested_metric_number_is_accepted():
    case = EvaluationCase(
        case_id="nested-approved-number",
        description="Number contained in an approved metric value.",
        narrative=(
            "Acute kidney injury was reported "
            "22 times among the top reactions."
        ),
        expected_valid=True,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is True


def test_unsupported_number_is_rejected():
    case = EvaluationCase(
        case_id="unsupported-number",
        description="Unsupported numeric claim.",
        narrative=(
            "The dataset contains {case_count} cases "
            "and 17 cases were received from one region."
        ),
        expected_valid=False,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_unknown_metric_slot_is_rejected():
    case = EvaluationCase(
        case_id="unknown-slot",
        description="Unknown metric slot.",
        narrative=(
            "The dataset contains {case_count} cases "
            "and {unknown_metric} additional cases."
        ),
        expected_valid=False,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_missing_required_metric_is_rejected():
    case = EvaluationCase(
        case_id="missing-required-metric",
        description="Required metric is not referenced.",
        narrative=(
            "The dataset contains {case_count} cases."
        ),
        expected_valid=False,
        required_metric_ids=(
            "case_count",
            "serious_case_percentage",
        ),
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_causal_claim_is_rejected():
    case = EvaluationCase(
        case_id="causal",
        description="Unsupported causal claim.",
        narrative=(
            "The reaction was caused by bisoprolol."
        ),
        expected_valid=False,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_signal_claim_is_rejected():
    case = EvaluationCase(
        case_id="signal",
        description="Unsupported safety signal claim.",
        narrative=(
            "The findings indicate a safety signal."
        ),
        expected_valid=False,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_expectedness_claim_is_rejected():
    case = EvaluationCase(
        case_id="expectedness",
        description="Unsupported expectedness claim.",
        narrative=(
            "The event was an expected adverse reaction."
        ),
        expected_valid=False,
    )

    result = evaluate_case(
        case,
        make_analyses(),
    )

    assert result.passed
    assert result.actual_valid is False


def test_suite_reports_all_cases():
    cases = [
        EvaluationCase(
            case_id="valid",
            description="Valid grounded narrative.",
            narrative=(
                "{case_count} cases were reported."
            ),
            expected_valid=True,
        ),
        EvaluationCase(
            case_id="approved-percentage",
            description="Approved percentage value.",
            narrative=(
                "{serious_case_percentage}% "
                "of cases were classified as serious."
            ),
            expected_valid=True,
        ),
        EvaluationCase(
            case_id="unsupported-number",
            description="Unsupported numeric claim.",
            narrative=(
                "{case_count} cases were reported, "
                "including 17 new cases."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="causal",
            description="Unsupported causal claim.",
            narrative=(
                "The reaction was caused by bisoprolol."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="expectedness",
            description="Unsupported expectedness claim.",
            narrative=(
                "The event was an expected adverse reaction."
            ),
            expected_valid=False,
        ),
    ]

    results = evaluate_suite(
        cases,
        make_analyses(),
    )

    assert len(results) == 5

    assert all(
        result.passed
        for result in results
    )

    assert_suite_passes(results)