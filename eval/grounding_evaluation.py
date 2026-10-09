from __future__ import annotations

from dataclasses import dataclass

from src.analysis_framework import AnalysisResult
from src.grounding_validator import validate_grounding


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    description: str
    narrative: str
    expected_valid: bool
    required_metric_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    expected_valid: bool
    actual_valid: bool
    passed: bool
    description: str
    issues: tuple[str, ...]


def evaluate_case(
    case: EvaluationCase,
    analyses: list[AnalysisResult],
) -> EvaluationResult:
    validation = validate_grounding(
        case.narrative,
        analyses,
        required_metric_ids=case.required_metric_ids or None,
    )

    actual_valid = validation.valid

    return EvaluationResult(
        case_id=case.case_id,
        expected_valid=case.expected_valid,
        actual_valid=actual_valid,
        passed=(actual_valid == case.expected_valid),
        description=case.description,
        issues=tuple(validation.messages),
    )


def evaluate_suite(
    cases: list[EvaluationCase],
    analyses: list[AnalysisResult],
) -> list[EvaluationResult]:
    return [
        evaluate_case(case, analyses)
        for case in cases
    ]


def assert_suite_passes(
    results: list[EvaluationResult],
) -> None:
    failures = [
        result
        for result in results
        if not result.passed
    ]

    if not failures:
        return

    details = "\n".join(
        (
            f"{result.case_id}: "
            f"expected={result.expected_valid}, "
            f"actual={result.actual_valid}, "
            f"issues={result.issues}"
        )
        for result in failures
    )

    raise AssertionError(
        "Grounding evaluation failed:\n"
        + details
    )