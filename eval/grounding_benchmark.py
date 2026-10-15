from __future__ import annotations

from dataclasses import dataclass

from src.analysis_framework import AnalysisResult
from eval.grounding_evaluation import (
    EvaluationCase,
    EvaluationResult,
    evaluate_suite,
)


@dataclass(frozen=True)
class BenchmarkMetrics:
    total: int
    passed: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    valid_acceptance_rate: float | None
    invalid_rejection_rate: float | None
    invalid_acceptance_rate: float | None
    accuracy: float


def build_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="valid_case_count",
            description="Approved metric slot and value.",
            narrative="The dataset contains {case_count} cases.",
            expected_valid=True,
            required_metric_ids=("case_count",),
        ),
        EvaluationCase(
            case_id="valid_neutral_review",
            description="Neutral monthly-review wording.",
            narrative=(
                "The review found no monthly deviations "
                "above the configured threshold."
            ),
            expected_valid=True,
        ),
        EvaluationCase(
            case_id="invalid_unsupported_number",
            description="Number not present in approved evidence.",
            narrative="The dataset contains 999 cases.",
            expected_valid=False,
            required_metric_ids=("case_count",),
        ),
        EvaluationCase(
            case_id="invalid_approved_number_outside_slot",
            description=(
                "An approved number must still be referenced through its slot."
            ),
            narrative="The report recorded 1024 cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_spelled_out_number",
            description="A spelled-out quantity must use a metric slot.",
            narrative="The dataset contained one thousand cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_unicode_digits",
            description="Unicode decimal digits outside slots are rejected.",
            narrative="The dataset contained １２３ cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_unknown_slot",
            description="Unknown metric slot.",
            narrative="The dataset contains {unknown_metric} cases.",
            expected_valid=False,
            required_metric_ids=("case_count",),
        ),
        EvaluationCase(
            case_id="invalid_causal_claim",
            description="Unsupported causal claim.",
            narrative="The drug caused the observed cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_safety_signal",
            description="Unsupported safety-signal claim.",
            narrative="A safety signal was detected.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_monthly_interpretation",
            description="Unsupported monthly variability interpretation.",
            narrative=(
                "The pattern reflects typical month-to-month variability."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="valid_neutral_causal_discussion",
            description="Neutral question about causality.",
            narrative=(
                "The report discusses whether the drug caused "
                "the observed cases."
            ),
            expected_valid=True,
        ),
        EvaluationCase(
            case_id="valid_causality_not_established",
            description="States that causality was not established.",
            narrative=(
                "The report did not establish that the drug caused "
                "the observed cases."
            ),
            expected_valid=True,
        ),
        EvaluationCase(
            case_id="invalid_led_to_claim",
            description="Unsupported causal claim using led to.",
            narrative="The drug led to the observed cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_negated_causal_claim",
            description="Unsupported causal assertion phrased as a denial.",
            narrative="The drug did not cause the observed cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_unsupported_statistic",
            description="Unsupported number in a narrative.",
            narrative="The report recorded 9999 cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_missing_required_metric",
            description="Required evidence slot is absent.",
            narrative="The report describes case volume.",
            expected_valid=False,
            required_metric_ids=("case_count",),
        ),
        EvaluationCase(
            case_id="invalid_causal_claim_after_comma",
            description="Neutral discussion must not shield a later causal claim.",
            narrative=(
                "The report discusses whether the treatment was appropriate, "
                "but the drug caused the observed cases."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causal_claim_after_but",
            description="A conjunction must end the neutral exception.",
            narrative=(
                "The report discusses whether the treatment was appropriate "
                "but the drug caused the observed cases."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causal_claim_after_because",
            description="A causal assertion after because is not exempt.",
            narrative=(
                "The report discusses whether the treatment was appropriate "
                "because the drug caused the observed cases."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causes_claim",
            description="Unsupported causal claim using causes.",
            narrative="The drug causes the observed cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causing_claim",
            description="Unsupported causal claim using causing.",
            narrative="The drug is causing the observed cases.",
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causal_claim_after_colon",
            description="A causal assertion after a colon is not exempt.",
            narrative=(
                "The report discusses whether the treatment was appropriate: "
                "the drug caused the observed cases."
            ),
            expected_valid=False,
        ),
        EvaluationCase(
            case_id="invalid_causal_claim_after_when",
            description="A causal assertion after when is not exempt.",
            narrative=(
                "The report discusses whether the treatment was appropriate "
                "when the drug caused the observed cases."
            ),
            expected_valid=False,
        ),
    ]


def calculate_metrics(
    results: list[EvaluationResult],
) -> BenchmarkMetrics:
    # "Valid" is the positive class.
    true_positive = sum(
        result.expected_valid and result.actual_valid
        for result in results
    )
    false_positive = sum(
        not result.expected_valid and result.actual_valid
        for result in results
    )
    true_negative = sum(
        not result.expected_valid and not result.actual_valid
        for result in results
    )
    false_negative = sum(
        result.expected_valid and not result.actual_valid
        for result in results
    )

    valid_count = true_positive + false_negative
    invalid_count = true_negative + false_positive
    total = len(results)

    return BenchmarkMetrics(
        total=total,
        passed=sum(result.passed for result in results),
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
        valid_acceptance_rate=(
            true_positive / valid_count if valid_count else None
        ),
        invalid_rejection_rate=(
            true_negative / invalid_count if invalid_count else None
        ),
        invalid_acceptance_rate=(
            false_positive / invalid_count if invalid_count else None
        ),
        accuracy=(
            (true_positive + true_negative) / total if total else 0.0
        ),
    )


def run_benchmark() -> tuple[BenchmarkMetrics, list[EvaluationResult]]:
    analyses = [
        AnalysisResult(
            metric_id="case_count",
            value=1024,
            definition="Number of unique safety report cases.",
            population="case",
            supporting_ids=[],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="monthly_review",
            value="No monthly deviations above the configured threshold.",
            definition="Configured monthly deviation review result.",
            population="case",
            supporting_ids=[],
            version="1.0.0",
        ),
    ]

    results = evaluate_suite(build_cases(), analyses)
    return calculate_metrics(results), results


def _format_rate(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "N/A"


def main() -> int:
    metrics, results = run_benchmark()

    print("Grounding benchmark")
    print("=" * 40)
    print(f"Cases:                  {metrics.total}")
    print(f"Correct classifications: {metrics.passed}/{metrics.total}")
    print(f"True positives:         {metrics.true_positive}")
    print(f"False positives:        {metrics.false_positive}")
    print(f"True negatives:         {metrics.true_negative}")
    print(f"False negatives:        {metrics.false_negative}")
    print(f"Valid acceptance rate:  {_format_rate(metrics.valid_acceptance_rate)}")
    print(f"Invalid rejection rate: {_format_rate(metrics.invalid_rejection_rate)}")
    print(f"Invalid acceptance rate: {_format_rate(metrics.invalid_acceptance_rate)}")
    print(f"Accuracy:               {_format_rate(metrics.accuracy)}")
    print()
    print("Case results:")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.case_id}: "
            f"expected_valid={result.expected_valid}, "
            f"actual_valid={result.actual_valid}"
        )
        for issue in result.issues:
            print(f"       - {issue}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
