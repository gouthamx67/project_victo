from eval.grounding_benchmark import calculate_metrics, run_benchmark
from eval.grounding_evaluation import EvaluationResult


def make_result(
    case_id: str,
    expected_valid: bool,
    actual_valid: bool,
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        expected_valid=expected_valid,
        actual_valid=actual_valid,
        passed=expected_valid == actual_valid,
        description="Metric calculation test",
        issues=[],
    )


def test_benchmark_metrics_use_correct_denominators() -> None:
    results = [
        make_result("valid_accepted", True, True),
        make_result("valid_rejected", True, False),
        make_result("invalid_accepted", False, True),
        make_result("invalid_rejected", False, False),
    ]

    metrics = calculate_metrics(results)

    assert metrics.total == 4
    assert metrics.passed == 2
    assert metrics.true_positive == 1
    assert metrics.false_negative == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.valid_acceptance_rate == 0.5
    assert metrics.invalid_rejection_rate == 0.5
    assert metrics.invalid_acceptance_rate == 0.5
    assert metrics.accuracy == 0.5


def test_empty_benchmark_has_defined_empty_metrics() -> None:
    metrics = calculate_metrics([])

    assert metrics.total == 0
    assert metrics.passed == 0
    assert metrics.valid_acceptance_rate is None
    assert metrics.invalid_rejection_rate is None
    assert metrics.invalid_acceptance_rate is None
    assert metrics.accuracy == 0.0


def test_grounding_benchmark_cases_all_pass() -> None:
    metrics, results = run_benchmark()

    assert metrics.total == 7
    assert metrics.passed == metrics.total
    assert all(result.passed for result in results)
