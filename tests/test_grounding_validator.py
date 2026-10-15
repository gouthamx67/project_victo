from __future__ import annotations

import pytest

from src.analysis_framework import AnalysisResult
from src.grounding_validator import (
    GroundingValidationError,
    assert_grounded,
    validate_grounding,
)


def make_result(
    metric_id: str,
    value,
    population: str = "case",
) -> AnalysisResult:
    return AnalysisResult(
        metric_id=metric_id,
        value=value,
        definition=f"Definition of {metric_id}.",
        population=population,
        supporting_ids=[],
        version="1.0.0",
    )


def test_valid_grounded_narrative():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The dataset contained {case_count} cases.",
        results,
    )

    assert result.valid is True
    assert result.issues == ()


def test_unknown_slot_is_invalid():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The dataset contained {invented_count} cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "invalid_metric_slot"
        for issue in result.issues
    )


def test_unsupported_number_is_invalid():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The dataset contained {case_count} cases and 999 additional cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )


def test_approved_number_is_allowed():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The approved count is 1024.",
        results,
    )

    assert result.valid is True


def test_approved_percentage_is_allowed():
    results = [
        make_result("serious_case_percentage", 99.9),
    ]

    result = validate_grounding(
        "Serious cases represented 99.9% of cases.",
        results,
    )

    assert result.valid is True


def test_banned_safety_signal_claim_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The data indicate a safety signal.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "banned_claim"
        for issue in result.issues
    )


def test_causal_claim_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The adverse event was caused by bisoprolol.",
        results,
    )

    assert result.valid is False


def test_expectedness_claim_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "This was an expected adverse reaction.",
        results,
    )

    assert result.valid is False


def test_required_metric_coverage_is_enforced():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_count", 1023),
    ]

    result = validate_grounding(
        "There were {case_count} cases.",
        results,
        required_metric_ids=[
            "case_count",
            "serious_case_count",
        ],
    )

    assert result.valid is False
    assert any(
        issue.code == "missing_required_metric"
        for issue in result.issues
    )


def test_required_metric_coverage_passes():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_count", 1023),
    ]

    result = validate_grounding(
        "There were {case_count} cases and "
        "{serious_case_count} serious cases.",
        results,
        required_metric_ids=[
            "case_count",
            "serious_case_count",
        ],
    )

    assert result.valid is True


def test_assert_grounded_raises():
    results = [
        make_result("case_count", 1024),
    ]

    with pytest.raises(
        GroundingValidationError,
        match="Grounding validation failed",
    ):
        assert_grounded(
            "There were 999 cases.",
            results,
        )


def test_assert_grounded_accepts_valid_text():
    results = [
        make_result("case_count", 1024),
    ]

    assert_grounded(
        "There were {case_count} cases.",
        results,
    )


def test_nested_metric_values_are_checked():
    results = [
        make_result(
            "outcomes",
            {
                "fatal": 3,
                "recovered": 20,
            },
        )
    ]

    assert validate_grounding(
        "There were 3 fatal and 20 recovered outcomes.",
        results,
    ).valid

    assert not validate_grounding(
        "There were 3 fatal and 999 recovered outcomes.",
        results,
    ).valid

def test_unsupported_typical_variability_claim_is_rejected():
    results = [
        make_result(
            "monthly_review",
            "No monthly deviations above the configured threshold.",
        ),
    ]

    result = validate_grounding(
        "The pattern reflects typical month-to-month variability "
        "without any deterministic review flags.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_interpretation"
        for issue in result.issues
    )


def test_neutral_monthly_description_is_not_flagged_as_interpretation():
    results = [
        make_result(
            "monthly_review",
            "No monthly deviations above the configured threshold.",
        ),
    ]

    result = validate_grounding(
        "The review found no monthly deviations above the configured threshold.",
        results,
    )

    assert result.valid is True
    assert not any(
        issue.code == "unsupported_interpretation"
        for issue in result.issues
    )


def test_unsupported_causal_claim_using_caused_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "banned_claim"
        for issue in result.issues
    )


def test_unsupported_causal_claim_using_led_to_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The drug led to the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "banned_claim"
        for issue in result.issues
    )


def test_neutral_causal_topic_without_causal_assertion_is_accepted():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The report discusses whether the drug caused the observed cases.",
        results,
    )

    assert result.valid is True
