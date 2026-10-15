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


def test_approved_number_outside_slot_is_invalid():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The approved count is 1024.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )


def test_approved_percentage_outside_slot_is_invalid():
    results = [
        make_result("serious_case_percentage", 99.9),
    ]

    result = validate_grounding(
        "Serious cases represented 99.9% of cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )


def test_approved_number_misattributed_to_wrong_metric_is_invalid():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_count", 12),
    ]

    # Both numbers occur in approved evidence, but their positions in this
    # sentence reverse the metric associations. The validator must require
    # slots instead of accepting globally approved numbers.
    result = validate_grounding(
        "There were 12 cases and 1024 serious cases.",
        results,
    )

    assert result.valid is False
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )


def test_metric_slot_allows_deterministic_value_insertion():
    results = [
        make_result("case_count", 1024),
    ]

    result = validate_grounding(
        "The dataset contained {case_count} cases.",
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
        "The outcome summary is {outcomes}.",
        results,
    ).valid

    assert not validate_grounding(
        "The outcome summary is {outcomes}, including 3 fatal "
        "and 999 recovered outcomes.",
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


def test_neutral_causal_uncertainty_is_accepted():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report did not establish that the drug caused the observed cases.",
        results,
    )

    assert result.valid is True


def test_unsupported_negated_causal_assertion_is_rejected():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The drug did not cause the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_unsupported_statistic_is_rejected():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report recorded 9999 cases.",
        results,
    )

    assert result.valid is False


def test_missing_required_metric_evidence_is_rejected():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report describes case volume.",
        results,
        required_metric_ids=["case_count"],
    )

    assert result.valid is False


def test_neutral_discussion_does_not_exempt_later_causal_assertion():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report discusses whether the treatment was appropriate, "
        "but the drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_neutral_discussion_does_not_exempt_assertion_after_but():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report discusses whether the treatment was appropriate "
        "but the drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_neutral_discussion_does_not_exempt_causal_claim_after_because():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report discusses whether the treatment was appropriate "
        "because the drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_unsupported_causal_claim_using_causes_is_rejected():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The drug causes the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_unsupported_causal_claim_using_causing_is_rejected():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The drug is causing the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_neutral_discussion_does_not_exempt_causal_claim_after_colon():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report discusses whether the treatment was appropriate: "
        "the drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_neutral_discussion_does_not_exempt_causal_claim_after_when():
    results = [make_result("case_count", 1024)]

    result = validate_grounding(
        "The report discusses whether the treatment was appropriate "
        "when the drug caused the observed cases.",
        results,
    )

    assert result.valid is False
    assert any(issue.code == "banned_claim" for issue in result.issues)


def test_digits_inside_metric_slot_name_are_not_narrative_numbers():
    from src.grounding_validator import _find_unsupported_numbers

    assert _find_unsupported_numbers("Use {metric_2}.") == []


def test_nested_and_unmatched_metric_braces_are_rejected():
    results = [make_result("case_count", 1024)]
    narratives = [
        "The dataset contained {{case_count}} cases.",
        "The dataset contained {case_count cases.",
        "The dataset contained case_count} cases.",
    ]

    for narrative in narratives:
        result = validate_grounding(narrative, results)
        assert not result.valid, narrative
        assert any(
            issue.code == "invalid_metric_slot"
            for issue in result.issues
        ), narrative


@pytest.mark.parametrize(
    "narrative",
    [
        "The dataset contained one thousand cases.",
        "Fifteen reactions were reported.",
        "Twenty-five serious reports were logged.",
    ],
)
def test_spelled_out_quantities_outside_slots_are_invalid(narrative):
    results = [make_result("case_count", 1024)]

    result = validate_grounding(narrative, results)

    assert not result.valid
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "narrative",
    [
        "The dataset contained １２３ cases.",
        "The dataset contained ١٢٣ cases.",
    ],
)
def test_unicode_decimal_digits_outside_slots_are_invalid(narrative):
    results = [make_result("case_count", 1024)]

    result = validate_grounding(narrative, results)

    assert not result.valid
    assert any(
        issue.code == "unsupported_number"
        for issue in result.issues
    )
