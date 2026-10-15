from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.analysis_framework import AnalysisResult
from src.grounding_validator import validate_grounding


@dataclass(frozen=True)
class MutationCase:
    mutation_id: str
    description: str
    check: Callable[[], bool]


def wrong_serious_denominator_detected() -> bool:
    """
    Mutation:
    Pretend row count is the denominator instead of unique cases.

    The real dataset has 1024 cases and 1068 rows, so a row-based
    denominator would produce a different percentage.
    """
    correct_case_count = 1024
    serious_case_count = 1023

    wrong_percentage = (
        serious_case_count / 1068
    ) * 100

    correct_percentage = (
        serious_case_count / correct_case_count
    ) * 100

    return round(wrong_percentage, 4) != round(
        correct_percentage,
        4,
    )


def unsupported_number_detected() -> bool:
    result = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of cases.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    validation = validate_grounding(
        "There were 999 cases.",
        [result],
        required_metric_ids=["case_count"],
    )

    return not validation.valid


def unknown_slot_detected() -> bool:
    result = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of cases.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    validation = validate_grounding(
        "There were {unknown_metric} cases.",
        [result],
        required_metric_ids=["case_count"],
    )

    return not validation.valid


def causal_claim_detected() -> bool:
    result = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of cases.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    validation = validate_grounding(
        "The drug caused the observed cases.",
        [result],
        required_metric_ids=["case_count"],
    )

    return not validation.valid



def unsupported_interpretation_detected() -> bool:
    result = AnalysisResult(
        metric_id="monthly_review",
        value="No monthly deviations above the configured threshold.",
        definition="Configured monthly deviation review result.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    validation = validate_grounding(
        "The pattern reflects typical month-to-month variability.",
        [result],
    )

    return any(
        issue.code == "unsupported_interpretation"
        for issue in validation.issues
    )


def safety_signal_claim_detected() -> bool:
    result = AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of cases.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )

    validation = validate_grounding(
        "A safety signal was detected.",
        [result],
        required_metric_ids=["case_count"],
    )

    return not validation.valid


MUTATION_CASES = [
    MutationCase(
        mutation_id="wrong_serious_denominator",
        description=(
            "Use row count rather than unique case count "
            "as the seriousness denominator."
        ),
        check=wrong_serious_denominator_detected,
    ),
    MutationCase(
        mutation_id="unsupported_number",
        description=(
            "Introduce a number that is not present in "
            "approved deterministic metrics."
        ),
        check=unsupported_number_detected,
    ),
    MutationCase(
        mutation_id="unknown_metric_slot",
        description=(
            "Introduce a metric slot that does not exist."
        ),
        check=unknown_slot_detected,
    ),
    MutationCase(
        mutation_id="causal_claim",
        description=(
            "Introduce unsupported causal language."
        ),
        check=causal_claim_detected,
    ),
    MutationCase(
        mutation_id="unsupported_interpretation",
        description=(
            "Introduce an interpretation of monthly variability "
            "that is not supported by the descriptive review result."
        ),
        check=unsupported_interpretation_detected,
    ),
    MutationCase(
        mutation_id="safety_signal_claim",
        description=(
            "Introduce unsupported safety-signal language."
        ),
        check=safety_signal_claim_detected,
    ),
]