from __future__ import annotations

import pandas as pd
import pytest

from src.analyses.core import (
    calculate_case_count,
    calculate_serious_case_percentage,
    calculate_top_reactions,
)
from src.analyses.reactions import (
    OUTCOME_CATEGORY_FATAL,
    OUTCOME_CATEGORY_NOT_RECOVERED,
    derive_case_outcome,
)
from src.analyses.trends import (
    calculate_monthly_case_counts,
    calculate_monthly_case_deviations,
)
from src.normalize import NormalizedData


def make_mutation_data() -> NormalizedData:
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
            ],
            "receivedate": [
                "2025-01-01",
                "2025-01-02",
                "2025-02-01",
                "2025-02-02",
            ],
            "serious": [
                True,
                False,
                True,
                False,
            ],
            "expedited": [
                True,
                False,
                True,
                False,
            ],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [
                "CASE-001:0:0",
                "CASE-001:0:1",
                "CASE-002:0:0",
                "CASE-003:0:0",
                "CASE-004:0:0",
            ],
            "case_id": [
                "CASE-001",
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
            ],
            "reaction_pt": [
                "Headache",
                "Nausea",
                "Headache",
                "Headache",
                "Headache",
            ],
            "outcome": [
                "fatal",
                "recovered/resolved",
                "unknown",
                "unknown",
                "unknown",
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_case_count_catches_raw_row_count_mutation():
    data = make_mutation_data()

    correct = calculate_case_count(data)

    assert correct.value == 4
    assert len(data.reactions) == 5

    # If implementation were accidentally changed to count reaction
    # rows, this invariant would fail.
    assert correct.value != len(data.reactions)


def test_monthly_case_counts_catches_reaction_based_counting():
    data = make_mutation_data()

    result = calculate_monthly_case_counts(data)

    assert result.value == [
        {
            "reporting_month": "2025-01",
            "case_count": 2,
        },
        {
            "reporting_month": "2025-02",
            "case_count": 2,
        },
    ]

    # January contains three reaction rows but only two cases.
    assert result.value[0]["case_count"] != 3


def test_serious_percentage_catches_reaction_denominator():
    data = make_mutation_data()

    result = calculate_serious_case_percentage(data)

    # 2 serious cases / 4 total cases = 50%.
    assert result.value == 50.0

    # A reaction-level denominator would be 40%, which is wrong.
    assert result.value != 40.0


def test_fatal_has_priority_over_recovered():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "fatal",
        ]
    )

    assert result == OUTCOME_CATEGORY_FATAL


def test_not_recovered_has_priority_over_recovered():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "not recovered/not resolved/ongoing",
        ]
    )

    assert result == OUTCOME_CATEGORY_NOT_RECOVERED


def test_top_reactions_uses_deterministic_tie_breaking():
    data = make_mutation_data()

    result = calculate_top_reactions(data)

    assert result.value[0]["reaction_pt"] == "Headache"
    assert result.value[0]["occurrences"] == 4

    # If two terms have equal counts, alphabetical ordering must
    # make the output deterministic.
    assert result.value[1]["reaction_pt"] == "Nausea"


def test_deviation_rule_is_strictly_greater_than_threshold():
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
            ],
            "receivedate": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
            ],
            "serious": [
                False,
                False,
                False,
            ],
            "expedited": [
                False,
                False,
                False,
            ],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [
                "CASE-001:0:0",
                "CASE-002:0:0",
                "CASE-003:0:0",
            ],
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
            ],
            "reaction_pt": [
                "A",
                "B",
                "C",
            ],
            "outcome": [
                "unknown",
                "unknown",
                "unknown",
            ],
        }
    )

    data = NormalizedData(
        cases=cases,
        reactions=reactions,
    )

    result = calculate_monthly_case_deviations(data)

    # Mean = 1, threshold = 1.5.
    assert all(
        row["deviation_flag"] is False
        for row in result.value
    )


def test_missing_receivedate_is_not_silently_replaced():
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
            ],
            "receivedate": [
                pd.NA,
            ],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [
                "CASE-001:0:0",
            ],
            "case_id": [
                "CASE-001",
            ],
            "reaction_pt": [
                "Headache",
            ],
            "outcome": [
                "unknown",
            ],
        }
    )

    data = NormalizedData(
        cases=cases,
        reactions=reactions,
    )

    result = calculate_monthly_case_counts(data)

    assert result.value == []