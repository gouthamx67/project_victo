from __future__ import annotations

import pandas as pd

import src.analyses
from src.analyses.core import (
    calculate_case_count,
    calculate_reaction_count,
    calculate_serious_case_count,
    calculate_serious_case_percentage,
)
from src.analyses.reactions import (
    OUTCOME_CATEGORY_FATAL,
    OUTCOME_CATEGORY_MISSING,
    OUTCOME_CATEGORY_UNKNOWN,
    derive_case_outcomes,
)
from src.analyses.trends import (
    calculate_monthly_case_counts,
)
from src.normalize import NormalizedData


def make_invariant_data() -> NormalizedData:
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
            ],
            "receivedate": [
                "2025-01-01",
                "2025-01-15",
                "2025-02-01",
            ],
            "serious": [
                True,
                False,
                True,
            ],
            "expedited": [
                True,
                False,
                True,
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
            ],
            "case_id": [
                "CASE-001",
                "CASE-001",
                "CASE-002",
                "CASE-003",
            ],
            "reaction_pt": [
                "Headache",
                "Nausea",
                "Headache",
                "Fatal event",
            ],
            "outcome": [
                "unknown",
                "fatal",
                "unknown",
                pd.NA,
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_case_count_uses_unique_cases():
    data = make_invariant_data()

    result = calculate_case_count(data)

    assert result.value == 3
    assert result.population == "case"


def test_reaction_count_can_exceed_case_count():
    data = make_invariant_data()

    case_result = calculate_case_count(data)
    reaction_result = calculate_reaction_count(data)

    assert case_result.value == 3
    assert reaction_result.value == 4
    assert reaction_result.value > case_result.value


def test_monthly_case_counts_do_not_use_reaction_rows():
    data = make_invariant_data()

    result = calculate_monthly_case_counts(data)

    assert result.value == [
        {
            "reporting_month": "2025-01",
            "case_count": 2,
        },
        {
            "reporting_month": "2025-02",
            "case_count": 1,
        },
    ]


def test_serious_case_count_is_case_level():
    data = make_invariant_data()

    result = calculate_serious_case_count(data)

    assert result.value == 2
    assert result.population == "case"


def test_serious_case_percentage_uses_case_denominator():
    data = make_invariant_data()

    result = calculate_serious_case_percentage(data)

    assert result.value == 66.6667


def test_case_outcome_has_one_row_per_case():
    data = make_invariant_data()

    result = derive_case_outcomes(data)

    assert len(result) == 3
    assert result["case_id"].is_unique


def test_fatal_reaction_can_drive_case_outcome():
    data = make_invariant_data()

    result = derive_case_outcomes(data)

    case_001 = result.loc[
        result["case_id"] == "CASE-001"
    ].iloc[0]

    assert (
        case_001["case_outcome"]
        == OUTCOME_CATEGORY_FATAL
    )


def test_missing_outcome_does_not_become_unknown():
    data = make_invariant_data()

    result = derive_case_outcomes(data)

    case_003 = result.loc[
        result["case_id"] == "CASE-003"
    ].iloc[0]

    assert (
        case_003["case_outcome"]
        == OUTCOME_CATEGORY_MISSING
    )


def test_explicit_unknown_remains_unknown():
    data = make_invariant_data()

    result = derive_case_outcomes(data)

    case_002 = result.loc[
        result["case_id"] == "CASE-002"
    ].iloc[0]

    assert (
        case_002["case_outcome"]
        == OUTCOME_CATEGORY_UNKNOWN
    )


def test_all_case_ids_are_strings_in_analysis_supporting_ids():
    data = make_invariant_data()

    result = calculate_case_count(data)

    assert all(
        isinstance(case_id, str)
        for case_id in result.supporting_ids
    )


def test_reaction_ids_are_strings_in_analysis_supporting_ids():
    data = make_invariant_data()

    result = calculate_reaction_count(data)

    assert all(
        isinstance(reaction_id, str)
        for reaction_id in result.supporting_ids
    )