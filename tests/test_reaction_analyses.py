from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analyses.reactions import (
    ANALYSIS_VERSION,
    OUTCOME_CATEGORY_FATAL,
    OUTCOME_CATEGORY_MISSING,
    OUTCOME_CATEGORY_NOT_RECOVERED,
    OUTCOME_CATEGORY_RECOVERED,
    OUTCOME_CATEGORY_RECOVERING,
    OUTCOME_CATEGORY_SEQUELAE,
    OUTCOME_CATEGORY_UNKNOWN,
    OUTCOME_CATEGORY_UNMAPPED,
    calculate_case_outcome_distribution,
    calculate_fatal_case_count,
    calculate_not_recovered_case_count,
    calculate_reaction_outcome_distribution,
    calculate_recovered_case_count,
    calculate_top_serious_reactions,
    classify_outcome,
    derive_case_outcome,
    derive_case_outcomes,
)
from src.ingest import load_raw_csv
from src.normalize import NormalizedData, normalize_dataset


CSV_PATH = Path(
    "data/Bisoprolol_icsr_sample_1068rows (1).csv"
)


def make_outcome_data() -> NormalizedData:
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
                "CASE-005",
                "CASE-006",
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
                "CASE-005:0:0",
                "CASE-006:0:0",
            ],
            "case_id": [
                "CASE-001",
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
                "CASE-005",
                "CASE-006",
            ],
            "reaction_pt": [
                "Reaction A",
                "Reaction B",
                "Reaction C",
                "Reaction D",
                "Reaction E",
                "Reaction F",
                "Reaction G",
            ],
            "outcome": [
                "recovered/resolved",
                "fatal",
                "not recovered/not resolved/ongoing",
                "recovered/resolved with sequelae",
                "recovering/resolving",
                "unknown",
                pd.NA,
            ],
            "outcome_present": [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_classify_known_outcomes():
    assert (
        classify_outcome("fatal")
        == OUTCOME_CATEGORY_FATAL
    )

    assert (
        classify_outcome(
            "not recovered/not resolved/ongoing"
        )
        == OUTCOME_CATEGORY_NOT_RECOVERED
    )

    assert (
        classify_outcome(
            "recovered/resolved with sequelae"
        )
        == OUTCOME_CATEGORY_SEQUELAE
    )

    assert (
        classify_outcome("recovering/resolving")
        == OUTCOME_CATEGORY_RECOVERING
    )

    assert (
        classify_outcome("recovered/resolved")
        == OUTCOME_CATEGORY_RECOVERED
    )

    assert (
        classify_outcome("unknown")
        == OUTCOME_CATEGORY_UNKNOWN
    )


def test_missing_outcome_is_classified_as_missing():
    assert classify_outcome(pd.NA) == OUTCOME_CATEGORY_MISSING
    assert classify_outcome("") == OUTCOME_CATEGORY_MISSING
    assert classify_outcome("   ") == OUTCOME_CATEGORY_MISSING


def test_unmapped_outcome_is_not_guessed():
    assert (
        classify_outcome("some-new-source-value")
        == OUTCOME_CATEGORY_UNMAPPED
    )


def test_case_outcome_fatal_has_highest_precedence():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "recovering/resolving",
            "fatal",
        ]
    )

    assert result == OUTCOME_CATEGORY_FATAL


def test_case_outcome_not_recovered_precedes_recovered():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "not recovered/not resolved/ongoing",
        ]
    )

    assert result == OUTCOME_CATEGORY_NOT_RECOVERED


def test_case_outcome_sequelae_precedes_recovering():
    result = derive_case_outcome(
        [
            "recovering/resolving",
            "recovered/resolved with sequelae",
        ]
    )

    assert result == OUTCOME_CATEGORY_SEQUELAE


def test_case_outcome_recovering_precedes_recovered():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "recovering/resolving",
        ]
    )

    assert result == OUTCOME_CATEGORY_RECOVERING


def test_case_outcome_recovered_is_used_when_only_recovered_exists():
    result = derive_case_outcome(
        [
            "recovered/resolved",
            "recovered/resolved",
        ]
    )

    assert result == OUTCOME_CATEGORY_RECOVERED


def test_case_outcome_unknown_is_used_when_only_unknown_exists():
    result = derive_case_outcome(
        [
            "unknown",
            "unknown",
        ]
    )

    assert result == OUTCOME_CATEGORY_UNKNOWN


def test_case_outcome_missing_when_no_outcome_exists():
    result = derive_case_outcome(
        [
            pd.NA,
            "",
        ]
    )

    assert result == OUTCOME_CATEGORY_MISSING


def test_case_outcome_unmapped_is_not_invented():
    result = derive_case_outcome(
        [
            "some-new-source-value",
        ]
    )

    assert result == OUTCOME_CATEGORY_UNMAPPED


def test_derive_case_outcomes_returns_one_row_per_case():
    data = make_outcome_data()

    result = derive_case_outcomes(data)

    assert len(result) == 6
    assert result["case_id"].is_unique


def test_derive_case_outcomes_uses_precedence():
    data = make_outcome_data()

    result = derive_case_outcomes(data)

    outcomes = dict(
        zip(
            result["case_id"],
            result["case_outcome"],
        )
    )

    assert outcomes["CASE-001"] == OUTCOME_CATEGORY_FATAL
    assert (
        outcomes["CASE-002"]
        == OUTCOME_CATEGORY_NOT_RECOVERED
    )
    assert outcomes["CASE-003"] == OUTCOME_CATEGORY_SEQUELAE
    assert outcomes["CASE-004"] == OUTCOME_CATEGORY_RECOVERING
    assert outcomes["CASE-005"] == OUTCOME_CATEGORY_UNKNOWN
    assert outcomes["CASE-006"] == OUTCOME_CATEGORY_MISSING


def test_case_outcomes_retain_supporting_reaction_ids():
    data = make_outcome_data()

    result = derive_case_outcomes(data)

    case_001 = result.loc[
        result["case_id"] == "CASE-001"
    ].iloc[0]

    assert set(
        case_001["outcome_supporting_reaction_ids"]
    ) == {
        "CASE-001:0:0",
        "CASE-001:0:1",
    }


def test_reaction_outcome_distribution():
    data = make_outcome_data()

    result = calculate_reaction_outcome_distribution(data)

    assert result.metric_id == "reaction_outcome_distribution"
    assert result.population == "reaction"
    assert result.version == ANALYSIS_VERSION

    assert result.value == [
        {
            "outcome_category": OUTCOME_CATEGORY_FATAL,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_NOT_RECOVERED,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_SEQUELAE,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_RECOVERING,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_RECOVERED,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_UNKNOWN,
            "count": 1,
        },
        {
            "outcome_category": OUTCOME_CATEGORY_MISSING,
            "count": 1,
        },
    ]


def test_case_outcome_distribution():
    data = make_outcome_data()

    result = calculate_case_outcome_distribution(data)

    assert result.metric_id == "case_outcome_distribution"
    assert result.population == "case"

    assert result.value == [
        {
            "case_outcome": OUTCOME_CATEGORY_FATAL,
            "count": 1,
        },
        {
            "case_outcome": OUTCOME_CATEGORY_NOT_RECOVERED,
            "count": 1,
        },
        {
            "case_outcome": OUTCOME_CATEGORY_SEQUELAE,
            "count": 1,
        },
        {
            "case_outcome": OUTCOME_CATEGORY_RECOVERING,
            "count": 1,
        },
        {
            "case_outcome": OUTCOME_CATEGORY_UNKNOWN,
            "count": 1,
        },
        {
            "case_outcome": OUTCOME_CATEGORY_MISSING,
            "count": 1,
        },
    ]


def test_fatal_case_count():
    data = make_outcome_data()

    result = calculate_fatal_case_count(data)

    assert result.metric_id == "fatal_case_count"
    assert result.value == 1
    assert result.supporting_ids == ["CASE-001"]


def test_not_recovered_case_count():
    data = make_outcome_data()

    result = calculate_not_recovered_case_count(data)

    assert result.metric_id == "not_recovered_case_count"
    assert result.value == 1
    assert result.supporting_ids == ["CASE-002"]


def test_recovered_case_count():
    data = make_outcome_data()

    result = calculate_recovered_case_count(data)

    assert result.metric_id == "recovered_case_count"
    assert result.value == 0
    assert result.supporting_ids == []


def test_all_case_outcome_supporting_ids_are_case_ids():
    data = make_outcome_data()

    result = calculate_case_outcome_distribution(data)

    assert set(result.supporting_ids) == {
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
        "CASE-005",
        "CASE-006",
    }


def test_top_serious_reactions_is_restricted_to_serious_cases():
    data = normalize_dataset(
        load_raw_csv(CSV_PATH)
    )

    result = calculate_top_serious_reactions(data)

    assert result.metric_id == "top_serious_reactions"
    assert result.population == "reaction"
    assert result.unit == "reaction_occurrences"
    assert len(result.value) <= 10

    for row in result.value:
        assert set(row) == {
            "reaction_pt",
            "occurrences",
            "distinct_cases",
        }
        assert row["occurrences"] >= row["distinct_cases"]

    serious_case_ids = set(
        data.cases.loc[
            data.cases["serious"].eq(True),
            "case_id",
        ]
        .astype(str)
    )

    assert set(result.supporting_ids).issubset(
        serious_case_ids
    )