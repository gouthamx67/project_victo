from __future__ import annotations

import pandas as pd

from src.analyses.core import (
    ANALYSIS_VERSION,
    TOP_REACTION_LIMIT,
    calculate_case_count,
    calculate_distinct_case_reaction_pairs,
    calculate_expedited_case_count,
    calculate_expedited_case_percentage,
    calculate_non_serious_case_count,
    calculate_reaction_count,
    calculate_serious_case_count,
    calculate_serious_case_percentage,
    calculate_top_reactions,
    calculate_unique_reaction_count,
)
from src.normalize import NormalizedData


def make_test_data() -> NormalizedData:
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-004",
            ],
            "serious": [
                True,
                True,
                False,
                pd.NA,
            ],
            "expedite": [
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
                "CASE-003:0:1",
            ],
            "case_id": [
                "CASE-001",
                "CASE-001",
                "CASE-002",
                "CASE-003",
                "CASE-003",
            ],
            "reaction_pt": [
                "Headache",
                "Nausea",
                "Headache",
                "Headache",
                "Dizziness",
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_case_count():
    data = make_test_data()

    result = calculate_case_count(data)

    assert result.metric_id == "case_count"
    assert result.value == 4
    assert result.population == "case"
    assert result.version == ANALYSIS_VERSION
    assert result.supporting_ids == [
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
    ]


def test_serious_case_count():
    data = make_test_data()

    result = calculate_serious_case_count(data)

    assert result.metric_id == "serious_case_count"
    assert result.value == 2
    assert result.supporting_ids == [
        "CASE-001",
        "CASE-002",
    ]


def test_non_serious_case_count():
    data = make_test_data()

    result = calculate_non_serious_case_count(data)

    assert result.metric_id == "non_serious_case_count"
    assert result.value == 1
    assert result.supporting_ids == [
        "CASE-003",
    ]


def test_serious_case_percentage_uses_case_denominator():
    data = make_test_data()

    result = calculate_serious_case_percentage(data)

    assert result.metric_id == "serious_case_percentage"
    assert result.value == 50.0
    assert result.supporting_ids == [
        "CASE-001",
        "CASE-002",
    ]


def test_expedited_case_count():
    data = make_test_data()

    result = calculate_expedited_case_count(data)

    assert result.metric_id == "expedited_case_count"
    assert result.value == 2
    assert result.supporting_ids == [
        "CASE-001",
        "CASE-003",
    ]


def test_expedited_case_percentage():
    data = make_test_data()

    result = calculate_expedited_case_percentage(data)

    assert result.metric_id == "expedited_case_percentage"
    assert result.value == 50.0
    assert result.supporting_ids == [
        "CASE-001",
        "CASE-003",
    ]


def test_reaction_count():
    data = make_test_data()

    result = calculate_reaction_count(data)

    assert result.metric_id == "reaction_count"
    assert result.value == 5
    assert result.population == "reaction"

    assert result.supporting_ids == [
        "CASE-001:0:0",
        "CASE-001:0:1",
        "CASE-002:0:0",
        "CASE-003:0:0",
        "CASE-003:0:1",
    ]


def test_unique_reaction_count():
    data = make_test_data()

    result = calculate_unique_reaction_count(data)

    assert result.metric_id == "unique_reaction_count"
    assert result.value == 3
    assert result.population == "reaction"


def test_distinct_case_reaction_pairs():
    data = make_test_data()

    result = calculate_distinct_case_reaction_pairs(data)

    assert result.metric_id == "distinct_case_reaction_pairs"
    assert result.value == 5
    assert result.population == "reaction"
    assert result.unit == "distinct_case_reaction_pairs"


def test_top_reactions():
    data = make_test_data()

    result = calculate_top_reactions(data)

    assert result.metric_id == "top_reactions"
    assert result.population == "reaction"

    assert result.value == [
        {
            "reaction_pt": "Headache",
            "occurrences": 3,
            "distinct_cases": 3,
        },
        {
            "reaction_pt": "Dizziness",
            "occurrences": 1,
            "distinct_cases": 1,
        },
        {
            "reaction_pt": "Nausea",
            "occurrences": 1,
            "distinct_cases": 1,
        },
    ]

    assert set(result.supporting_ids) == {
        "CASE-001",
        "CASE-002",
        "CASE-003",
    }


def test_top_reactions_has_deterministic_tie_breaking():
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-001", "CASE-002"],
            "serious": [True, True],
            "expedite": [True, True],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [
                "CASE-001:0:0",
                "CASE-002:0:0",
            ],
            "case_id": [
                "CASE-001",
                "CASE-002",
            ],
            "reaction_pt": [
                "Zebra reaction",
                "Alpha reaction",
            ],
        }
    )

    data = NormalizedData(
        cases=cases,
        reactions=reactions,
    )

    result = calculate_top_reactions(data)

    assert result.value == [
       {
    "reaction_pt": "Alpha reaction",
    "occurrences": 1,
    "distinct_cases": 1,
},
{
    "reaction_pt": "Zebra reaction",
    "occurrences": 1,
    "distinct_cases": 1,
},
    ]


def test_top_reaction_limit_is_explicit():
    assert TOP_REACTION_LIMIT == 10


def test_metric_definitions_are_present():
    data = make_test_data()

    results = [
        calculate_case_count(data),
        calculate_serious_case_count(data),
        calculate_non_serious_case_count(data),
        calculate_serious_case_percentage(data),
        calculate_expedited_case_count(data),
        calculate_expedited_case_percentage(data),
        calculate_reaction_count(data),
        calculate_unique_reaction_count(data),
        calculate_top_reactions(data),
    ]


    for result in results:
        assert result.definition.strip()
        assert result.version == ANALYSIS_VERSION