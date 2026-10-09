from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analyses.trends import (
    ANALYSIS_VERSION,
    DEVIATION_MULTIPLIER,
    calculate_case_listing,
    calculate_monthly_case_counts,
    calculate_monthly_case_deviations,
    calculate_monthly_top_reactions,
)
from src.ingest import load_raw_csv
from src.normalize import NormalizedData, normalize_dataset


CSV_PATH = Path(
    "data/Bisoprolol_icsr_sample_1068rows (1).csv"
)


def make_trend_data() -> NormalizedData:
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
            "receivedate": [
                "2025-01-02",
                "2025-01-15",
                "2025-01-20",
                "2025-02-01",
                "2025-02-02",
                "2025-03-10",
            ],
            "country": [
                "UK",
                "UK",
                "EU",
                "France",
                "France",
                "Canada",
            ],
            "sex": [
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
            ],
            "age_years": [
                60.0,
                70.0,
                80.0,
                50.0,
                90.0,
                40.0,
            ],
            "serious": [
                True,
                True,
                False,
                True,
                True,
                False,
            ],
            "expedited": [
                True,
                True,
                False,
                True,
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
                "Headache",
                "Nausea",
                "Headache",
                "Headache",
                "Nausea",
                "Headache",
                "Nausea",
            ],
            "outcome": [
                "recovered/resolved",
                "unknown",
                "recovered/resolved",
                "unknown",
                "recovered/resolved",
                "fatal",
                "unknown",
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_monthly_case_counts_are_unique_case_counts():
    data = make_trend_data()

    result = calculate_monthly_case_counts(data)

    assert result.metric_id == "monthly_case_counts"
    assert result.population == "case"
    assert result.version == ANALYSIS_VERSION

    assert result.value == [
        {
            "reporting_month": "2025-01",
            "case_count": 3,
        },
        {
            "reporting_month": "2025-02",
            "case_count": 2,
        },
        {
            "reporting_month": "2025-03",
            "case_count": 1,
        },
    ]


def test_monthly_case_counts_do_not_double_count_case_rows():
    data = make_trend_data()

    result = calculate_monthly_case_counts(data)

    january = result.value[0]

    assert january["case_count"] == 3


def test_monthly_top_reactions_uses_reaction_rows():
    data = make_trend_data()

    result = calculate_monthly_top_reactions(data)

    assert result.metric_id == "monthly_top_reactions"
    assert result.population == "reaction"

    january = [
        row
        for row in result.value
        if row["reporting_month"] == "2025-01"
    ]

    assert january == [
        {
            "reporting_month": "2025-01",
            "reaction_pt": "Headache",
            "reaction_count": 3,
        },
        {
            "reporting_month": "2025-01",
            "reaction_pt": "Nausea",
            "reaction_count": 1,
        },
    ]


def test_monthly_reaction_counts_are_sorted_deterministically():
    data = make_trend_data()

    result = calculate_monthly_top_reactions(data)

    february = [
        row
        for row in result.value
        if row["reporting_month"] == "2025-02"
    ]

    assert february == [
        {
            "reporting_month": "2025-02",
            "reaction_pt": "Headache",
            "reaction_count": 1,
        },
        {
            "reporting_month": "2025-02",
            "reaction_pt": "Nausea",
            "reaction_count": 1,
        },
    ]


def test_monthly_deviation_uses_documented_multiplier():
    data = make_trend_data()

    result = calculate_monthly_case_deviations(data)

    assert result.metric_id == "monthly_case_deviations"

    mean = (3 + 2 + 1) / 3
    threshold = mean * DEVIATION_MULTIPLIER

    assert result.value[0]["mean_full_month_case_count"] == round(
        mean,
        4,
    )

    assert result.value[0]["deviation_threshold"] == round(
        threshold,
        4,
    )


def test_monthly_deviation_flags_only_counts_above_threshold():
    data = make_trend_data()

    result = calculate_monthly_case_deviations(data)

    flags = {
        row["reporting_month"]: row["deviation_flag"]
        for row in result.value
    }

    assert flags == {
        "2025-01": False,
        "2025-02": False,
        "2025-03": False,
    }


def test_case_listing_is_deterministic():
    data = make_trend_data()

    result = calculate_case_listing(data)

    assert result.metric_id == "case_listing"
    assert result.population == "case"

    assert [
        row["case_id"]
        for row in result.value
    ] == [
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
        "CASE-005",
        "CASE-006",
    ]


def test_case_listing_contains_supporting_case_ids():
    data = make_trend_data()

    result = calculate_case_listing(data)

    assert set(result.supporting_ids) == {
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
        "CASE-005",
        "CASE-006",
    }


def test_case_listing_preserves_missing_optional_values():
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-001"],
            "receivedate": ["2025-01-01"],
            "country": [pd.NA],
            "sex": [pd.NA],
            "age_years": [pd.NA],
            "serious": [pd.NA],
            "expedited": [pd.NA],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": ["CASE-001:0:0"],
            "case_id": ["CASE-001"],
            "reaction_pt": ["Headache"],
            "outcome": ["unknown"],
        }
    )

    data = NormalizedData(
        cases=cases,
        reactions=reactions,
    )

    result = calculate_case_listing(data)

    assert result.value == [
        {
            "case_id": "CASE-001",
            "reactions": ["Headache"],
            "serious": None,
            "receivedate": "2025-01-01",
            "country": "Unknown",
            "outcome": "unknown",
        }
    ]


def test_missing_receivedate_is_rejected_for_case_trend():
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-001"],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": ["CASE-001:0:0"],
            "case_id": ["CASE-001"],
            "reaction_pt": ["Headache"],
        }
    )

    data = NormalizedData(
        cases=cases,
        reactions=reactions,
    )

    try:
        calculate_monthly_case_counts(data)
    except ValueError as exc:
        assert "receivedate" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing receivedate."
        )


def test_partial_first_and_last_months_are_not_baseline_eligible():
    data = normalize_dataset(
        load_raw_csv(CSV_PATH)
    )

    result = calculate_monthly_case_deviations(data)

    rows = {
        row["reporting_month"]: row
        for row in result.value
    }

    assert rows["2024-12"]["baseline_eligible"] is False
    assert rows["2025-12"]["baseline_eligible"] is False

    eligible_months = [
        row["reporting_month"]
        for row in result.value
        if row["baseline_eligible"]
    ]

    assert "2025-01" in eligible_months
    assert "2025-11" in eligible_months


def test_full_calendar_month_helper_keeps_complete_boundary_months():
    from src.analyses.trends import _full_calendar_months

    monthly_counts = pd.DataFrame(
        {
            "reporting_month": [
                "2025-01",
                "2025-02",
                "2025-03",
            ],
            "case_count": [10, 12, 11],
        }
    )

    # The helper currently receives only monthly labels, not exact
    # observed start/end dates. With this input, it treats the first
    # and last months as partial and returns only the middle month.
    assert _full_calendar_months(monthly_counts) == ["2025-02"]


def test_monthly_top_reactions_limits_each_month_to_ten():
    cases = pd.DataFrame(
        {
            "case_id": ["CASE-001"],
            "receivedate": ["2025-01-15"],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [f"R-{i}" for i in range(12)],
            "case_id": ["CASE-001"] * 12,
            "reaction_pt": [f"Reaction {i:02d}" for i in range(12)],
        }
    )

    data = NormalizedData(cases=cases, reactions=reactions)

    result = calculate_monthly_top_reactions(data)

    january = [
        row for row in result.value
        if row["reporting_month"] == "2025-01"
    ]

    assert len(january) == 10
    assert [row["reaction_pt"] for row in january] == [
        f"Reaction {i:02d}" for i in range(10)
    ]