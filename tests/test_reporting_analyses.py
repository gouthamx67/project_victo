from pathlib import Path

import pandas as pd

from src.ingest import load_raw_csv
from src.normalize import normalize_dataset
from src.analyses.reporting import (
    AGE_BUCKETS,
    calculate_age_distribution,
    calculate_country_distribution,
    calculate_history_of_actions,
    calculate_reporter_qualification_distribution,
    calculate_reporting_period,
    calculate_sex_distribution,
)


CSV_PATH = Path("tests/fixtures/synthetic_icsr.csv")


def load_normalized_data():
    return normalize_dataset(
        load_raw_csv(CSV_PATH)
    )


def test_reporting_period_matches_observed_dataset():
    data = load_normalized_data()

    result = calculate_reporting_period(data)

    assert result.unit == "cases"
    assert result.value == {
        "start_date": "2024-12-27",
        "end_date": "2025-12-26",
        "days": 365,
    }
    assert len(result.supporting_ids) == len(data.cases)


def test_age_distribution_uses_required_buckets():
    data = load_normalized_data()

    result = calculate_age_distribution(data)

    assert result.unit == "cases"

    assert [
        row["age_bucket"]
        for row in result.value
    ] == list(AGE_BUCKETS)

    assert sum(
        row["count"]
        for row in result.value
    ) == len(data.cases)

    unknown = next(
        row["count"]
        for row in result.value
        if row["age_bucket"] == "unknown"
    )

    assert unknown == 0


def test_sex_distribution_is_case_level():
    data = load_normalized_data()

    result = calculate_sex_distribution(data)

    assert result.unit == "cases"

    expected_sexes = {
        label
        if label not in {"", "nan", "none", "<na>"}
        else "unknown"
        for label in (
            str(value).strip().lower()
            for value in data.cases["sex"].tolist()
        )
    }

    assert {row["sex"] for row in result.value} == expected_sexes
    assert sum(row["count"] for row in result.value) == len(data.cases)


def test_country_distribution_retains_eu_region():
    data = load_normalized_data()

    result = calculate_country_distribution(data)

    assert result.unit == "cases"

    rows = {
        row["country_or_region"]: row["count"]
        for row in result.value
    }

    assert rows
    assert sum(rows.values()) == len(data.cases)


def test_reporter_qualification_distribution_is_case_level():
    data = load_normalized_data()

    result = calculate_reporter_qualification_distribution(data)

    assert result.unit == "cases"
    assert sum(row["count"] for row in result.value) == len(data.cases)

    qualifications = {
        row["reporter_qualification"]
        for row in result.value
    }

    assert qualifications
    assert all(value == value.lower() for value in qualifications)


def test_history_of_actions_is_explicitly_none_provided():
    data = load_normalized_data()

    result = calculate_history_of_actions(data)

    assert result.value == {
        "status": "none_provided",
        "statement": (
            "No history of actions was provided in the supplied "
            "reporting data."
        ),
    }

    assert result.supporting_ids == []