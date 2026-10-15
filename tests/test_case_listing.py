from pathlib import Path

from src.ingest import load_raw_csv
from src.normalize import normalize_dataset
from src.analyses.reporting import calculate_case_listing


CSV_PATH = Path("tests/fixtures/synthetic_icsr.csv")


def load_normalized_data():
    return normalize_dataset(
        load_raw_csv(CSV_PATH)
    )


def test_case_listing_has_required_fields():
    data = load_normalized_data()

    result = calculate_case_listing(data)

    assert result.unit == "case_listing"
    assert len(result.value) == len(data.cases)
    assert len(result.supporting_ids) == len(data.cases)

    required_fields = {
        "case_id",
        "reactions",
        "serious",
        "receivedate",
        "country",
        "outcome",
    }

    assert set(result.value[0]) == required_fields


def test_case_listing_is_deterministically_ordered():
    data = load_normalized_data()

    result = calculate_case_listing(data)

    dates_and_ids = [
        (
            row["receivedate"],
            row["case_id"],
        )
        for row in result.value
    ]

    assert dates_and_ids == sorted(dates_and_ids)


def test_case_listing_contains_reactions():
    data = load_normalized_data()

    result = calculate_case_listing(data)

    assert any(
        row["reactions"]
        for row in result.value
    )

    assert all(
        isinstance(row["reactions"], list)
        for row in result.value
    )


def test_case_listing_has_deterministic_seriousness_and_outcome():
    data = load_normalized_data()

    result = calculate_case_listing(data)

    assert all(
        row["serious"] is None
        or isinstance(row["serious"], bool)
        for row in result.value
    )

    assert all(
        isinstance(row["outcome"], str)
        for row in result.value
    )