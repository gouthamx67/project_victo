from pathlib import Path

import pandas as pd

from src.ingest import (
    REQUIRED_COLUMNS,
    ingest_and_validate,
    load_raw_csv,
    validate_raw_dataframe,
)


def make_minimal_dataframe() -> pd.DataFrame:
    """Build a structurally valid synthetic dataset for unit tests."""
    data = {
        column: ["value"]
        for column in REQUIRED_COLUMNS
    }

    data["safetyreportid"] = ["12345"]
    data["receivedate"] = ["20241227"]

    return pd.DataFrame(data)


def test_receivedate_uses_yyyymmdd_parsing():
    df = make_minimal_dataframe()

    report = validate_raw_dataframe(df)

    assert report.is_valid
    assert report.observations["receivedate_min"] == "2024-12-27"
    assert report.observations["receivedate_max"] == "2024-12-27"


def test_invalid_receivedate_is_rejected():
    df = make_minimal_dataframe()
    df["receivedate"] = ["20241399"]

    report = validate_raw_dataframe(df)

    assert not report.is_valid
    assert any(
        "invalid calendar dates" in error
        for error in report.errors
    )


def test_wrong_receivedate_shape_is_rejected():
    df = make_minimal_dataframe()
    df["receivedate"] = ["2024-12-27"]

    report = validate_raw_dataframe(df)

    assert not report.is_valid
    assert any(
        "YYYYMMDD-shaped" in error
        for error in report.errors
    )


def test_missing_required_column_is_rejected():
    df = make_minimal_dataframe()
    df = df.drop(columns=["safetyreportid"])

    report = validate_raw_dataframe(df)

    assert not report.is_valid
    assert any(
        "safetyreportid" in error
        for error in report.errors
    )


def test_blank_case_id_is_rejected():
    df = make_minimal_dataframe()
    df["safetyreportid"] = [pd.NA]

    report = validate_raw_dataframe(df)

    assert not report.is_valid
    assert any(
        "safetyreportid" in error
        for error in report.errors
    )


def test_exact_duplicate_rows_are_rejected():
    df = pd.concat(
        [make_minimal_dataframe(), make_minimal_dataframe()],
        ignore_index=True,
    )

    report = validate_raw_dataframe(df)

    assert not report.is_valid
    assert any(
        "exact duplicate rows" in error
        for error in report.errors
    )


def test_multiple_rows_per_case_are_allowed():
    df = pd.concat(
        [make_minimal_dataframe(), make_minimal_dataframe()],
        ignore_index=True,
    )

    df.loc[1, "safetyreportid"] = "12345"
    df.loc[1, "receivedate"] = "20241228"

    report = validate_raw_dataframe(df)

    # These are not exact duplicates because receivedate differs.
    assert report.is_valid
    assert report.observations["unique_safetyreportid"] == 1
    assert report.observations["multi_row_cases"] == 1


def test_missing_domain_values_are_observed_not_globally_rejected():
    df = make_minimal_dataframe()

    df["occurcountry"] = [pd.NA]
    df["patient_patientonsetage"] = [pd.NA]
    df["patient_patientonsetageunit"] = [pd.NA]

    report = validate_raw_dataframe(df)

    assert report.is_valid

    missingness = report.observations["required_field_missingness"]

    assert missingness["occurcountry"] == 1
    assert missingness["patient_patientonsetage"] == 1
    assert missingness["patient_patientonsetageunit"] == 1


def test_categorical_values_are_not_silently_normalized():
    df = make_minimal_dataframe()

    df["serious"] = ["SERIOUS_CUSTOM_VALUE"]
    df["patient_patientsex"] = ["CustomSex"]

    report = validate_raw_dataframe(df)

    assert report.is_valid

    categorical = report.observations["categorical_values"]

    assert categorical["serious"] == ["SERIOUS_CUSTOM_VALUE"]
    assert categorical["patient_patientsex"] == ["CustomSex"]


def test_synthetic_csv_file_is_loaded_and_validated(tmp_path: Path):
    source = make_minimal_dataframe()
    csv_path = tmp_path / "synthetic_ingest.csv"
    source.to_csv(csv_path, index=False)

    df, report = ingest_and_validate(csv_path)

    assert report.is_valid, report.summary()
    assert len(df) == 1
    assert df["safetyreportid"].nunique() == 1
    assert report.observations["receivedate_min"] == "2024-12-27"
    assert report.observations["receivedate_max"] == "2024-12-27"
