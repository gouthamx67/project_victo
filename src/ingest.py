from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# These are structural/core fields needed by the first reporting pipeline.
# We intentionally do NOT require every one of the 67 observed CSV columns.
REQUIRED_COLUMNS = {
    "safetyreportid",
    "receivedate",
    "serious",
    "seriousnessdeath",
    "seriousnesslifethreatening",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
    "fulfillexpeditecriteria",
    "occurcountry",
    "primarysource_reportercountry",
    "patient_patientonsetage",
    "patient_patientonsetageunit",
    "patient_patientsex",
    "patient_reaction_reactionmeddrapt",
    "patient_reaction_reactionoutcome",
    "patient_drug_drugcharacterization",
    "patient_drug_medicinalproduct",
    "patient_drug_actiondrug",
}


# These fields are expected to contain categorical values.
# We inspect their actual vocabulary rather than assuming it.
CATEGORICAL_COLUMNS = [
    "serious",
    "seriousnessdeath",
    "seriousnesslifethreatening",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
    "fulfillexpeditecriteria",
    "patient_patientsex",
    "patient_patientonsetageunit",
    "patient_patientagegroup",
    "patient_reaction_reactionoutcome",
    "patient_drug_drugcharacterization",
    "patient_drug_actiondrug",
]


@dataclass
class ValidationReport:
    """Results of validating one raw safety dataset."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"

        lines = [
            "=" * 72,
            "INGEST VALIDATION",
            "=" * 72,
            f"status: {status}",
            f"errors: {len(self.errors)}",
            f"warnings: {len(self.warnings)}",
        ]

        if self.errors:
            lines.append("")
            lines.append("ERRORS")
            for error in self.errors:
                lines.append(f"- {error}")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS")
            for warning in self.warnings:
                lines.append(f"- {warning}")

        if self.observations:
            lines.append("")
            lines.append("OBSERVATIONS")
            for key, value in self.observations.items():
                lines.append(f"- {key}: {value}")

        lines.append("=" * 72)
        return "\n".join(lines)


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV without prematurely interpreting domain values.

    All columns are initially read as strings. Missing values are represented
    by pandas NA. Downstream stages are responsible for explicit conversion.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype="string",
        keep_default_na=True,
        na_values=["", "NA", "N/A", "NULL", "null"],
    )

    if df.empty:
        raise ValueError("CSV contains zero rows.")

    return df


def _blank_count(series: pd.Series) -> int:
    """Count null or whitespace-only values exactly once."""
    normalized = series.astype("string").str.strip()
    return int(normalized.isna().sum() + normalized.eq("").sum())


def _validate_required_columns(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    actual = set(df.columns)
    missing = sorted(REQUIRED_COLUMNS - actual)

    if missing:
        report.add_error(
            "Missing required columns: " + ", ".join(missing)
        )

    duplicate_column_names = [
        column
        for column in df.columns
        if list(df.columns).count(column) > 1
    ]

    if duplicate_column_names:
        report.add_error(
            "Duplicate column names detected: "
            + ", ".join(sorted(set(duplicate_column_names)))
        )


def _validate_case_id(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    if "safetyreportid" not in df.columns:
        return

    series = df["safetyreportid"]

    blank_count = _blank_count(series)

    if blank_count:
        report.add_error(
            f"safetyreportid contains {blank_count} blank/null rows."
        )

    non_numeric = pd.to_numeric(series, errors="coerce").isna()

    if int(non_numeric.sum()) > 0:
        report.add_error(
            "safetyreportid contains non-numeric values: "
            f"{int(non_numeric.sum())} rows."
        )

    duplicate_rows = int(df.duplicated().sum())
    unique_cases = int(series.nunique(dropna=True))

    report.observations["rows"] = int(len(df))
    report.observations["unique_safetyreportid"] = unique_cases
    report.observations["duplicate_rows"] = duplicate_rows

    if duplicate_rows:
        report.add_error(
            f"Found {duplicate_rows} exact duplicate rows."
        )

    rows_per_case = series.value_counts(dropna=True)

    if not rows_per_case.empty:
        report.observations["max_rows_per_case"] = int(
            rows_per_case.max()
        )
        report.observations["multi_row_cases"] = int(
            (rows_per_case > 1).sum()
        )


def _validate_receivedate(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    if "receivedate" not in df.columns:
        return

    series = df["receivedate"].fillna("").str.strip()

    blank_count = int(series.eq("").sum())

    if blank_count:
        report.add_error(
            f"receivedate contains {blank_count} blank/null rows."
        )

    # Explicitly require eight digits.
    eight_digits = series.str.fullmatch(r"\d{8}", na=False)

    invalid_format_count = int((~eight_digits & series.ne("")).sum())

    if invalid_format_count:
        report.add_error(
            "receivedate contains values that are not YYYYMMDD-shaped: "
            f"{invalid_format_count} rows."
        )

    # Parse only values that have the expected shape.
    parsed = pd.to_datetime(
        series.where(eight_digits),
        format="%Y%m%d",
        errors="coerce",
    )

    invalid_calendar_count = int(
        (parsed.isna() & series.ne("")).sum()
    )

    if invalid_calendar_count:
        report.add_error(
            "receivedate contains invalid calendar dates: "
            f"{invalid_calendar_count} rows."
        )

    valid_dates = parsed.dropna()

    if not valid_dates.empty:
        report.observations["receivedate_min"] = (
            valid_dates.min().date().isoformat()
        )
        report.observations["receivedate_max"] = (
            valid_dates.max().date().isoformat()
        )

    report.observations["receivedate_parseable"] = int(
        valid_dates.shape[0]
    )


def _validate_required_field_missingness(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """
    Required columns may still legitimately contain missing domain values.

    Therefore we do not reject missingness globally. We only report it so
    normalization can apply field-specific rules later.
    """
    missingness: dict[str, int] = {}

    for column in sorted(REQUIRED_COLUMNS):
        if column not in df.columns:
            continue

        count = _blank_count(df[column])

        if count:
            missingness[column] = count

    report.observations["required_field_missingness"] = missingness


def _inspect_categorical_values(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    """
    Record actual categorical values.

    No domain vocabulary is imposed here because CP1 established that
    several fields need to be learned from the actual dataset before
    normalization rules are written.
    """
    categorical_values: dict[str, list[str]] = {}

    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .dropna()
            .astype("string")
            .str.strip()
        )

        values = values[values.ne("")]

        unique_values = sorted(values.unique().tolist())

        # Keep the validation output manageable for very high-cardinality
        # fields such as reaction outcomes.
        if len(unique_values) <= 100:
            categorical_values[column] = unique_values
        else:
            categorical_values[column] = [
                f"<{len(unique_values)} unique values; inspect upstream>"
            ]

    report.observations["categorical_values"] = categorical_values


def _inspect_age_fields(
    df: pd.DataFrame,
    report: ValidationReport,
) -> None:
    if "patient_patientonsetage" in df.columns:
        age = pd.to_numeric(
            df["patient_patientonsetage"],
            errors="coerce",
        )

        report.observations["numeric_age_parseable"] = int(
            age.notna().sum()
        )
        report.observations["numeric_age_missing_or_non_numeric"] = int(
            age.isna().sum()
        )

        if age.notna().any():
            report.observations["numeric_age_min"] = float(age.min())
            report.observations["numeric_age_max"] = float(age.max())

    if "patient_patientonsetageunit" in df.columns:
        values = (
            df["patient_patientonsetageunit"]
            .dropna()
            .astype("string")
            .str.strip()
        )
        values = values[values.ne("")]

        report.observations["age_unit_values"] = sorted(
            values.unique().tolist()
        )


def validate_raw_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Validate a dataframe at the raw-ingest boundary."""
    report = ValidationReport()

    if df.empty:
        report.add_error("Dataset contains zero rows.")
        return report

    _validate_required_columns(df, report)

    # Continue with fields that exist so the report remains useful even
    # when required columns are missing.
    _validate_case_id(df, report)
    _validate_receivedate(df, report)
    _validate_required_field_missingness(df, report)
    _inspect_categorical_values(df, report)
    _inspect_age_fields(df, report)

    return report


def ingest_and_validate(
    path: str | Path,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Load and validate a raw safety CSV.

    The dataframe is returned unchanged apart from pandas' representation
    of missing values. No domain normalization occurs here.
    """
    df = load_raw_csv(path)
    report = validate_raw_dataframe(df)

    return df, report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate a raw safety CSV."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the raw CSV file.",
    )

    args = parser.parse_args()

    try:
        _, report = ingest_and_validate(args.csv_path)
    except Exception as exc:
        print(f"INGEST ERROR: {exc}")
        return 1

    print(report.summary())

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())