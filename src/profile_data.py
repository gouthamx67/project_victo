
import argparse
from pathlib import Path

import pandas as pd


def value_counts(df: pd.DataFrame, column: str) -> None:
    print(f"\n### VALUE COUNTS: {column}")

    if column not in df.columns:
        print("COLUMN NOT FOUND")
        return

    counts = df[column].value_counts(dropna=False)

    for value, count in counts.items():
        print(f"{value!r}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the raw safety CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise SystemExit(f"CSV not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path, low_memory=False)

    print("=" * 80)
    print("DATASET PROFILE")
    print("=" * 80)

    print("\n### FILE")
    print(args.csv_path.resolve())

    print("\n### SHAPE")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")

    print("\n### COLUMNS")
    for column in df.columns:
        print(f"- {column}")

    print("\n### DTYPES")
    print(df.dtypes.to_string())

    print("\n### NULL COUNTS")
    print(df.isna().sum().to_string())

    print("\n### NULL PERCENTAGES")
    print((df.isna().mean() * 100).round(2).to_string())

    print("\n### DUPLICATE ROWS")
    print(f"duplicate_rows: {df.duplicated().sum()}")

    print("\n### SAFETYREPORTID")

    id_column = "safetyreportid"

    if id_column in df.columns:
        print(f"unique_safetyreportid: {df[id_column].nunique(dropna=True)}")
        print(f"missing_safetyreportid: {df[id_column].isna().sum()}")

        rows_per_case = df.groupby(id_column, dropna=True).size()

        print("\nrows_per_case:")
        print(rows_per_case.describe().to_string())

        print("\nrows_per_case_frequency:")
        print(rows_per_case.value_counts().sort_index().to_string())
    else:
        print("COLUMN NOT FOUND: safetyreportid")

    print("\n### RECEIVEDATE")

    date_column = "receivedate"

    if date_column in df.columns:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

        print(f"parseable_dates: {parsed_dates.notna().sum()}")
        print(f"unparseable_dates: {parsed_dates.isna().sum()}")

        if parsed_dates.notna().any():
            print(f"minimum_date: {parsed_dates.min()}")
            print(f"maximum_date: {parsed_dates.max()}")

            print("\nreceivedate_value_examples:")
            print(df[date_column].dropna().head(10).to_string(index=False))
    else:
        print("COLUMN NOT FOUND: receivedate")

    requested_value_columns = [
        "serious",
        "fulfillexpeditecriteria",
        "patient_patientsex",
        "patient_reaction_reactionoutcome",
        "occurcountry",
        "primarysource_reportercountry",
        "patient_patientagegroup",
    ]

    for column in requested_value_columns:
        value_counts(df, column)

    print("\n### AGE NUMERIC FIELD")

    age_column = "patient_patientonsetage"

    if age_column in df.columns:
        age_numeric = pd.to_numeric(df[age_column], errors="coerce")

        print(f"parseable_numeric_age: {age_numeric.notna().sum()}")
        print(f"missing_or_non_numeric_age: {age_numeric.isna().sum()}")

        if age_numeric.notna().any():
            print("\nage_describe:")
            print(age_numeric.describe().to_string())

            print("\nage_value_counts_top_30:")
            print(age_numeric.value_counts().head(30).to_string())
    else:
        print("COLUMN NOT FOUND: patient_patientonsetage")

    print("\n### COUNTRY MISMATCH")

    occur_column = "occurcountry"
    reporter_column = "primarysource_reportercountry"

    if occur_column in df.columns and reporter_column in df.columns:
        both_present = df[occur_column].notna() & df[reporter_column].notna()

        occur = (
            df.loc[both_present, occur_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        reporter = (
            df.loc[both_present, reporter_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        mismatch = occur != reporter

        print(f"rows_with_both_countries: {both_present.sum()}")
        print(f"country_mismatches: {mismatch.sum()}")

        if both_present.sum():
            print(f"mismatch_rate_percent: {mismatch.mean() * 100:.2f}")

            print("\n### MISMATCH PAIRS")

            mismatch_pairs = pd.DataFrame(
                {
                    "occur": occur[mismatch],
                    "reporter": reporter[mismatch],
                }
            )

            if len(mismatch_pairs):
                print(
                    mismatch_pairs.value_counts().to_string()
                )
            else:
                print("No mismatches.")
    else:
        print(
            f"Could not compare countries: "
            f"{occur_column!r} and/or {reporter_column!r} missing."
        )

    print("\n### REACTION FIELD")

    reaction_column = "patient_reaction_reactionmeddrapt"

    if reaction_column in df.columns:
        print(
            f"unique_reaction_values: "
            f"{df[reaction_column].nunique(dropna=True)}"
        )
        print(
            f"missing_reaction_values: "
            f"{df[reaction_column].isna().sum()}"
        )

        print("\ntop_30_reactions:")
        print(
            df[reaction_column]
            .value_counts(dropna=False)
            .head(30)
            .to_string()
        )
    else:
        print(f"COLUMN NOT FOUND: {reaction_column}")

    print("\n### ALL COLUMN UNIQUE COUNTS")
    print(df.nunique(dropna=True).sort_values().to_string())

    print("\n" + "=" * 80)
    print("END PROFILE")
    print("=" * 80)


if __name__ == "__main__":
    main()
PY