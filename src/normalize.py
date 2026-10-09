from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


AGE_UNIT_TO_YEARS = {
    "800": 10.0,
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 1.0 / 52.1775,
    "day": 1.0 / 365.25,
}


SERIOUSNESS_FLAG_COLUMNS = {
    "seriousnessdeath": "serious_death",
    "seriousnesslifethreatening": "serious_life_threatening",
    "seriousnesshospitalization": "serious_hospitalization",
    "seriousnessdisabling": "serious_disabling",
    "seriousnesscongenitalanomali": "serious_congenital_anomaly",
    "seriousnessother": "serious_other",
}


SAFETY_REPORT_ID_COLUMN = "safetyreportid"

REACTION_PT_COLUMN = "patient_reaction_reactionmeddrapt"
REACTION_OUTCOME_COLUMN = "patient_reaction_reactionoutcome"
REACTION_VERSION_COLUMN = (
    "patient_reaction_reactionmeddraversionpt"
)


@dataclass
class NormalizedData:
    """
    Analysis-ready representation of the validated raw dataset.

    cases:
        Exactly one row per safetyreportid.

    reactions:
        Exactly one row per logical reaction occurrence.
    """

    cases: pd.DataFrame
    reactions: pd.DataFrame


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_text(value: object) -> str | pd.NA:
    """Strip whitespace and convert blank values to pandas.NA."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    return text


def _normalize_yes_no(
    series: pd.Series,
    *,
    true_value: str = "yes",
    false_value: str = "no",
) -> pd.Series:
    """
    Convert a verified yes/no source field into nullable booleans.

    Unexpected values are not guessed; they become <NA>.
    """
    normalized = series.astype("string").str.strip().str.lower()

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[normalized == true_value] = True
    result.loc[normalized == false_value] = False

    return result


def _normalize_serious(series: pd.Series) -> pd.Series:
    """Normalize the verified serious/not serious vocabulary."""
    normalized = series.astype("string").str.strip().str.lower()

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="boolean",
    )

    result.loc[normalized == "serious"] = True
    result.loc[normalized == "not serious"] = False

    return result


def _unique_non_missing(values: Iterable[object]) -> list[str]:
    """Return sorted unique non-empty values."""
    cleaned = set()

    for value in values:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if text:
            cleaned.add(text)

    return sorted(cleaned)


def _case_single_value(
    group: pd.DataFrame,
    column: str,
) -> tuple[object, str]:
    """
    Return a case-level value only when exactly one
    non-missing value exists.

    Status:
        single
        missing
        ambiguous
    """
    values = _unique_non_missing(group[column].tolist())

    if not values:
        return pd.NA, "missing"

    if len(values) == 1:
        return values[0], "single"

    return pd.NA, "ambiguous"


COUNTRY_CODE_NAMES = {
    "IE": "Ireland",
    "RO": "Romania",
    "SA": "Saudi Arabia",
    "HR": "Croatia",
    "UA": "Ukraine",
    "FI": "Finland",
}


def _normalize_country_name(value: object) -> str | pd.NA:
    """Normalize known country codes and standardize country display names."""
    cleaned = _clean_text(value)

    if pd.isna(cleaned):
        return pd.NA

    text = str(cleaned).strip()
    upper = text.upper()

    if upper in COUNTRY_CODE_NAMES:
        return COUNTRY_CODE_NAMES[upper]

    if upper == "EU":
        return "European Union (region)"

    # Standardize country-name casing without changing the source meaning.
    return text.title()


def _case_country_value(
    group: pd.DataFrame,
    column: str = "occurcountry",
) -> tuple[object, str]:
    """
    Return a country only when exactly one normalized value exists.

    Missing values remain missing. Multiple distinct values within a case
    are marked ambiguous rather than resolved arbitrarily.
    """
    values = []

    for value in group[column]:
        normalized = _normalize_country_name(value)

        if pd.isna(normalized):
            continue

        values.append(str(normalized))

    unique_values = sorted(set(values))

    if not unique_values:
        return pd.NA, "missing"

    if len(unique_values) == 1:
        return unique_values[0], "single"

    return pd.NA, "ambiguous"


def _normalize_age_pair(
    age_value: object,
    unit_value: object,
) -> tuple[object, object, object]:
    """
    Normalize one age/unit pair.

    Returns:
        raw_age, normalized_unit, age_years
    """
    if pd.isna(age_value) or pd.isna(unit_value):
        return pd.NA, pd.NA, pd.NA

    age = pd.to_numeric(
        pd.Series([age_value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(age):
        return pd.NA, pd.NA, pd.NA

    unit = str(unit_value).strip().lower()

    multiplier = AGE_UNIT_TO_YEARS.get(unit)

    if multiplier is None:
        return float(age), unit, pd.NA

    return (
        float(age),
        unit,
        float(age) * multiplier,
    )


def _case_age(group: pd.DataFrame) -> dict[str, object]:
    """
    Determine a case-level age without selecting arbitrarily
    among conflicting valid age/unit pairs.
    """
    pairs: set[tuple[float, str]] = set()

    for _, row in group.iterrows():
        age = pd.to_numeric(
            pd.Series([row["patient_patientonsetage"]]),
            errors="coerce",
        ).iloc[0]

        unit_raw = row["patient_patientonsetageunit"]

        if pd.isna(age) or pd.isna(unit_raw):
            continue

        unit = str(unit_raw).strip().lower()

        if unit not in AGE_UNIT_TO_YEARS:
            continue

        pairs.add((float(age), unit))

    if not pairs:
        return {
            "onset_age": pd.NA,
            "onset_age_unit": pd.NA,
            "age_years": pd.NA,
            "age_status": "missing",
        }

    if len(pairs) > 1:
        return {
            "onset_age": pd.NA,
            "onset_age_unit": pd.NA,
            "age_years": pd.NA,
            "age_status": "ambiguous",
        }

    age, unit = next(iter(pairs))

    return {
        "onset_age": age,
        "onset_age_unit": unit,
        "age_years": age * AGE_UNIT_TO_YEARS[unit],
        "age_status": "single",
    }


def _case_seriousness_flags(
    group: pd.DataFrame,
) -> dict[str, object]:
    """
    Aggregate seriousness flags conservatively.

    any YES -> True
    all observed NO -> False
    no usable values -> <NA>

    Both YES and NO are retained as a conflict flag.
    """
    result: dict[str, object] = {}

    for source_column, output_column in SERIOUSNESS_FLAG_COLUMNS.items():
        values = (
            group[source_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        has_yes = bool((values == "yes").any())
        has_no = bool((values == "no").any())

        if has_yes:
            result[output_column] = True
        elif has_no:
            result[output_column] = False
        else:
            result[output_column] = pd.NA

        result[f"{output_column}_conflict"] = (
            has_yes and has_no
        )

    return result


# ---------------------------------------------------------------------------
# Case normalization
# ---------------------------------------------------------------------------

def normalize_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Create exactly one normalized case record per safetyreportid."""
    records: list[dict[str, object]] = []

    for case_id, group in df.groupby(
        "safetyreportid",
        sort=True,
        dropna=False,
    ):
        record: dict[str, object] = {
            "case_id": str(case_id),
            "source_row_count": len(group),
        }

        receivedate = pd.to_datetime(
            group["receivedate"],
            format="%Y%m%d",
            errors="coerce",
        )

        unique_dates = receivedate.dropna().unique()

        if len(unique_dates) == 1:
            date = pd.Timestamp(unique_dates[0])
            record["receivedate"] = date
            record["report_month"] = date.strftime("%Y-%m")
            record["receivedate_status"] = "single"
        elif len(unique_dates) == 0:
            record["receivedate"] = pd.NaT
            record["report_month"] = pd.NA
            record["receivedate_status"] = "missing"
        else:
            record["receivedate"] = pd.NaT
            record["report_month"] = pd.NA
            record["receivedate_status"] = "ambiguous"

        serious_values = (
            group["serious"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        serious_unique = sorted(
            set(serious_values.dropna().tolist())
        )

        if serious_unique == ["serious"]:
            record["serious"] = True
            record["serious_status"] = "single"
        elif serious_unique == ["not serious"]:
            record["serious"] = False
            record["serious_status"] = "single"
        elif not serious_unique:
            record["serious"] = pd.NA
            record["serious_status"] = "missing"
        else:
            record["serious"] = pd.NA
            record["serious_status"] = "ambiguous"

        record.update(_case_seriousness_flags(group))

        expedite_values = (
            group["fulfillexpeditecriteria"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        expedite_unique = sorted(
            set(expedite_values.dropna().tolist())
        )

        if expedite_unique == ["yes"]:
            record["expedite"] = True
            record["expedite_status"] = "single"
        elif expedite_unique == ["no"]:
            record["expedite"] = False
            record["expedite_status"] = "single"
        elif not expedite_unique:
            record["expedite"] = pd.NA
            record["expedite_status"] = "missing"
        else:
            record["expedite"] = pd.NA
            record["expedite_status"] = "ambiguous"

        sex, sex_status = _case_single_value(
            group,
            "patient_patientsex",
        )

        record["sex"] = (
            str(sex).strip().lower()
            if not pd.isna(sex)
            else pd.NA
        )
        record["sex_status"] = sex_status

        record.update(_case_age(group))

        age_group, age_group_status = _case_single_value(
            group,
            "patient_patientagegroup",
        )

        record["age_group"] = (
            str(age_group).strip().lower()
            if not pd.isna(age_group)
            else pd.NA
        )
        record["age_group_status"] = age_group_status

        occurcountry, occurcountry_status = _case_country_value(
            group,
            "occurcountry",
        )

        reporter_country, reporter_country_status = (
            _case_country_value(
                group,
                "primarysource_reportercountry",
            )
        )

        record["occurcountry"] = occurcountry
        record["occurcountry_status"] = occurcountry_status

        record["reporter_country"] = reporter_country
        record["reporter_country_status"] = (
            reporter_country_status
        )

        row_occur = (
            group["occurcountry"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        row_reporter = (
            group["primarysource_reportercountry"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        mismatch = (
            row_occur.notna()
            & row_reporter.notna()
            & row_occur.ne("")
            & row_reporter.ne("")
            & row_occur.ne(row_reporter)
        )

        record["country_mismatch"] = bool(mismatch.any())

        records.append(record)

    cases = pd.DataFrame(records)

    if cases.empty:
        raise ValueError("Normalization produced zero cases.")

    return cases


# ---------------------------------------------------------------------------
# Reaction parsing
# ---------------------------------------------------------------------------

EMBEDDED_COMMA_PTS = {
    "Hallucinations, mixed",
    "Hallucination, visual",
    "Hallucination, auditory",
}

_ALIGNMENT_FILL = ""


def _split_source_list(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        token.strip()
        for token in text.split(",")
        if token.strip()
    ]


def _parse_reaction_alignment_field(value: object) -> list[str]:
    return _split_source_list(value)


def _reconstruct_reaction_pts(
    raw_value: object,
) -> tuple[list[str], int]:
    """
    Convert raw comma-delimited PT tokens into logical MedDRA PTs.

    Returns:
        logical PT list,
        number of embedded-comma PTs reconstructed
    """
    tokens = _split_source_list(raw_value)

    logical_pts: list[str] = []
    reconstruction_count = 0

    index = 0

    while index < len(tokens):
        if index + 1 < len(tokens):
            candidate = f"{tokens[index]}, {tokens[index + 1]}"

            if candidate in EMBEDDED_COMMA_PTS:
                logical_pts.append(candidate)
                reconstruction_count += 1
                index += 2
                continue

        logical_pts.append(tokens[index])
        index += 1

    return logical_pts, reconstruction_count


def _aligned_reaction_fields(
    row: pd.Series,
) -> tuple[list[str], list[str], list[str], int]:
    """
    Parse one source row and align logical PTs, outcomes, and versions.

    Embedded-comma PTs are reconstructed before positional alignment.

    Missing optional outcomes or versions remain missing; they are never
    invented. Extra source outcomes or versions are reported through
    alignment status by the caller.
    """
    pts, reconstruction_count = _reconstruct_reaction_pts(
        row.get(REACTION_PT_COLUMN)
    )

    outcomes = _parse_reaction_alignment_field(
        row.get(REACTION_OUTCOME_COLUMN)
    )

    # Some valid callers/test fixtures do not provide a MedDRA version
    # column. Preserve missing versions instead of rejecting the entire row.
    versions = _parse_reaction_alignment_field(
        row.get(REACTION_VERSION_COLUMN)
    )

    # Short outcome lists are allowed; the caller is responsible for
    # marking missing outcomes. Excess outcomes are preserved so the
    # caller can report an explicit alignment status rather than
    # silently dropping source information.
    #
    # Pad versions up to len(pts) so positional alignment never truncates
    # a PT just because a version value was missing.
    if len(versions) < len(pts):
        versions.extend(
            [_ALIGNMENT_FILL] * (len(pts) - len(versions))
        )

    return pts, outcomes, versions, reconstruction_count


def _normalize_optional_string(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    return text if text else None


def _reaction_alignment_status(
    pt_value: str | None,
    outcome_value: str | None,
    version_value: str | None,
) -> str:
    """
    Describe how a single reaction row aligned against its source lists.

    Values:
        aligned
            PT, outcome, and version were all present.
        missing_outcome
            PT present, outcome absent, version present.
        missing_version
            PT and outcome present, version absent.
        missing_outcome_and_version
            PT present, outcome and version both absent.
        orphan_outcome
            Outcome present but no PT. Kept for traceability; it is not
            a logical reaction occurrence.
    """
    if pt_value is None:
        return "orphan_outcome"

    if outcome_value is None and version_value is None:
        return "missing_outcome_and_version"

    if outcome_value is None:
        return "missing_outcome"

    if version_value is None:
        return "missing_version"

    return "aligned"


def normalize_reactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize one logical reaction occurrence per output row.

    Raw PT commas are corrected before positional alignment.

    One output row is emitted per logical PT. Alignment between the PT
    list and the outcome list is reported through
    ``outcome_alignment_status``:

        aligned
            PT and outcome both present at this position.
        missing_source_outcome
            The outcome list ran out, or the outcome token was blank.
        extra_source_outcome
            The outcome list is longer than the PT list for this source
            row, so at least one source outcome is not attached to a PT.

    The normalized output keeps the original source column names for
    traceability and also exposes stable analysis-facing aliases:

        reaction_pt
        outcome
        meddra_version
    """
    required = {
        SAFETY_REPORT_ID_COLUMN,
        REACTION_PT_COLUMN,
        REACTION_OUTCOME_COLUMN,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required reaction columns: {missing}"
        )

    records: list[dict[str, object]] = []

    for row_index, row in df.iterrows():
        safety_report_id = _normalize_optional_string(
            row.get(SAFETY_REPORT_ID_COLUMN)
        )

        if safety_report_id is None:
            raise ValueError(
                f"Missing safetyreportid at source row {row_index}"
            )

        (
            pts,
            outcomes,
            versions,
            reconstruction_count,
        ) = _aligned_reaction_fields(row)

        for reaction_index, pt in enumerate(pts):
            outcome = (
                outcomes[reaction_index]
                if reaction_index < len(outcomes)
                else None
            )

            version = (
                versions[reaction_index]
                if reaction_index < len(versions)
                else None
            )

            outcome_value = _normalize_optional_string(outcome)
            version_value = _normalize_optional_string(version)

            if len(outcomes) > len(pts):
                alignment_status = "extra_source_outcome"
            elif reaction_index >= len(outcomes):
                alignment_status = "missing_source_outcome"
            elif outcome_value is None:
                alignment_status = "missing_source_outcome"
            else:
                alignment_status = "aligned"

            is_reconstructed_pt = pt in EMBEDDED_COMMA_PTS

            records.append(
                {
                    # Stable normalized identifiers.
                    "reaction_id": (
                        f"{safety_report_id}:"
                        f"{row_index}:"
                        f"{reaction_index}"
                    ),
                    "case_id": safety_report_id,

                    # Row / position provenance.
                    "source_row_index": row_index,
                    "reaction_position": reaction_index,

                    # Stable analysis-facing normalized fields.
                    "reaction_pt": pt,
                    "outcome": outcome_value,
                    "meddra_version": version_value,

                    # Original source fields retained for traceability.
                    SAFETY_REPORT_ID_COLUMN: safety_report_id,
                    REACTION_PT_COLUMN: pt,
                    REACTION_OUTCOME_COLUMN: outcome_value,
                    REACTION_VERSION_COLUMN: version_value,

                    # Validation / provenance metadata.
                    "outcome_present": outcome_value is not None,
                    "meddra_version_present": (
                        version_value is not None
                    ),
                    "reaction_alignment_status": "aligned",
                    "outcome_alignment_status": alignment_status,
                    "pt_reconstruction_status": (
                        "reconstructed_embedded_comma"
                        if is_reconstructed_pt
                        else "no_reconstruction"
                    ),
                    "pt_reconstruction_count": (
                        1 if is_reconstructed_pt else 0
                    ),
                    "source_pt_reconstruction_count": (
                        reconstruction_count
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def normalize_dataset(df: pd.DataFrame) -> NormalizedData:
    """
    Normalize a validated raw dataframe into case and reaction tables.

    No LLM or external service is used.
    """
    cases = normalize_cases(df)
    reactions = normalize_reactions(df)

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def main() -> int:
    import argparse

    from src.ingest import ingest_and_validate

    parser = argparse.ArgumentParser(
        description="Normalize a validated safety CSV."
    )

    parser.add_argument(
        "csv_path",
        help="Path to the raw CSV file.",
    )

    args = parser.parse_args()

    df, validation = ingest_and_validate(args.csv_path)

    if not validation.is_valid:
        print(validation.summary())
        return 1

    normalized = normalize_dataset(df)

    print("=" * 72)
    print("NORMALIZATION")
    print("=" * 72)
    print(f"cases: {len(normalized.cases)}")
    print(f"reactions: {len(normalized.reactions)}")

    print(
        "ambiguous country cases:",
        int(
            (
                normalized.cases["occurcountry_status"]
                == "ambiguous"
            ).sum()
        ),
    )

    print(
        "ambiguous age cases:",
        int(
            (
                normalized.cases["age_status"]
                == "ambiguous"
            ).sum()
        ),
    )

    print(
        "reaction rows reconstructed:",
        int(
            (
                normalized.reactions[
                    "pt_reconstruction_status"
                ]
                == "reconstructed_embedded_comma"
            ).sum()
        ),
    )

    print("outcome alignment status:")

    status_counts = (
        normalized.reactions["outcome_alignment_status"]
        .value_counts(dropna=False)
        .sort_index()
    )

    for status, count in status_counts.items():
        print(f"  {status}: {int(count)}")

    non_aligned = int(
        (
            normalized.reactions["outcome_alignment_status"]
            != "aligned"
        ).sum()
    )

    print(f"  non-aligned rows: {non_aligned}")

    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())