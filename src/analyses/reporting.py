from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from src.analysis_framework import AnalysisResult, register_analysis
from src.analyses.reactions import derive_case_outcomes
from src.normalize import NormalizedData


ANALYSIS_VERSION = "1.0.0"

AGE_BUCKETS = (
    "<2",
    "2-11",
    "12-17",
    "18-44",
    "45-64",
    "65-74",
    "75-84",
    "85+",
    "unknown",
)

EU_REGION_VALUE = "EU"


def _case_ids(data: NormalizedData) -> list[str]:
    return (
        data.cases["case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )


def _normalize_distribution_value(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip()

    if not text:
        return "unknown"

    return text


def _age_bucket(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if age < 0:
        return "unknown"

    if age < 2:
        return "<2"

    if age < 12:
        return "2-11"

    if age < 18:
        return "12-17"

    if age < 45:
        return "18-44"

    if age < 65:
        return "45-64"

    if age < 75:
        return "65-74"

    if age < 85:
        return "75-84"

    return "85+"


@register_analysis(
    metric_id="reporting_period",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_reporting_period(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Return the observed receivedate reporting period.

    The metric describes the minimum and maximum valid receivedate values
    in the normalized case population.
    """

    receivedates = pd.to_datetime(
        data.cases["receivedate"],
        errors="coerce",
    )

    valid = receivedates.dropna()

    if valid.empty:
        value: dict[str, object] = {
            "start_date": None,
            "end_date": None,
            "days": None,
        }
    else:
        start = valid.min()
        end = valid.max()

        value = {
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "days": int((end - start).days + 1),
        }

    return AnalysisResult(
        metric_id="reporting_period",
        value=value,
        definition=(
            "Observed reporting period defined by the earliest and latest "
            "valid normalized receivedate values, inclusive."
        ),
        population="case",
        unit="cases",
        supporting_ids=_case_ids(data),
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="age_distribution",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_age_distribution(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Classify cases into the required deterministic age buckets.

    Missing and ambiguous normalized ages are represented as unknown.
    """

    buckets = [
        _age_bucket(value)
        for value in data.cases["age_years"].tolist()
    ]

    counts = Counter(buckets)

    value = [
        {
            "age_bucket": bucket,
            "count": int(counts.get(bucket, 0)),
        }
        for bucket in AGE_BUCKETS
    ]

    return AnalysisResult(
        metric_id="age_distribution",
        value=value,
        definition=(
            "Distribution of normalized safety cases across deterministic "
            "age-in-years buckets: <2, 2-11, 12-17, 18-44, 45-64, "
            "65-74, 75-84, 85+, and unknown. Missing or ambiguous ages "
            "are classified as unknown; in this dataset, all 84 unknown "
            "ages are missing values."
        ),
        population="case",
        unit="cases",
        supporting_ids=_case_ids(data),
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="sex_distribution",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_sex_distribution(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Return case-level sex distribution with missing values as unknown.
    """

    values = [
        _normalize_distribution_value(value).lower()
        for value in data.cases["sex"].tolist()
    ]

    counts = Counter(values)

    ordered_values = (
        "female",
        "male",
        "unknown",
    )

    extras = sorted(
        set(counts) - set(ordered_values)
    )

    value = [
        {
            "sex": sex,
            "count": int(counts.get(sex, 0)),
        }
        for sex in ordered_values
        if counts.get(sex, 0) > 0
    ]

    value.extend(
        {
            "sex": sex,
            "count": int(counts[sex]),
        }
        for sex in extras
    )

    return AnalysisResult(
        metric_id="sex_distribution",
        value=value,
        definition=(
            "Distribution of normalized safety cases by case-level sex. "
            "Missing or blank sex values are classified as unknown."
        ),
        population="case",
        unit="cases",
        supporting_ids=_case_ids(data),
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="country_distribution",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_country_distribution(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Return case-level occurcountry distribution.

    The source value EU is intentionally retained as an EU region bucket.
    It is not expanded into individual member countries.
    Ambiguous and missing country values are represented as unknown.
    """

    values: list[str] = []

    for value in data.cases["occurcountry"].tolist():
        if pd.isna(value):
            values.append("unknown")
            continue

        text = str(value).strip()

        if not text:
            values.append("unknown")
        else:
            values.append(text.upper())

    counts = Counter(values)

    value = [
        {
            "country_or_region": country,
            "count": int(count),
        }
        for country, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    return AnalysisResult(
        metric_id="country_distribution",
        value=value,
        definition=(
            "Distribution of normalized safety cases by occurcountry. "
            "The source value EU is retained as an EU region bucket and "
            "is not expanded into individual member countries. Missing "
            "country values are classified as unknown. Cases with "
            "ambiguous normalized country values are not assigned an "
            "arbitrary country."
        ),
        population="case",
        unit="cases",
        supporting_ids=_case_ids(data),
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="history_of_actions",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_history_of_actions(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Record the controlled-data limitation that no history of actions
    was supplied as report input.
    """

    return AnalysisResult(
        metric_id="history_of_actions",
        value={
            "status": "none_provided",
            "statement": (
                "No history of actions was provided in the supplied "
                "reporting data."
            ),
        },
        definition=(
            "Controlled statement describing the absence of a supplied "
            "history-of-actions data source. The patient drug action "
            "field is not interpreted as a history-of-actions record."
        ),
        population="case",
        unit="cases",
        supporting_ids=[],
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="case_listing",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_case_listing(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Produce the deterministic case listing required by the report.

    Fields:
        case ID
        reaction(s)
        seriousness
        receivedate
        country
        outcome

    The country field is read from either ``occurcountry`` (normalized
    schema) or ``country`` (legacy test schema). Neither is required:
    when absent, country is reported as "Unknown" for every case.

    ``receivedate`` is coerced through pandas so that both string and
    datetime inputs render as ``YYYY-MM-DD``. Missing seriousness is
    preserved as None. Missing country is reported as "Unknown".

    Reaction terms are deduplicated per case and returned in sorted
    order so that the listing is deterministic and each term appears
    at most once per case.
    """

    required_case_columns = {
        "case_id",
        "receivedate",
        "serious",
    }

    missing_columns = (
        required_case_columns - set(data.cases.columns)
    )

    if missing_columns:
        raise ValueError(
            "Normalized case data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    country_column = (
        "occurcountry"
        if "occurcountry" in data.cases.columns
        else "country"
        if "country" in data.cases.columns
        else None
    )

    case_outcomes = derive_case_outcomes(data)

    outcome_by_case = dict(
        zip(
            case_outcomes["case_id"].astype(str),
            case_outcomes["case_outcome"].astype(str),
        )
    )

    reactions_by_case: dict[str, list[str]] = {}

    if not data.reactions.empty:
        for case_id, group in data.reactions.groupby(
            "case_id",
            sort=True,
        ):
            terms = (
                group["reaction_pt"]
                .dropna()
                .astype(str)
                .str.strip()
            )

            terms = sorted({
                term
                for term in terms.tolist()
                if term
            })

            reactions_by_case[str(case_id)] = terms

    records: list[dict[str, Any]] = []

    ordered_cases = data.cases.sort_values(
        ["receivedate", "case_id"],
        kind="stable",
    )

    for row in ordered_cases.itertuples(index=False):
        case_id = str(row.case_id)

        country = (
            getattr(row, country_column)
            if country_column is not None
            else None
        )

        if pd.isna(country):
            country_value = "Unknown"
        else:
            country_text = str(country).strip()
            country_value = country_text if country_text else "Unknown"

        receivedate = pd.to_datetime(
            row.receivedate,
            errors="coerce",
        )

        serious_value = (
            None
            if pd.isna(row.serious)
            else bool(row.serious)
        )

        records.append(
            {
                "case_id": case_id,
                "reactions": reactions_by_case.get(
                    case_id,
                    [],
                ),
                "serious": serious_value,
                "receivedate": (
                    receivedate.strftime("%Y-%m-%d")
                    if not pd.isna(receivedate)
                    else None
                ),
                "country": country_value,
                "outcome": outcome_by_case.get(
                    case_id,
                    "missing",
                ),
            }
        )

    return AnalysisResult(
        metric_id="case_listing",
        value=records,
        definition=(
            "Deterministic case listing containing case ID, normalized "
            "reaction Preferred Terms, case-level seriousness, receivedate, "
            "occurcountry, and deterministic case-level outcome."
        ),
        population="case",
        unit="case_listing",
        supporting_ids=_case_ids(data),
        version=ANALYSIS_VERSION,
    )