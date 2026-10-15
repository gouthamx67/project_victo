from __future__ import annotations

from collections import Counter

import pandas as pd

from src.analysis_framework import AnalysisResult, register_analysis
from src.normalize import NormalizedData


ANALYSIS_VERSION = "1.0.0"

# A month is descriptively flagged when its case count is greater than
# this multiplier times the mean monthly case count.
DEVIATION_MULTIPLIER = 1.5


def _monthly_case_counts(data: NormalizedData) -> pd.DataFrame:
    """
    Return one row per reporting month with unique case counts.

    The normalized case table is expected to contain a receivedate column
    in YYYY-MM-DD-compatible form.
    """

    if data.cases.empty:
        return pd.DataFrame(
            columns=[
                "reporting_month",
                "case_count",
            ]
        )

    cases = data.cases.copy()

    if "receivedate" not in cases.columns:
        raise ValueError(
            "Normalized case data must contain 'receivedate' "
            "for monthly trend analysis."
        )

    receivedates = pd.to_datetime(
        cases["receivedate"],
        errors="coerce",
    )

    valid = receivedates.notna()

    if not valid.any():
        return pd.DataFrame(
            columns=[
                "reporting_month",
                "case_count",
            ]
        )

    trend = pd.DataFrame(
        {
            "reporting_month": receivedates.loc[valid]
            .dt.to_period("M")
            .astype(str),
            "case_id": cases.loc[valid, "case_id"].astype(str),
        }
    )

    result = (
        trend.groupby("reporting_month")["case_id"]
        .nunique()
        .reset_index(name="case_count")
        .sort_values("reporting_month")
        .reset_index(drop=True)
    )

    return result


def _monthly_reaction_counts(
    data: NormalizedData,
) -> pd.DataFrame:
    """
    Return monthly reaction counts by normalized reaction term.

    Reaction rows are the unit of counting.
    """

    if data.reactions.empty:
        return pd.DataFrame(
            columns=[
                "reporting_month",
                "reaction_pt",
                "reaction_count",
            ]
        )

    reactions = data.reactions.copy()

    required_columns = {
        "reaction_id",
        "reaction_pt",
        "case_id",
    }

    missing_columns = required_columns - set(
        reactions.columns
    )

    if missing_columns:
        raise ValueError(
            "Normalized reaction data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if "receivedate" not in reactions.columns:
        # The normalized reaction table currently does not necessarily
        # carry receivedate, so obtain it from the case table.
        if "receivedate" not in data.cases.columns:
            raise ValueError(
                "Either normalized reactions or normalized cases must "
                "contain 'receivedate' for monthly reaction analysis."
            )

        reactions = reactions.merge(
            data.cases[
                ["case_id", "receivedate"]
            ],
            on="case_id",
            how="left",
            validate="many_to_one",
        )

    receivedates = pd.to_datetime(
        reactions["receivedate"],
        errors="coerce",
    )

    valid = (
        receivedates.notna()
        & reactions["reaction_pt"].notna()
        & reactions["reaction_pt"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    if not valid.any():
        return pd.DataFrame(
            columns=[
                "reporting_month",
                "reaction_pt",
                "reaction_count",
            ]
        )

    trend = pd.DataFrame(
        {
            "reporting_month": receivedates.loc[valid]
            .dt.to_period("M")
            .astype(str),
            "reaction_pt": reactions.loc[
                valid,
                "reaction_pt",
            ]
            .astype(str)
            .str.strip(),
            "reaction_id": reactions.loc[
                valid,
                "reaction_id",
            ].astype(str),
        }
    )

    result = (
        trend.groupby(
            [
                "reporting_month",
                "reaction_pt",
            ]
        )["reaction_id"]
        .nunique()
        .reset_index(name="reaction_count")
        .sort_values(
            [
                "reporting_month",
                "reaction_count",
                "reaction_pt",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    return result


def _full_calendar_months(
    monthly_counts: pd.DataFrame,
) -> list[str]:
    """
    Return months that are complete calendar months within the observed
    reporting period.

    The first and last observed calendar months are excluded when they
    are partial. A one-month dataset therefore has no full-month baseline.
    """

    if monthly_counts.empty:
        return []

    months = (
        monthly_counts["reporting_month"]
        .astype(str)
        .tolist()
    )

    if len(months) <= 2:
        return []

    return months[1:-1]


def _deviation_rows(
    monthly_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the documented descriptive monthly deviation rule.

    Partial first and last calendar months are excluded from the baseline
    calculation. They remain in the returned monthly observations, but
    their deviation_flag is False because they are not eligible for the
    full-month baseline comparison.

    This is a descriptive review method, not statistical signal detection.
    """

    if monthly_counts.empty:
        result = monthly_counts.copy()

        result["baseline_eligible"] = pd.Series(
            dtype="bool"
        )

        result["mean_full_month_case_count"] = pd.Series(
            dtype="float64"
        )

        result["deviation_threshold"] = pd.Series(
            dtype="float64"
        )

        result["deviation_flag"] = pd.Series(
            dtype="bool"
        )

        return result

    full_months = _full_calendar_months(
        monthly_counts
    )

    result = monthly_counts.copy()

    result["baseline_eligible"] = (
        result["reporting_month"]
        .astype(str)
        .isin(full_months)
    )

    eligible_counts = result.loc[
        result["baseline_eligible"],
        "case_count",
    ]

    if eligible_counts.empty:
        mean_count = float("nan")
    else:
        mean_count = float(eligible_counts.mean())

    if pd.notna(mean_count):
        threshold = mean_count * DEVIATION_MULTIPLIER
    else:
        threshold = float("nan")

    result["mean_full_month_case_count"] = mean_count
    result["deviation_threshold"] = threshold

    result["deviation_flag"] = (
        result["baseline_eligible"]
        & result["case_count"].gt(threshold)
    )

    return result


@register_analysis(
    metric_id="monthly_case_counts",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_monthly_case_counts(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Calculate unique safety case counts by receivedate month.
    """

    monthly = _monthly_case_counts(data)

    value = [
        {
            "reporting_month": row.reporting_month,
            "case_count": int(row.case_count),
        }
        for row in monthly.itertuples(index=False)
    ]

    supporting_ids = (
        data.cases["case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="monthly_case_counts",
        value=value,
        definition=(
            "Number of unique normalized safety cases by calendar "
            "month derived from receivedate."
        ),
        population="case",
        unit="monthly_case_counts",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="monthly_top_reactions",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_monthly_top_reactions(
    data: NormalizedData,
) -> AnalysisResult:
    """Calculate the top 10 reaction terms for each reporting month."""

    monthly = _monthly_reaction_counts(data)

    value: list[dict[str, object]] = []

    for month, group in monthly.groupby(
        "reporting_month",
        sort=True,
    ):
        top_group = group.sort_values(
            ["reaction_count", "reaction_pt"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)

        for row in top_group.itertuples(index=False):
            value.append(
                {
                    "reporting_month": month,
                    "reaction_pt": row.reaction_pt,
                    "reaction_count": int(
                        row.reaction_count
                    ),
                }
            )

    supporting_ids = (
        data.reactions["reaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="monthly_top_reactions",
        value=value,
        definition=(
            "Highest-count monthly normalized reaction terms, "
            "returned as a limited top list, ordered by reaction "
            "count descending and reaction term ascending for ties. "
            "Counts use reaction rows and the associated safety case "
            "receivedate. Low-count results should be interpreted "
            "cautiously."
        ),
        population="reaction",
        unit="monthly_reaction_occurrences",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="monthly_case_deviations",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_monthly_case_deviations(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Flag months whose case count is strictly greater than 1.5
    times the mean case count across eligible full months.

    The 1.5x threshold is a descriptive heuristic for human
    review, not a validated safety-signal threshold. A month
    not being flagged does not establish absence of a safety signal.
    """

    monthly = _monthly_case_counts(data)
    result = _deviation_rows(monthly)

    value = [
        {
            "reporting_month": row.reporting_month,
            "case_count": int(row.case_count),
            "baseline_eligible": bool(
                row.baseline_eligible
            ),
            "mean_full_month_case_count": (
                round(
                    float(row.mean_full_month_case_count),
                    4,
                )
                if pd.notna(
                    row.mean_full_month_case_count
                )
                else None
            ),
            "deviation_threshold": (
                round(
                    float(row.deviation_threshold),
                    4,
                )
                if pd.notna(
                    row.deviation_threshold
                )
                else None
            ),
            "deviation_flag": bool(
                row.deviation_flag
            ),
        }
        for row in result.itertuples(index=False)
    ]

    supporting_ids = (
        data.cases["case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="monthly_case_deviations",
        value=value,
        definition=(
            "Descriptive monthly case-count review flag. The first and "
            "last observed calendar months are treated as partial months "
            "and excluded from the full-month baseline when more than one "
            "month is present. Eligible full months are flagged when their "
            "unique case count exceeds the review threshold, a fixed "
            "multiple of the mean case count across eligible full months. "
            "Partial months remain visible but are not flagged. This is "
            "not a validated statistical method."
        ),
        population="case",
        unit="monthly_case_deviation",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


# Backward-compatible import; registration is owned by reporting.py.
from src.analyses.reporting import calculate_case_listing