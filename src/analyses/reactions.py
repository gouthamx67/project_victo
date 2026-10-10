from __future__ import annotations

from collections import Counter

import pandas as pd

from src.analysis_framework import AnalysisResult, register_analysis
from src.normalize import NormalizedData


ANALYSIS_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Outcome categories
# ---------------------------------------------------------------------------

OUTCOME_CATEGORY_FATAL = "fatal"
OUTCOME_CATEGORY_NOT_RECOVERED = "not_recovered"
OUTCOME_CATEGORY_SEQUELAE = "recovered_with_sequelae"
OUTCOME_CATEGORY_RECOVERING = "recovering"
OUTCOME_CATEGORY_RECOVERED = "recovered"
OUTCOME_CATEGORY_UNKNOWN = "unknown"
OUTCOME_CATEGORY_MISSING = "missing"
OUTCOME_CATEGORY_UNMAPPED = "unmapped"


# Explicit source-value mappings verified from the supplied dataset.
OUTCOME_VALUE_TO_CATEGORY = {
    "fatal": OUTCOME_CATEGORY_FATAL,
    "not recovered/not resolved/ongoing": OUTCOME_CATEGORY_NOT_RECOVERED,
    "recovered/resolved with sequelae": OUTCOME_CATEGORY_SEQUELAE,
    "recovering/resolving": OUTCOME_CATEGORY_RECOVERING,
    "recovered/resolved": OUTCOME_CATEGORY_RECOVERED,
    "unknown": OUTCOME_CATEGORY_UNKNOWN,
}


# Higher priority appears first.
CASE_OUTCOME_PRECEDENCE = (
    OUTCOME_CATEGORY_FATAL,
    OUTCOME_CATEGORY_NOT_RECOVERED,
    OUTCOME_CATEGORY_SEQUELAE,
    OUTCOME_CATEGORY_RECOVERING,
    OUTCOME_CATEGORY_RECOVERED,
    OUTCOME_CATEGORY_UNKNOWN,
)


def classify_outcome(value: object) -> str:
    """
    Map one normalized source outcome value to an explicit outcome category.

    Missing values become "missing".

    Values not explicitly recognized become "unmapped".
    They are never silently interpreted as another clinical category.
    """

    if pd.isna(value):
        return OUTCOME_CATEGORY_MISSING

    text = str(value).strip().lower()

    if not text:
        return OUTCOME_CATEGORY_MISSING

    return OUTCOME_VALUE_TO_CATEGORY.get(
        text,
        OUTCOME_CATEGORY_UNMAPPED,
    )


def derive_case_outcome(
    outcome_values: list[object],
) -> str:
    """
    Derive one deterministic case-level outcome category.

    Precedence:

        fatal
        not recovered
        recovered with sequelae
        recovering
        recovered
        unknown

    If recognized outcomes are absent but unmapped outcomes exist,
    return "unmapped".

    If no usable outcome exists, return "missing".
    """

    categories = [
        classify_outcome(value)
        for value in outcome_values
    ]

    category_set = set(categories)

    for category in CASE_OUTCOME_PRECEDENCE:
        if category in category_set:
            return category

    if OUTCOME_CATEGORY_UNMAPPED in category_set:
        return OUTCOME_CATEGORY_UNMAPPED

    return OUTCOME_CATEGORY_MISSING


def derive_case_outcomes(
    data: NormalizedData,
) -> pd.DataFrame:
    """
    Derive exactly one outcome classification per normalized safety case.

    Output columns:

        case_id
        case_outcome
        outcome_supporting_reaction_ids

    Supporting reaction IDs are retained so later evidence tracing can
    identify which normalized reaction records caused the case-level
    classification.
    """

    cached = getattr(data, "_derived_case_outcomes", None)

    if cached is not None:
        return cached

    records: list[dict[str, object]] = []

    reactions = data.reactions.copy()

    if reactions.empty:
        result = pd.DataFrame(
            columns=[
                "case_id",
                "case_outcome",
                "outcome_supporting_reaction_ids",
            ]
        )
        data._derived_case_outcomes = result
        return result

    for case_id, group in reactions.groupby(
        "case_id",
        sort=True,
        dropna=False,
    ):
        case_id_text = str(case_id)

        outcome_values = group["outcome"].tolist()

        case_outcome = derive_case_outcome(
            outcome_values
        )

        supporting_reaction_ids = (
            group.loc[
                group["outcome"].notna()
                & group["outcome"].astype(str).str.strip().ne(""),
                "reaction_id",
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

        records.append(
            {
                "case_id": case_id_text,
                "case_outcome": case_outcome,
                "outcome_supporting_reaction_ids": (
                    supporting_reaction_ids
                ),
            }
        )

    result = pd.DataFrame(records)

    data._derived_case_outcomes = result

    return result


@register_analysis(
    metric_id="reaction_outcome_distribution",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_reaction_outcome_distribution(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Count normalized reaction outcomes by explicit outcome category.

    Missing and unmapped outcomes are retained as separate categories.
    """

    categories: list[str] = []

    for value in data.reactions["outcome"].tolist():
        categories.append(
            classify_outcome(value)
        )

    counts = Counter(categories)

    ordered_categories = (
        OUTCOME_CATEGORY_FATAL,
        OUTCOME_CATEGORY_NOT_RECOVERED,
        OUTCOME_CATEGORY_SEQUELAE,
        OUTCOME_CATEGORY_RECOVERING,
        OUTCOME_CATEGORY_RECOVERED,
        OUTCOME_CATEGORY_UNKNOWN,
        OUTCOME_CATEGORY_MISSING,
        OUTCOME_CATEGORY_UNMAPPED,
    )

    value = [
        {
            "outcome_category": category,
            "count": int(counts.get(category, 0)),
        }
        for category in ordered_categories
        if counts.get(category, 0) > 0
    ]

    supporting_ids = (
        data.reactions["reaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="reaction_outcome_distribution",
        value=value,
        definition=(
            "Distribution of normalized reaction outcomes after mapping "
            "only explicitly supported source outcome values to "
            "deterministic categories. Missing and unmapped values are "
            "retained rather than inferred."
        ),
        population="reaction",
        unit="reaction_outcome_distribution",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="case_outcome_distribution",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_case_outcome_distribution(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Calculate the distribution of derived case-level outcomes.

    One outcome is assigned per normalized case.
    """

    case_outcomes = derive_case_outcomes(data)

    counts = Counter(
        case_outcomes["case_outcome"].tolist()
    )

    ordered_categories = (
        OUTCOME_CATEGORY_FATAL,
        OUTCOME_CATEGORY_NOT_RECOVERED,
        OUTCOME_CATEGORY_SEQUELAE,
        OUTCOME_CATEGORY_RECOVERING,
        OUTCOME_CATEGORY_RECOVERED,
        OUTCOME_CATEGORY_UNKNOWN,
        OUTCOME_CATEGORY_MISSING,
        OUTCOME_CATEGORY_UNMAPPED,
    )

    value = [
        {
            "case_outcome": category,
            "count": int(counts.get(category, 0)),
        }
        for category in ordered_categories
        if counts.get(category, 0) > 0
    ]

    supporting_ids = (
        case_outcomes["case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="case_outcome_distribution",
        value=value,
        definition=(
            "Distribution of one deterministic derived outcome "
            "classification per normalized safety case using the "
            "documented outcome precedence rule."
        ),
        population="case",
        unit="case_outcome_distribution",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="fatal_case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_fatal_case_count(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Count cases whose derived case-level outcome is fatal.
    """

    case_outcomes = derive_case_outcomes(data)

    mask = (
        case_outcomes["case_outcome"]
        == OUTCOME_CATEGORY_FATAL
    )

    supporting_ids = (
        case_outcomes.loc[mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="fatal_case_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized safety cases whose derived "
            "case-level outcome is fatal under the documented "
            "outcome precedence rule."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="not_recovered_case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_not_recovered_case_count(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Count cases whose derived outcome is not recovered/not resolved/ongoing.
    """

    case_outcomes = derive_case_outcomes(data)

    mask = (
        case_outcomes["case_outcome"]
        == OUTCOME_CATEGORY_NOT_RECOVERED
    )

    supporting_ids = (
        case_outcomes.loc[mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="not_recovered_case_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized safety cases whose derived "
            "case-level outcome is not recovered/not resolved/ongoing."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="recovered_case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_recovered_case_count(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Count cases whose derived outcome is recovered/resolved.
    """

    case_outcomes = derive_case_outcomes(data)

    mask = (
        case_outcomes["case_outcome"]
        == OUTCOME_CATEGORY_RECOVERED
    )

    supporting_ids = (
        case_outcomes.loc[mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="recovered_case_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized safety cases whose derived "
            "case-level outcome is recovered/resolved."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


# ---------------------------------------------------------------------------
# Reaction frequency analyses
# ---------------------------------------------------------------------------

TOP_REACTION_LIMIT = 10


@register_analysis(
    metric_id="top_serious_reactions",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_top_serious_reactions(
    data: NormalizedData,
) -> AnalysisResult:
    """
    Identify the ten most frequently reported reaction Preferred Terms
    among cases whose normalized case-level seriousness flag is True.

    The counting unit remains reaction occurrences. Distinct cases are
    reported separately for each Preferred Term.
    """

    serious_cases = data.cases.loc[
        data.cases["serious"].eq(True),
        ["case_id"],
    ].copy()

    serious_cases["case_id"] = (
        serious_cases["case_id"]
        .astype(str)
        .str.strip()
    )

    valid_reactions = data.reactions.loc[
        data.reactions["reaction_pt"].notna()
        & data.reactions["reaction_pt"].astype(str).str.strip().ne("")
    ].copy()

    valid_reactions["reaction_pt"] = (
        valid_reactions["reaction_pt"]
        .astype(str)
        .str.strip()
    )

    valid_reactions["case_id"] = (
        valid_reactions["case_id"]
        .astype(str)
        .str.strip()
    )

    serious_reactions = valid_reactions.merge(
        serious_cases,
        on="case_id",
        how="inner",
        validate="many_to_one",
    )

    grouped = (
        serious_reactions
        .groupby("reaction_pt", sort=False)
        .agg(
            occurrences=("reaction_pt", "size"),
            distinct_cases=("case_id", "nunique"),
        )
        .reset_index()
    )

    ranked = grouped.sort_values(
        by=["occurrences", "reaction_pt"],
        ascending=[False, True],
        kind="stable",
    )

    top_terms = ranked.head(TOP_REACTION_LIMIT)

    value = [
        {
            "reaction_pt": str(row["reaction_pt"]),
            "occurrences": int(row["occurrences"]),
            "distinct_cases": int(row["distinct_cases"]),
        }
        for _, row in top_terms.iterrows()
    ]

    top_term_names = set(
        top_terms["reaction_pt"].astype(str)
    )

    supporting_ids = sorted(
        set(
            serious_reactions.loc[
                serious_reactions["reaction_pt"].isin(top_term_names),
                "case_id",
            ]
            .dropna()
            .astype(str)
        )
    )

    return AnalysisResult(
        metric_id="top_serious_reactions",
        value=value,
        definition=(
            "The ten most frequently reported normalized reaction "
            "Preferred Terms among reactions belonging to cases whose "
            "case-level seriousness classification is True. Occurrences "
            "count reaction records, while distinct_cases counts unique "
            "serious safety cases containing each Preferred Term. "
            "Frequency ties are resolved alphabetically."
        ),
        population="reaction",
        unit="reaction_occurrences",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )