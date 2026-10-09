from __future__ import annotations

from src.analysis_framework import AnalysisResult, register_analysis
from src.normalize import NormalizedData


ANALYSIS_VERSION = "1.0.0"
TOP_REACTION_LIMIT = 10


@register_analysis(
    metric_id="case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_case_count(data: NormalizedData) -> AnalysisResult:
    """Count unique normalized safety cases."""

    supporting_ids = (
        data.cases["case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="case_count",
        value=len(supporting_ids),
        definition=(
            "Number of unique normalized safety cases, with one "
            "case record per safetyreportid."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="serious_case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_serious_case_count(data: NormalizedData) -> AnalysisResult:
    """Count normalized cases classified as serious."""

    mask = data.cases["serious"].eq(True)

    supporting_ids = (
        data.cases.loc[mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="serious_case_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized safety cases whose case-level "
            "serious classification is True."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="serious_case_percentage",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_serious_case_percentage(
    data: NormalizedData,
) -> AnalysisResult:
    """Calculate serious cases as a percentage of all cases."""

    case_ids = (
        data.cases["case_id"]
        .dropna()
        .astype(str)
    )

    total_cases = len(case_ids)

    serious_mask = data.cases["serious"].eq(True)

    supporting_ids = (
        data.cases.loc[serious_mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if total_cases == 0:
        value = None
    else:
        value = round(
            (len(supporting_ids) / total_cases) * 100,
            4,
        )

    return AnalysisResult(
        metric_id="serious_case_percentage",
        value=value,
        definition=(
            "Percentage of normalized safety cases classified as "
            "serious, calculated as serious cases divided by all "
            "normalized cases multiplied by 100."
        ),
        population="case",
        unit="case_percentage",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="expedited_case_count",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_expedited_case_count(
    data: NormalizedData,
) -> AnalysisResult:
    """Count normalized cases meeting the expedited criterion."""

    mask = data.cases["expedite"].eq(True)

    supporting_ids = (
        data.cases.loc[mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="expedited_case_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized safety cases whose case-level "
            "expedited reporting criterion is True."
        ),
        population="case",
        unit="cases",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="expedited_case_percentage",
    population="case",
    version=ANALYSIS_VERSION,
)
def calculate_expedited_case_percentage(
    data: NormalizedData,
) -> AnalysisResult:
    """Calculate expedited cases as a percentage of all cases."""

    case_ids = (
        data.cases["case_id"]
        .dropna()
        .astype(str)
    )

    total_cases = len(case_ids)

    expedited_mask = data.cases["expedite"].eq(True)

    supporting_ids = (
        data.cases.loc[expedited_mask, "case_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if total_cases == 0:
        value = None
    else:
        value = round(
            (len(supporting_ids) / total_cases) * 100,
            4,
        )

    return AnalysisResult(
        metric_id="expedited_case_percentage",
        value=value,
        definition=(
            "Percentage of normalized safety cases meeting the "
            "expedited reporting criterion, calculated as expedited "
            "cases divided by all normalized cases multiplied by 100."
        ),
        population="case",
        unit="case_percentage",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="reaction_count",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_reaction_count(data: NormalizedData) -> AnalysisResult:
    """Count normalized reaction records."""

    supporting_ids = (
        data.reactions["reaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="reaction_count",
        value=len(supporting_ids),
        definition=(
            "Number of normalized reaction records after expanding "
            "source reaction terms into individual reaction records."
        ),
        population="reaction",
        unit="reaction_occurrences",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="unique_reaction_count",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_unique_reaction_count(
    data: NormalizedData,
) -> AnalysisResult:
    """Count distinct normalized reaction Preferred Terms."""

    valid_reactions = data.reactions.loc[
        data.reactions["reaction_pt"].notna()
        & data.reactions["reaction_pt"].astype(str).str.strip().ne("")
    ]

    unique_terms = sorted(
        set(
            valid_reactions["reaction_pt"]
            .astype(str)
            .str.strip()
            .tolist()
        )
    )

    supporting_ids = (
        valid_reactions["reaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="unique_reaction_count",
        value=len(unique_terms),
        definition=(
            "Number of distinct normalized reaction Preferred Terms. "
            "No SOC-level inference is performed."
        ),
        population="reaction",
        unit="reaction_distinct_terms",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="distinct_case_reaction_pairs",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_distinct_case_reaction_pairs(
    data: NormalizedData,
) -> AnalysisResult:
    """Count distinct case and reaction-term pairs."""

    valid_reactions = data.reactions.loc[
        data.reactions["case_id"].notna()
        & data.reactions["reaction_pt"].notna()
        & data.reactions["reaction_pt"].astype(str).str.strip().ne("")
    ].copy()

    valid_reactions["case_id"] = (
        valid_reactions["case_id"].astype(str).str.strip()
    )

    valid_reactions["reaction_pt"] = (
        valid_reactions["reaction_pt"].astype(str).str.strip()
    )

    pairs = valid_reactions.drop_duplicates(
        subset=["case_id", "reaction_pt"]
    )

    supporting_ids = (
        pairs["reaction_id"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return AnalysisResult(
        metric_id="distinct_case_reaction_pairs",
        value=len(pairs),
        definition=(
            "Number of distinct case and normalized reaction-term "
            "pairs. Repeated occurrences of the same reaction term "
            "within one case count once."
        ),
        population="reaction",
        unit="distinct_case_reaction_pairs",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )


@register_analysis(
    metric_id="top_reactions",
    population="reaction",
    version=ANALYSIS_VERSION,
)
def calculate_top_reactions(data: NormalizedData) -> AnalysisResult:
    """
    Identify the ten most frequently reported normalized reaction
    Preferred Terms.

    Each result reports both:

        occurrences:
            number of normalized reaction occurrences

        distinct_cases:
            number of distinct safety cases containing the PT

    Frequency ties are resolved alphabetically.
    """

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

    grouped = (
        valid_reactions
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
            valid_reactions.loc[
                valid_reactions["reaction_pt"].isin(top_term_names),
                "case_id",
            ]
            .dropna()
            .astype(str)
        )
    )

    return AnalysisResult(
        metric_id="top_reactions",
        value=value,
        definition=(
            "The ten most frequently reported normalized reaction "
            "Preferred Terms. Occurrences count normalized reaction "
            "records, while distinct_cases counts unique safetyreportid "
            "cases containing each Preferred Term. Frequency ties are "
            "resolved alphabetically."
        ),
        population="reaction",
        unit="reaction_occurrences",
        supporting_ids=supporting_ids,
        version=ANALYSIS_VERSION,
    )