from pathlib import Path

import pandas as pd

import src.analyses.core
import src.analyses.reactions

from src.analysis_framework import run_analysis
from src.normalize import normalize_dataset


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "data"
    / "Bisoprolol_icsr_sample_1068rows (1).csv"
)


def _load_normalized():
    raw = pd.read_csv(
        CSV_PATH,
        dtype=str,
        keep_default_na=False,
    )

    return normalize_dataset(raw)


def test_reaction_count_analysis_uses_corrected_parser():
    data = _load_normalized()

    result = run_analysis(
        "reaction_count",
        data,
    )

    assert result.metric_id == "reaction_count"
    assert result.population == "reaction"
    assert result.value == 3642


def test_unique_reaction_count_matches_normalized_data():
    data = _load_normalized()

    result = run_analysis(
        "unique_reaction_count",
        data,
    )

    expected = data.reactions[
        "patient_reaction_reactionmeddrapt"
    ].nunique()

    assert result.value == expected


def test_top_reactions_uses_corrected_reactions():
    data = _load_normalized()

    result = run_analysis(
        "top_reactions",
        data,
    )

    assert result.metric_id == "top_reactions"
    assert result.population == "reaction"
    assert isinstance(result.value, list)
    assert result.value

    first = result.value[0]

    assert first["reaction_pt"] == "Acute kidney injury"
    assert first["occurrences"] == 81
    assert first["distinct_cases"] == 80


def test_top_reactions_supporting_ids_are_distinct_cases():
    data = _load_normalized()

    result = run_analysis(
        "top_reactions",
        data,
    )

    aki = data.reactions[
        data.reactions["reaction_pt"].eq(
            "Acute kidney injury"
        )
    ]

    expected_aki_case_ids = set(
        aki["case_id"].astype(str)
    )

    assert len(expected_aki_case_ids) == 80

    # Supporting IDs are case IDs for all top reactions.
    assert expected_aki_case_ids.issubset(
        set(result.supporting_ids)
    )

    assert all(
        isinstance(case_id, str)
        for case_id in result.supporting_ids
    )


def test_reaction_outcome_distribution_covers_all_reactions():
    data = _load_normalized()

    result = run_analysis(
        "reaction_outcome_distribution",
        data,
    )

    assert result.metric_id == "reaction_outcome_distribution"
    assert result.population == "reaction"

    total = sum(
        item["count"]
        for item in result.value
    )

    assert total == 3642


def test_reaction_outcome_supporting_ids_cover_all_reactions():
    data = _load_normalized()

    result = run_analysis(
        "reaction_outcome_distribution",
        data,
    )

    assert len(result.supporting_ids) == 3642

    assert set(result.supporting_ids) == set(
        data.reactions["reaction_id"].astype(str)
    )