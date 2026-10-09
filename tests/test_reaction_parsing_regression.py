from pathlib import Path

import pandas as pd

from src.normalize import (
    EMBEDDED_COMMA_PTS,
    REACTION_OUTCOME_COLUMN,
    REACTION_PT_COLUMN,
    REACTION_VERSION_COLUMN,
    normalize_reactions,
)


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "data"
    / "Bisoprolol_icsr_sample_1068rows (1).csv"
)

CASE_COLUMN = "safetyreportid"


def _load_raw() -> pd.DataFrame:
    return pd.read_csv(
        CSV_PATH,
        dtype=str,
        keep_default_na=False,
    )


def test_raw_pt_token_count_is_3648():
    raw = _load_raw()

    total = 0

    for value in raw[REACTION_PT_COLUMN]:
        text = str(value).strip()

        if text:
            total += len(
                [
                    token.strip()
                    for token in text.split(",")
                    if token.strip()
                ]
            )

    assert total == 3648


def test_raw_outcome_token_count_is_3642():
    raw = _load_raw()

    total = 0

    for value in raw[REACTION_OUTCOME_COLUMN]:
        text = str(value).strip()

        if text:
            total += len(
                [
                    token.strip()
                    for token in text.split(",")
                    if token.strip()
                ]
            )

    assert total == 3642


def test_raw_version_token_count_is_3642():
    raw = _load_raw()

    total = 0

    for value in raw[REACTION_VERSION_COLUMN]:
        text = str(value).strip()

        if text:
            total += len(
                [
                    token.strip()
                    for token in text.split(",")
                    if token.strip()
                ]
            )

    assert total == 3642


def test_normalized_reaction_count_is_3642():
    reactions = normalize_reactions(_load_raw())

    assert len(reactions) == 3642


def test_every_normalized_reaction_is_aligned():
    reactions = normalize_reactions(_load_raw())

    assert (
        reactions["reaction_alignment_status"]
        .eq("aligned")
        .all()
    )

    assert (
        reactions[REACTION_PT_COLUMN]
        .notna()
        .all()
    )

    assert (
        reactions[REACTION_OUTCOME_COLUMN]
        .notna()
        .all()
    )

    assert (
        reactions[REACTION_VERSION_COLUMN]
        .notna()
        .all()
    )


def test_no_fragment_pts_are_present():
    reactions = normalize_reactions(_load_raw())

    values = set(
        reactions[REACTION_PT_COLUMN]
        .astype(str)
        .str.strip()
    )

    assert "visual" not in values
    assert "auditory" not in values
    assert "mixed" not in values


def test_embedded_comma_pts_are_preserved():
    reactions = normalize_reactions(_load_raw())

    values = set(
        reactions[REACTION_PT_COLUMN]
        .astype(str)
        .str.strip()
    )

    assert EMBEDDED_COMMA_PTS.issubset(values)


def test_exactly_six_source_rows_require_reconstruction():
    raw = _load_raw()

    affected_rows = 0

    for _, row in raw.iterrows():
        pt_tokens = [
            token.strip()
            for token in str(row[REACTION_PT_COLUMN]).split(",")
            if token.strip()
        ]

        outcome_tokens = [
            token.strip()
            for token in str(row[REACTION_OUTCOME_COLUMN]).split(",")
            if token.strip()
        ]

        version_tokens = [
            token.strip()
            for token in str(row[REACTION_VERSION_COLUMN]).split(",")
            if token.strip()
        ]

        if (
            len(pt_tokens) != len(outcome_tokens)
            or len(pt_tokens) != len(version_tokens)
        ):
            affected_rows += 1

    assert affected_rows == 6


def test_exactly_six_embedded_comma_reactions_are_reconstructed():
    reactions = normalize_reactions(_load_raw())

    reconstructed = reactions[
        reactions["pt_reconstruction_status"]
        .eq("reconstructed_embedded_comma")
    ]

    assert len(reconstructed) == 6


def test_reconstructed_pts_are_exactly_the_verified_embedded_comma_pts():
    reactions = normalize_reactions(_load_raw())

    reconstructed = reactions.loc[
        reactions["pt_reconstruction_status"]
        .eq("reconstructed_embedded_comma"),
        REACTION_PT_COLUMN,
    ]

    assert set(reconstructed) == EMBEDDED_COMMA_PTS


def test_non_reconstructed_pts_have_zero_reconstruction_count():
    reactions = normalize_reactions(_load_raw())

    normal = reactions[
        reactions["pt_reconstruction_status"]
        .eq("no_reconstruction")
    ]

    assert (
        normal["pt_reconstruction_count"]
        .eq(0)
        .all()
    )