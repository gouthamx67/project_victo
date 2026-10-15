from __future__ import annotations

import pandas as pd
import pytest

from src.normalize import (
    EMBEDDED_COMMA_PTS,
    REACTION_OUTCOME_COLUMN,
    REACTION_PT_COLUMN,
    REACTION_VERSION_COLUMN,
    normalize_reactions,
)


def make_raw(rows, *, include_version=True) -> pd.DataFrame:
    """Build small synthetic reaction rows without external datasets."""
    data = {
        "safetyreportid": [row[0] for row in rows],
        REACTION_PT_COLUMN: [row[1] for row in rows],
        REACTION_OUTCOME_COLUMN: [row[2] for row in rows],
    }
    if include_version:
        data[REACTION_VERSION_COLUMN] = [row[3] for row in rows]
    return pd.DataFrame(data)


def test_single_reaction_is_aligned():
    result = normalize_reactions(
        make_raw([("900001", "Headache", "1", "27.0")])
    )

    assert len(result) == 1
    assert result.iloc[0]["case_id"] == "900001"
    assert result.iloc[0]["reaction_pt"] == "Headache"
    assert result.iloc[0]["outcome"] == "1"
    assert result.iloc[0]["meddra_version"] == "27.0"
    assert result.iloc[0]["outcome_alignment_status"] == "aligned"


def test_all_verified_embedded_comma_terms_are_reconstructed():
    rows = [
        ("900001", term, "1", "27.0")
        for term in sorted(EMBEDDED_COMMA_PTS)
    ]

    result = normalize_reactions(make_raw(rows))

    assert set(result["reaction_pt"]) == EMBEDDED_COMMA_PTS
    assert len(result) == len(EMBEDDED_COMMA_PTS)
    assert result["pt_reconstruction_status"].eq(
        "reconstructed_embedded_comma"
    ).all()


def test_fragment_words_alone_are_not_reconstructed():
    terms = ["visual", "auditory", "mixed"]
    rows = [
        ("900001", term, "1", "27.0")
        for term in terms
    ]

    result = normalize_reactions(make_raw(rows))

    assert set(result["reaction_pt"]) == set(terms)
    assert result["pt_reconstruction_status"].eq("no_reconstruction").all()


def test_comma_delimited_fields_align_positionally():
    result = normalize_reactions(
        make_raw([
            ("900001", "Headache, Nausea", "1, 2", "27.0, 27.0")
        ])
    )

    assert result["reaction_pt"].tolist() == ["Headache", "Nausea"]
    assert result["outcome"].tolist() == ["1", "2"]
    assert result["meddra_version"].tolist() == ["27.0", "27.0"]
    assert result["outcome_alignment_status"].eq("aligned").all()


def test_missing_outcomes_are_not_invented():
    result = normalize_reactions(
        make_raw([
            ("900001", "Headache, Nausea", "1", "27.0, 27.0")
        ])
    )

    assert len(result) == 2
    assert result.iloc[0]["outcome"] == "1"
    assert pd.isna(result.iloc[1]["outcome"])
    assert result.iloc[1]["outcome_present"] == False
    assert (
        result.iloc[1]["outcome_alignment_status"]
        == "missing_source_outcome"
    )


def test_extra_outcomes_are_flagged():
    result = normalize_reactions(
        make_raw([("900001", "Headache", "1, 2", "27.0")])
    )

    assert len(result) == 1
    assert (
        result.iloc[0]["outcome_alignment_status"]
        == "extra_source_outcome"
    )


def test_missing_meddra_version_is_not_invented():
    result = normalize_reactions(
        make_raw(
            [("900001", "Headache", "1", "")],
            include_version=False,
        )
    )

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["meddra_version"])
    assert not result.iloc[0]["meddra_version_present"]


def test_blank_tokens_do_not_create_phantom_reactions():
    result = normalize_reactions(
        make_raw([
            ("900001", "Headache, , Nausea", "1,,2", "27.0,,27.0")
        ])
    )

    assert result["reaction_pt"].tolist() == ["Headache", "Nausea"]
    assert result["reaction_pt"].astype(str).str.strip().ne("").all()
    assert result["outcome"].tolist() == ["1", "2"]


def test_reaction_ids_are_unique_across_rows_for_same_case():
    result = normalize_reactions(
        make_raw([
            ("900001", "Headache", "1", "27.0"),
            ("900001", "Nausea", "2", "27.0"),
        ])
    )

    assert len(result) == 2
    assert result["case_id"].eq("900001").all()
    assert result["reaction_id"].is_unique
    assert result["source_row_index"].tolist() == [0, 1]


def test_missing_case_identifier_is_rejected():
    with pytest.raises(ValueError, match="Missing safetyreportid"):
        normalize_reactions(
            make_raw([(None, "Headache", "1", "27.0")])
        )


def test_missing_required_reaction_column_is_rejected():
    frame = pd.DataFrame({
        "safetyreportid": ["900001"],
        REACTION_PT_COLUMN: ["Headache"],
    })

    with pytest.raises(
        ValueError,
        match="Missing required reaction columns",
    ):
        normalize_reactions(frame)
