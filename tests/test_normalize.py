from pathlib import Path

import pandas as pd

from src.ingest import ingest_and_validate
from src.normalize import (
    AGE_UNIT_TO_YEARS,
    normalize_dataset,
    normalize_reactions,
)


DATA_PATH = Path("tests/fixtures/synthetic_icsr.csv")


def load_validated_dataframe() -> pd.DataFrame:
    """Load the real challenge dataset through the same ingest path used by the CLI."""
    df, validation = ingest_and_validate(DATA_PATH)

    assert validation.is_valid, validation.summary()

    return df


def test_age_unit_mapping():
    assert AGE_UNIT_TO_YEARS["800"] == 10.0
    assert AGE_UNIT_TO_YEARS["year"] == 1.0
    assert AGE_UNIT_TO_YEARS["month"] == 1.0 / 12.0
    assert AGE_UNIT_TO_YEARS["week"] == 1.0 / 52.1775
    assert AGE_UNIT_TO_YEARS["day"] == 1.0 / 365.25


def test_age_conversion():
    from src.normalize import _case_age

    frame = pd.DataFrame(
        {
            "patient_patientonsetage": ["50"],
            "patient_patientonsetageunit": ["year"],
        }
    )

    result = _case_age(frame)

    assert result["onset_age"] == 50.0
    assert result["onset_age_unit"] == "year"
    assert result["age_years"] == 50.0
    assert result["age_status"] == "single"


def test_age_conversion_from_months():
    from src.normalize import _case_age

    frame = pd.DataFrame(
        {
            "patient_patientonsetage": ["12"],
            "patient_patientonsetageunit": ["month"],
        }
    )

    result = _case_age(frame)

    assert result["onset_age"] == 12.0
    assert result["onset_age_unit"] == "month"
    assert result["age_years"] == 1.0
    assert result["age_status"] == "single"


def test_age_conversion_from_decade_code():
    from src.normalize import _case_age

    frame = pd.DataFrame(
        {
            "patient_patientonsetage": ["7"],
            "patient_patientonsetageunit": ["800"],
        }
    )

    result = _case_age(frame)

    assert result["onset_age"] == 7.0
    assert result["onset_age_unit"] == "800"
    assert result["age_years"] == 70.0
    assert result["age_status"] == "single"


def test_missing_age_is_not_invented():
    from src.normalize import _case_age

    frame = pd.DataFrame(
        {
            "patient_patientonsetage": [""],
            "patient_patientonsetageunit": [""],
        }
    )

    result = _case_age(frame)

    assert pd.isna(result["onset_age"])
    assert pd.isna(result["onset_age_unit"])
    assert pd.isna(result["age_years"])
    assert result["age_status"] == "missing"


def test_ambiguous_age_is_not_arbitrarily_selected():
    from src.normalize import _case_age

    frame = pd.DataFrame(
        {
            "patient_patientonsetage": ["50", "60"],
            "patient_patientonsetageunit": ["year", "year"],
        }
    )

    result = _case_age(frame)

    assert pd.isna(result["onset_age"])
    assert pd.isna(result["onset_age_unit"])
    assert pd.isna(result["age_years"])
    assert result["age_status"] == "ambiguous"


def test_seriousness_is_boolean():
    from src.normalize import _case_seriousness_flags

    frame = pd.DataFrame(
        {
            "seriousnessdeath": ["yes"],
            "seriousnesslifethreatening": ["no"],
            "seriousnesshospitalization": ["no"],
            "seriousnessdisabling": ["no"],
            "seriousnesscongenitalanomali": ["no"],
            "seriousnessother": ["no"],
        }
    )

    result = _case_seriousness_flags(frame)

    assert bool(result["serious_death"]) is True
    assert bool(result["serious_life_threatening"]) is False
    assert bool(result["serious_hospitalization"]) is False
    assert bool(result["serious_disabling"]) is False
    assert bool(result["serious_congenital_anomaly"]) is False
    assert bool(result["serious_other"]) is False


def test_seriousness_conflict_is_detected():
    from src.normalize import _case_seriousness_flags

    frame = pd.DataFrame(
        {
            "seriousnessdeath": ["yes", "no"],
            "seriousnesslifethreatening": ["no", "no"],
            "seriousnesshospitalization": ["no", "no"],
            "seriousnessdisabling": ["no", "no"],
            "seriousnesscongenitalanomali": ["no", "no"],
            "seriousnessother": ["no", "no"],
        }
    )

    result = _case_seriousness_flags(frame)

    assert bool(result["serious_death"]) is True
    assert bool(result["serious_death_conflict"]) is True


def test_seriousness_without_usable_value_is_missing():
    from src.normalize import _case_seriousness_flags

    frame = pd.DataFrame(
        {
            "seriousnessdeath": [""],
            "seriousnesslifethreatening": [""],
            "seriousnesshospitalization": [""],
            "seriousnessdisabling": [""],
            "seriousnesscongenitalanomali": [""],
            "seriousnessother": [""],
        }
    )

    result = _case_seriousness_flags(frame)

    assert pd.isna(result["serious_death"])


def test_ambiguous_country_is_not_arbitrarily_selected():
    from src.normalize import _case_country_value

    frame = pd.DataFrame(
        {
            "occurcountry": ["UNITED STATES", "EU"],
        }
    )

    value, status = _case_country_value(frame, "occurcountry")

    assert pd.isna(value)
    assert status == "ambiguous"


def test_single_country_is_preserved():
    from src.normalize import _case_country_value

    frame = pd.DataFrame(
        {
            "occurcountry": ["India", "India"],
        }
    )

    value, status = _case_country_value(frame, "occurcountry")

    assert value == "India"
    assert status == "single"


def test_country_iso_codes_are_normalized():
    from src.normalize import _normalize_country_name

    expected = {
        "IE": "Ireland",
        "RO": "Romania",
        "SA": "Saudi Arabia",
        "HR": "Croatia",
        "UA": "Ukraine",
        "FI": "Finland",
    }

    for code, country_name in expected.items():
        assert _normalize_country_name(code) == country_name
        assert _normalize_country_name(code.lower()) == country_name


def test_eu_is_labelled_as_a_region():
    from src.normalize import _normalize_country_name

    assert _normalize_country_name("EU") == "European Union (region)"


def test_country_conflicts_remain_ambiguous_after_normalization():
    import pandas as pd
    from src.normalize import _case_country_value

    frame = pd.DataFrame({
        "occurcountry": ["IE", "France"],
    })

    value, status = _case_country_value(frame, "occurcountry")

    assert pd.isna(value)
    assert status == "ambiguous"


def test_country_code_and_name_agree_after_normalization():
    import pandas as pd
    from src.normalize import _case_country_value

    frame = pd.DataFrame({
        "occurcountry": ["IE", "Ireland"],
    })

    value, status = _case_country_value(frame, "occurcountry")

    assert value == "Ireland"
    assert status == "single"


def test_reaction_list_is_expanded():
    frame = pd.DataFrame(
        {
            "safetyreportid": ["CASE-1"],
            "patient_reaction_reactionmeddrapt": [
                "Headache, Nausea, Vomiting"
            ],
            "patient_reaction_reactionoutcome": [
                "recovered/resolved, "
                "not recovered/not resolved/ongoing, "
                "fatal"
            ],
        }
    )

    result = normalize_reactions(frame)

    assert len(result) == 3

    assert result.iloc[0]["reaction_pt"] == "Headache"
    assert result.iloc[1]["reaction_pt"] == "Nausea"
    assert result.iloc[2]["reaction_pt"] == "Vomiting"

    assert result.iloc[0]["outcome"] == "recovered/resolved"
    assert (
        result.iloc[1]["outcome"]
        == "not recovered/not resolved/ongoing"
    )
    assert result.iloc[2]["outcome"] == "fatal"

    assert result.iloc[0]["outcome_alignment_status"] == "aligned"
    assert result.iloc[1]["outcome_alignment_status"] == "aligned"
    assert result.iloc[2]["outcome_alignment_status"] == "aligned"


def test_missing_outcome_is_not_invented():
    frame = pd.DataFrame(
        {
            "safetyreportid": ["CASE-1"],
            "patient_reaction_reactionmeddrapt": [
                "Headache, Nausea"
            ],
            "patient_reaction_reactionoutcome": [
                "recovered/resolved"
            ],
        }
    )

    result = normalize_reactions(frame)

    assert len(result) == 2

    assert bool(result.iloc[0]["outcome_present"]) is True
    assert result.iloc[0]["outcome"] == "recovered/resolved"

    assert bool(result.iloc[1]["outcome_present"]) is False
    assert pd.isna(result.iloc[1]["outcome"])

    assert (
        result.iloc[1]["outcome_alignment_status"]
        == "missing_source_outcome"
    )


def test_extra_source_outcome_is_detected():
    frame = pd.DataFrame(
        {
            "safetyreportid": ["CASE-1"],
            "patient_reaction_reactionmeddrapt": [
                "Headache"
            ],
            "patient_reaction_reactionoutcome": [
                "recovered/resolved, fatal"
            ],
        }
    )

    result = normalize_reactions(frame)

    assert len(result) == 1
    assert result.iloc[0]["outcome"] == "recovered/resolved"
    assert (
        result.iloc[0]["outcome_alignment_status"]
        == "extra_source_outcome"
    )


def test_normalize_dataset_produces_cases_and_reactions():
    df = load_validated_dataframe()

    normalized = normalize_dataset(df)

    assert len(normalized.cases) == 6
    assert len(normalized.reactions) == 7

    assert normalized.cases["case_id"].is_unique
    assert normalized.reactions["case_id"].notna().all()


def test_normalize_dataset_has_expected_case_columns():
    df = load_validated_dataframe()

    normalized = normalize_dataset(df)

    expected_columns = {
        "case_id",
        "source_row_count",
        "receivedate",
        "report_month",
        "serious",
        "expedite",
        "sex",
        "onset_age",
        "onset_age_unit",
        "age_years",
        "age_status",
        "age_group",
        "occurcountry",
        "occurcountry_status",
        "reporter_country",
        "reporter_country_status",
        "country_mismatch",
    }

    assert expected_columns.issubset(
        set(normalized.cases.columns)
    )


def test_normalize_dataset_has_expected_reaction_columns():
    df = load_validated_dataframe()

    normalized = normalize_dataset(df)

    expected_columns = {
        "reaction_id",
        "case_id",
        "source_row_index",
        "reaction_position",
        "reaction_pt",
        "outcome",
        "outcome_present",
        "outcome_alignment_status",
    }

    assert expected_columns.issubset(
        set(normalized.reactions.columns)
    )


def test_case_normalization_preserves_multi_row_cases():
    df = load_validated_dataframe()

    normalized = normalize_dataset(df)

    assert normalized.cases["source_row_count"].max() == 2
    assert normalized.cases["source_row_count"].gt(1).sum() == 1


def test_country_mismatch_is_boolean():
    df = load_validated_dataframe()

    normalized = normalize_dataset(df)

    assert normalized.cases["country_mismatch"].dtype == bool


def test_conflicting_countries_are_marked_ambiguous():
    df = load_validated_dataframe()

    first = df.iloc[[0]].copy()
    second = first.copy()

    first.loc[:, "safetyreportid"] = "900099"
    first.loc[:, "occurcountry"] = "US"
    first.loc[:, "receivedate"] = "20241227"

    second.loc[:, "safetyreportid"] = "900099"
    second.loc[:, "occurcountry"] = "GB"
    second.loc[:, "receivedate"] = "20241228"

    raw = pd.concat([first, second], ignore_index=True)
    normalized = normalize_dataset(raw)

    case = normalized.cases.loc[
        normalized.cases["case_id"].eq("900099")
    ].iloc[0]

    assert case["occurcountry_status"] == "ambiguous"
    assert pd.isna(case["occurcountry"])
