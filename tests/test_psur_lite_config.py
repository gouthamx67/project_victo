from pathlib import Path

from src.report_config import load_report_config


CONFIG = Path("configs/psur_lite.yaml")


def test_psur_lite_config_loads() -> None:
    config = load_report_config(CONFIG)

    assert config.report.id == "psur-lite"
    assert config.report.version == "1.0.0"


def test_psur_lite_has_expected_sections() -> None:
    config = load_report_config(CONFIG)

    sections = sorted(
        config.report.sections,
        key=lambda section: section.order,
    )

    assert [section.id for section in sections] == [
        "reporting_period",
        "narrative_summary_and_analysis",
        "summary_analysis_of_cases",
        "reaction_adverse_event_analysis",
        "serious_cases_15_day_alerts",
        "trends_and_important_observations",
        "history_of_actions",
        "case_index_listing",
    ]


def test_psur_lite_uses_existing_metrics() -> None:
    config = load_report_config(CONFIG)

    expected = {
        "reporting_period",
        "case_count",
        "serious_case_count",
        "non_serious_case_count",
        "serious_case_percentage",
        "expedited_case_count",
        "expedited_case_percentage",
        "fatal_case_count",
        "age_distribution",
        "sex_distribution",
        "country_distribution",
        "reporter_qualification_distribution",
        "reaction_count",
        "unique_reaction_count",
        "top_reactions",
        "top_serious_reactions",
        "reaction_outcome_distribution",
        "case_outcome_distribution",
        "monthly_case_counts",
        "monthly_case_deviations",
        "monthly_top_reactions",
        "history_of_actions",
        "case_listing",
    }

    actual = {
        metric_id
        for section in config.report.sections
        for metric_id in section.needs
    }

    assert actual == expected


def test_psur_lite_has_both_ai_and_deterministic_sections() -> None:
    config = load_report_config(CONFIG)

    assert any(
        section.ai
        for section in config.report.sections
    )

    assert any(
        not section.ai
        for section in config.report.sections
    )


def test_psur_lite_does_not_claim_full_regulatory_scope() -> None:
    config = load_report_config(CONFIG)

    assert (
        "demonstration configuration"
        in config.report.regulatory_basis.reference.lower()
    )


def test_psur_lite_runs_through_same_pipeline() -> None:
    from src.report_pipeline import run_report_pipeline

    dataset = Path("tests/fixtures/synthetic_icsr.csv")

    prompt = Path(
        "prompts/sections/default.txt"
    )

    result = run_report_pipeline(
        csv_path=dataset,
        config_path=CONFIG,
        prompt_path=prompt,
        use_llm=False,
    )

    assert result.config.report.id == "psur-lite"

    assert len(result.normalized.cases) == 6
    assert len(result.normalized.reactions) == 7

    assert [
        section.section_id
        for section in result.sections
    ] == [
        "reporting_period",
        "narrative_summary_and_analysis",
        "summary_analysis_of_cases",
        "reaction_adverse_event_analysis",
        "serious_cases_15_day_alerts",
        "trends_and_important_observations",
        "history_of_actions",
        "case_index_listing",
    ]

    assert result.sections[-1].ai_enabled is False