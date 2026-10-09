from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.report_config import (
    ReportConfig,
    all_required_metrics,
    get_section,
    load_report_config,
    ordered_sections,
    validate_report_config,
)


CONFIG_PATH = Path("configs/pader.yaml")


def test_pader_config_loads():
    config = load_report_config(CONFIG_PATH)

    assert config.report.id == "pader"
    assert config.report.version == "1.0.0"
    assert config.report.regulatory_basis.jurisdiction == "US"


def test_config_contains_expected_sections():
    config = load_report_config(CONFIG_PATH)

    section_ids = [
        section.id
        for section in config.report.sections
    ]

    assert section_ids == [
        "executive_summary",
        "case_volume",
        "seriousness",
        "expedited_cases",
        "reaction_profile",
        "outcomes",
        "trend_review",
        "case_listing",
    ]


def test_sections_have_deterministic_order():
    config = load_report_config(CONFIG_PATH)

    sections = ordered_sections(config)

    assert [
        section.id
        for section in sections
    ] == [
        "executive_summary",
        "case_volume",
        "seriousness",
        "expedited_cases",
        "reaction_profile",
        "outcomes",
        "trend_review",
        "case_listing",
    ]


def test_ai_flags_are_explicit():
    config = load_report_config(CONFIG_PATH)

    ai_by_section = {
        section.id: section.ai
        for section in config.report.sections
    }

    assert ai_by_section["executive_summary"] is True
    assert ai_by_section["case_listing"] is False


def test_case_listing_is_deterministic_only():
    config = load_report_config(CONFIG_PATH)

    section = get_section(
        config,
        "case_listing",
    )

    assert section.ai is False
    assert section.needs == ["case_listing"]


def test_all_required_metrics_are_unique_and_ordered():
    config = load_report_config(CONFIG_PATH)

    metrics = all_required_metrics(config)

    assert metrics == [
        "case_count",
        "serious_case_count",
        "serious_case_percentage",
        "expedited_case_count",
        "expedited_case_percentage",
        "fatal_case_count",
        "monthly_case_deviations",
        "monthly_case_counts",
        "top_reactions",
        "unique_reaction_count",
        "reaction_outcome_distribution",
        "case_outcome_distribution",
        "not_recovered_case_count",
        "recovered_case_count",
        "case_listing",
    ]


def test_get_section_returns_requested_section():
    config = load_report_config(CONFIG_PATH)

    section = get_section(
        config,
        "reaction_profile",
    )

    assert section.title == "Reaction Profile"
    assert section.ai is True


def test_get_section_rejects_unknown_section():
    config = load_report_config(CONFIG_PATH)

    with pytest.raises(KeyError):
        get_section(
            config,
            "does_not_exist",
        )


def test_duplicate_section_ids_are_rejected():
    raw = {
        "report": {
            "id": "test",
            "title": "Test",
            "version": "1.0.0",
            "regulatory_basis": {
                "jurisdiction": "US",
                "reference": "test",
            },
            "sections": [
                {
                    "id": "same",
                    "title": "A",
                    "order": 1,
                    "needs": ["case_count"],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                },
                {
                    "id": "same",
                    "title": "B",
                    "order": 2,
                    "needs": ["case_count"],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                },
            ],
        }
    }

    config = ReportConfig.model_validate(raw)

    with pytest.raises(
        ValueError,
        match="Duplicate report section IDs",
    ):
        validate_report_config(config)


def test_duplicate_section_orders_are_rejected():
    raw = {
        "report": {
            "id": "test",
            "title": "Test",
            "version": "1.0.0",
            "regulatory_basis": {
                "jurisdiction": "US",
                "reference": "test",
            },
            "sections": [
                {
                    "id": "a",
                    "title": "A",
                    "order": 1,
                    "needs": ["case_count"],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                },
                {
                    "id": "b",
                    "title": "B",
                    "order": 1,
                    "needs": ["case_count"],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                },
            ],
        }
    }

    config = ReportConfig.model_validate(raw)

    with pytest.raises(
        ValueError,
        match="Duplicate report section orders",
    ):
        validate_report_config(config)


def test_empty_metric_id_is_rejected():
    raw = {
        "report": {
            "id": "test",
            "title": "Test",
            "version": "1.0.0",
            "regulatory_basis": {
                "jurisdiction": "US",
                "reference": "test",
            },
            "sections": [
                {
                    "id": "a",
                    "title": "A",
                    "order": 1,
                    "needs": [""],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                }
            ],
        }
    }

    config = ReportConfig.model_validate(raw)

    with pytest.raises(
        ValueError,
        match="empty metric ID",
    ):
        validate_report_config(config)


def test_unknown_section_field_is_rejected():
    raw = {
        "report": {
            "id": "test",
            "title": "Test",
            "version": "1.0.0",
            "regulatory_basis": {
                "jurisdiction": "US",
                "reference": "test",
            },
            "sections": [
                {
                    "id": "a",
                    "title": "A",
                    "order": 1,
                    "needs": ["case_count"],
                    "ai": False,
                    "instructions": "Test",
                    "style": {
                        "tone": "formal",
                        "length": "short",
                    },
                    "unexpected": "should fail",
                }
            ],
        }
    }

    with pytest.raises(ValidationError):
        ReportConfig.model_validate(raw)


def test_empty_config_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        load_report_config(path)


def test_non_mapping_root_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"

    path.write_text(
        yaml.safe_dump(["not", "a", "mapping"]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="root must be a YAML mapping",
    ):
        load_report_config(path)


def test_missing_config_file_is_rejected(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_report_config(path)