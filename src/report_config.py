from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


AI_MODE_VALUES = {"true", "false"}


class StyleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tone: str
    length: str


class RegulatoryBasis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jurisdiction: str
    reference: str


class SectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    order: int
    needs: list[str] = Field(min_length=1)
    ai: bool
    instructions: str
    style: StyleConfig


class ReportDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    version: str
    product_name: str = "Not supplied"
    application_identifier: str = "not supplied"
    report_type: str = "Periodic Safety Update Report"
    regulatory_basis: RegulatoryBasis
    sections: list[SectionConfig] = Field(min_length=1)


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: ReportDefinition


def _validate_section_ids(
    sections: list[SectionConfig],
) -> None:
    section_ids = [
        section.id
        for section in sections
    ]

    duplicates = sorted(
        {
            section_id
            for section_id in section_ids
            if section_ids.count(section_id) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Duplicate report section IDs: "
            + ", ".join(duplicates)
        )


def _validate_section_orders(
    sections: list[SectionConfig],
) -> None:
    orders = [
        section.order
        for section in sections
    ]

    duplicates = sorted(
        {
            order
            for order in orders
            if orders.count(order) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "Duplicate report section orders: "
            + ", ".join(map(str, duplicates))
        )


def _validate_metric_ids(
    sections: list[SectionConfig],
) -> None:
    for section in sections:
        for metric_id in section.needs:
            if not metric_id.strip():
                raise ValueError(
                    f"Section '{section.id}' contains an empty "
                    "metric ID."
                )


def validate_report_config(
    config: ReportConfig,
) -> ReportConfig:
    """
    Apply semantic validation beyond Pydantic schema validation.
    """

    sections = config.report.sections

    _validate_section_ids(sections)
    _validate_section_orders(sections)
    _validate_metric_ids(sections)

    return config


def load_report_config(
    path: str | Path,
) -> ReportConfig:
    """
    Load a YAML report configuration and validate its schema
    and semantic constraints.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Report configuration not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Report configuration path is not a file: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(
            f"Report configuration is empty: {config_path}"
        )

    if not isinstance(raw, dict):
        raise ValueError(
            "Report configuration root must be a YAML mapping."
        )

    config = ReportConfig.model_validate(raw)

    return validate_report_config(config)


def get_section(
    config: ReportConfig,
    section_id: str,
) -> SectionConfig:
    """
    Return one configured report section by ID.
    """

    for section in config.report.sections:
        if section.id == section_id:
            return section

    raise KeyError(
        f"Report section not found: {section_id}"
    )


def ordered_sections(
    config: ReportConfig,
) -> list[SectionConfig]:
    """
    Return sections in deterministic configured order.
    """

    return sorted(
        config.report.sections,
        key=lambda section: (
            section.order,
            section.id,
        ),
    )


def all_required_metrics(
    config: ReportConfig,
) -> list[str]:
    """
    Return unique metric IDs required by the report,
    preserving first appearance order.
    """

    result: list[str] = []
    seen: set[str] = set()

    for section in ordered_sections(config):
        for metric_id in section.needs:
            if metric_id not in seen:
                result.append(metric_id)
                seen.add(metric_id)

    return result