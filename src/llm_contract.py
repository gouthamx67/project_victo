from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricContext(BaseModel):
    """
    Approved deterministic metric supplied to the LLM.
    """

    model_config = ConfigDict(extra="forbid")

    metric_id: str
    value: Any
    definition: str
    population: str
    supporting_ids: list[str]
    version: str


class SectionContext(BaseModel):
    """
    Minimal evidence packet supplied to one report section.
    """

    model_config = ConfigDict(extra="forbid")

    report_id: str
    report_version: str
    section_id: str
    section_title: str
    instructions: str
    style: dict[str, str]
    metrics: list[MetricContext]


class LLMNarrative(BaseModel):
    """
    Raw narrative response returned by the LLM provider.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class LLMGenerationConfig(BaseModel):
    """
    Reproducibility and provider configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "mock"
    model: str = "mock-model"
    temperature: float = 0.0
    max_tokens: int = 1000