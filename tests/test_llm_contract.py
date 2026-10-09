from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm_contract import (
    LLMGenerationConfig,
    LLMNarrative,
    MetricContext,
    SectionContext,
)


def make_metric() -> MetricContext:
    return MetricContext(
        metric_id="case_count",
        value=1024,
        definition="Number of unique safety cases.",
        population="case",
        supporting_ids=[
            "CASE-001",
            "CASE-002",
        ],
        version="1.0.0",
    )


def make_context() -> SectionContext:
    return SectionContext(
        report_id="pader",
        report_version="1.0.0",
        section_id="case_volume",
        section_title="Case Volume",
        instructions="Describe case volume only.",
        style={
            "tone": "formal",
            "length": "concise",
        },
        metrics=[make_metric()],
    )


def test_metric_context_validates():
    metric = make_metric()

    assert metric.metric_id == "case_count"
    assert metric.value == 1024
    assert metric.population == "case"


def test_section_context_validates():
    context = make_context()

    assert context.report_id == "pader"
    assert context.section_id == "case_volume"
    assert len(context.metrics) == 1


def test_llm_narrative_requires_text():
    narrative = LLMNarrative(
        text="Test narrative."
    )

    assert narrative.text == "Test narrative."


def test_llm_narrative_rejects_empty_text():
    with pytest.raises(ValidationError):
        LLMNarrative(text="")


def test_generation_config_defaults_to_deterministic_temperature():
    config = LLMGenerationConfig()

    assert config.temperature == 0.0
    assert config.provider == "mock"
    assert config.model == "mock-model"


def test_extra_metric_fields_are_rejected():
    with pytest.raises(ValidationError):
        MetricContext(
            metric_id="case_count",
            value=10,
            definition="test",
            population="case",
            supporting_ids=[],
            version="1.0.0",
            invented_number=999,
        )


def test_extra_section_fields_are_rejected():
    with pytest.raises(ValidationError):
        SectionContext(
            report_id="pader",
            report_version="1.0.0",
            section_id="test",
            section_title="Test",
            instructions="test",
            style={
                "tone": "formal",
                "length": "short",
            },
            metrics=[],
            raw_csv="secret",
        )