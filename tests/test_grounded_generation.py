from __future__ import annotations

from src.analysis_framework import AnalysisResult
from src.grounding_validator import (
    GroundingValidationError,
)
from src.llm_contract import (
    LLMNarrative,
    SectionContext,
)
from src.llm_provider import (
    LLMProvider,
    NoLLMProvider,
    generate_grounded_narrative,
)


def make_result(
    metric_id: str,
    value,
) -> AnalysisResult:
    return AnalysisResult(
        metric_id=metric_id,
        value=value,
        definition=f"Definition of {metric_id}.",
        population="case",
        supporting_ids=[],
        version="1.0.0",
    )


def make_context() -> SectionContext:
    return SectionContext(
        report_id="pader",
        report_version="1.0.0",
        section_id="test",
        section_title="Test Section",
        instructions="Use approved evidence only.",
        style={
            "tone": "formal",
            "length": "short",
        },
        metrics=[
            {
                "metric_id": "case_count",
                "value": 1024,
                "definition": "Number of unique cases.",
                "population": "case",
                "supporting_ids": [],
                "version": "1.0.0",
            }
        ],
    )


class InvalidThenValidProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        self.calls += 1

        if self.calls == 1:
            return LLMNarrative(
                text="There were 999 cases."
            )

        return LLMNarrative(
            text="There were {case_count} cases."
        )


class AlwaysInvalidProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        self.calls += 1

        return LLMNarrative(
            text="There were 999 cases."
        )


class ValidProvider(LLMProvider):
    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        return LLMNarrative(
            text="There were {case_count} cases."
        )


def test_invalid_generation_is_retried():
    provider = InvalidThenValidProvider()

    result = generate_grounded_narrative(
        provider,
        make_context(),
        "prompt",
        [make_result("case_count", 1024)],
        max_retries=1,
    )

    assert provider.calls == 2
    assert result.text == "There were {case_count} cases."


def test_valid_generation_does_not_retry():
    provider = ValidProvider()

    result = generate_grounded_narrative(
        provider,
        make_context(),
        "prompt",
        [make_result("case_count", 1024)],
        max_retries=1,
    )

    assert result.text == "There were {case_count} cases."


def test_invalid_generation_falls_back_after_retry():
    provider = AlwaysInvalidProvider()

    result = generate_grounded_narrative(
        provider,
        make_context(),
        "prompt",
        [make_result("case_count", 1024)],
        max_retries=1,
        fallback_provider=NoLLMProvider(),
    )

    assert provider.calls == 2
    assert result.text == (
        "[NO-LLM DRAFT] Test Section: {case_count}"
    )


def test_zero_retries_means_one_generation_then_fallback():
    provider = AlwaysInvalidProvider()

    result = generate_grounded_narrative(
        provider,
        make_context(),
        "prompt",
        [make_result("case_count", 1024)],
        max_retries=0,
        fallback_provider=NoLLMProvider(),
    )

    assert provider.calls == 1
    assert result.text == (
        "[NO-LLM DRAFT] Test Section: {case_count}"
    )


def test_fallback_is_itself_grounded():
    provider = AlwaysInvalidProvider()

    result = generate_grounded_narrative(
        provider,
        make_context(),
        "prompt",
        [make_result("case_count", 1024)],
        max_retries=0,
    )

    assert "{case_count}" in result.text
    assert "999" not in result.text