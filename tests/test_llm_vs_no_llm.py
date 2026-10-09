from __future__ import annotations

from src.analysis_framework import AnalysisResult
from src.grounding_validator import validate_grounding
from src.llm_contract import LLMNarrative, SectionContext
from src.llm_provider import (
    LLMProvider,
    NoLLMProvider,
    generate_grounded_narrative,
)


def _sample_results() -> list[AnalysisResult]:
    return [
        AnalysisResult(
            metric_id="case_count",
            value=1024,
            definition="Number of unique safety report cases.",
            population="case",
            supporting_ids=[f"CASE-{index:04d}" for index in range(1, 1025)],
            version="1.0.0",
        ),
        AnalysisResult(
            metric_id="serious_case_percentage",
            value=99.9,
            definition="Percentage of unique cases classified as serious.",
            population="case",
            supporting_ids=[f"CASE-{index:04d}" for index in range(1, 1024)],
            version="1.0.0",
        ),
    ]


def _sample_context() -> SectionContext:
    return SectionContext(
        report_id="pader",
        report_version="1.0.0",
        section_id="executive_summary",
        section_title="Executive Summary",
        instructions="Summarize only the supplied approved metrics.",
        style={
            "tone": "formal",
            "length": "short",
        },
        metrics=[
            {
                "metric_id": "case_count",
                "value": 1024,
                "definition": "Number of unique safety report cases.",
                "population": "case",
                "supporting_ids": ["CASE-0001"],
                "version": "1.0.0",
            },
            {
                "metric_id": "serious_case_percentage",
                "value": 99.9,
                "definition": "Percentage of unique cases classified as serious.",
                "population": "case",
                "supporting_ids": ["CASE-0001"],
                "version": "1.0.0",
            },
        ],
    )


class FixedNarrativeProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        self.calls += 1
        return LLMNarrative(text=self.text)


def test_no_llm_provider_is_deterministically_grounded():
    provider = NoLLMProvider()
    results = _sample_results()
    context = _sample_context()

    narrative = generate_grounded_narrative(
        provider=provider,
        context=context,
        prompt="Produce a grounded summary.",
        results=results,
        required_metric_ids=[
            "case_count",
            "serious_case_percentage",
        ],
        max_retries=0,
    )

    validation = validate_grounding(
        narrative.text,
        results,
        required_metric_ids=[
            "case_count",
            "serious_case_percentage",
        ],
    )

    assert validation.valid
    assert "{case_count}" in narrative.text
    assert "{serious_case_percentage}" in narrative.text


def test_valid_llm_output_passes_grounding():
    provider = FixedNarrativeProvider(
        "The dataset contains {case_count} cases, with "
        "{serious_case_percentage}% classified as serious."
    )

    results = _sample_results()
    context = _sample_context()

    narrative = generate_grounded_narrative(
        provider=provider,
        context=context,
        prompt="Produce a grounded summary.",
        results=results,
        required_metric_ids=[
            "case_count",
            "serious_case_percentage",
        ],
        max_retries=0,
    )

    assert provider.calls == 1
    assert narrative.text == (
        "The dataset contains {case_count} cases, with "
        "{serious_case_percentage}% classified as serious."
    )


def test_unsupported_llm_number_is_rejected():
    provider = FixedNarrativeProvider(
        "The dataset contains {case_count} cases and 9999 "
        "additional cases were identified."
    )

    results = _sample_results()

    validation = validate_grounding(
        provider.text,
        results,
        required_metric_ids=["case_count"],
    )

    assert not validation.valid
    assert any(
        issue.code == "unsupported_number"
        for issue in validation.issues
    )


def test_causal_or_signal_claim_is_rejected():
    provider = FixedNarrativeProvider(
        "The data indicates a safety signal caused by bisoprolol."
    )

    results = _sample_results()

    validation = validate_grounding(
        provider.text,
        results,
    )

    assert not validation.valid

    issue_codes = {issue.code for issue in validation.issues}

    assert "banned_claim" in issue_codes


def test_invalid_llm_output_falls_back_to_deterministic_narrative():
    provider = FixedNarrativeProvider(
        "The data indicates a safety signal caused by bisoprolol "
        "and includes 9999 cases."
    )

    results = _sample_results()
    context = _sample_context()

    narrative = generate_grounded_narrative(
        provider=provider,
        context=context,
        prompt="Produce a grounded summary.",
        results=results,
        required_metric_ids=[
            "case_count",
            "serious_case_percentage",
        ],
        max_retries=0,
        fallback_provider=NoLLMProvider(),
    )

    validation = validate_grounding(
        narrative.text,
        results,
        required_metric_ids=[
            "case_count",
            "serious_case_percentage",
        ],
    )

    assert validation.valid

    assert "{case_count}" in narrative.text
    assert "{serious_case_percentage}" in narrative.text

    assert "9999" not in narrative.text
    assert "safety signal" not in narrative.text.lower()


def test_invalid_llm_output_is_retried_before_fallback():
    provider = FixedNarrativeProvider(
        "The data indicates a safety signal."
    )

    results = _sample_results()
    context = _sample_context()

    narrative = generate_grounded_narrative(
        provider=provider,
        context=context,
        prompt="Produce a grounded summary.",
        results=results,
        required_metric_ids=["case_count"],
        max_retries=2,
        fallback_provider=NoLLMProvider(),
    )

    assert provider.calls == 3

    validation = validate_grounding(
        narrative.text,
        results,
        required_metric_ids=["case_count"],
    )

    assert validation.valid
    assert "{case_count}" in narrative.text
    assert "safety signal" not in narrative.text.lower()