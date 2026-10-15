from __future__ import annotations

from src.analysis_framework import AnalysisResult
from src.llm_contract import LLMNarrative, MetricContext, SectionContext
from src.llm_provider import LLMProvider, generate_grounded_narrative


def _case_count_result() -> AnalysisResult:
    return AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of unique safety report cases.",
        population="case",
        supporting_ids=["CASE-001", "CASE-002"],
        version="1.0.0",
    )


def _case_count_context() -> SectionContext:
    return SectionContext(
        report_id="pader",
        report_version="1.0.0",
        section_id="case_volume",
        section_title="Case Volume",
        instructions="Describe case volume.",
        style={"tone": "formal", "length": "concise"},
        metrics=[
            MetricContext(
                metric_id="case_count",
                value=1024,
                definition="Number of unique safety report cases.",
                population="case",
                supporting_ids=["CASE-001", "CASE-002"],
                version="1.0.0",
            )
        ],
    )


class SequenceProvider(LLMProvider):
    """Test provider that returns predetermined responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, context, prompt) -> LLMNarrative:
        self.calls += 1

        if not self.responses:
            raise RuntimeError("No test response remaining.")

        return LLMNarrative(text=self.responses.pop(0))


def test_valid_llm_output_is_accepted() -> None:
    provider = SequenceProvider(
        ["There were {case_count} cases in the reporting period."]
    )

    result = generate_grounded_narrative(
        provider=provider,
        context=_case_count_context(),
        prompt="Write a case-volume narrative.",
        results=[_case_count_result()],
        max_retries=1,
    )

    assert result.text == (
        "There were {case_count} cases in the reporting period."
    )
    assert provider.calls == 1


def test_unsupported_number_causes_retry() -> None:
    provider = SequenceProvider(
        [
            "There were 9999 cases in the reporting period.",
            "There were {case_count} cases in the reporting period.",
        ]
    )

    result = generate_grounded_narrative(
        provider=provider,
        context=_case_count_context(),
        prompt="Write a case-volume narrative.",
        results=[_case_count_result()],
        max_retries=1,
    )

    assert result.text == (
        "There were {case_count} cases in the reporting period."
    )
    assert provider.calls == 2


def test_banned_claim_causes_retry() -> None:
    provider = SequenceProvider(
        [
            "The data show a safety signal involving the product.",
            "The data include {case_count} cases.",
        ]
    )

    result = generate_grounded_narrative(
        provider=provider,
        context=_case_count_context(),
        prompt="Write a case-volume narrative.",
        results=[_case_count_result()],
        max_retries=1,
    )

    assert result.text == "The data include {case_count} cases."
    assert provider.calls == 2


def test_exhausted_retries_use_deterministic_fallback() -> None:
    provider = SequenceProvider(
        [
            "There were 9999 cases.",
            "The product caused all reported events.",
        ]
    )

    result = generate_grounded_narrative(
        provider=provider,
        context=_case_count_context(),
        prompt="Write a case-volume narrative.",
        results=[_case_count_result()],
        max_retries=1,
    )

    # Both LLM responses fail grounding validation, so the helper
    # returns a deterministic fallback containing the approved metric slot.
    assert result.text == "[NO-LLM DRAFT] Case Volume: {case_count}"
    assert provider.calls == 2