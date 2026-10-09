from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm_contract import (
    LLMGenerationConfig,
    SectionContext,
)
from src.llm_provider import (
    OpenAIResponsesProvider,
)


def make_context() -> SectionContext:
    return SectionContext(
        report_id="pader",
        report_version="1.0.0",
        section_id="case_volume",
        section_title="Case Volume",
        instructions="Describe case volume.",
        style={
            "tone": "formal",
            "length": "concise",
        },
        metrics=[],
    )


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            output_text=(
                "Case volume is based only on "
                "the approved evidence."
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def make_config() -> LLMGenerationConfig:
    return LLMGenerationConfig(
        provider="openai",
        model="test-model",
        temperature=0.0,
        max_tokens=200,
    )


def test_openai_provider_uses_responses_api():
    client = FakeClient()

    provider = OpenAIResponsesProvider(
        make_config(),
        client=client,
    )

    result = provider.generate(
        make_context(),
        "Write the section.",
    )

    assert (
        result.text
        == "Case volume is based only on "
        "the approved evidence."
    )

    assert len(
        client.responses.calls
    ) == 1

    call = client.responses.calls[0]

    assert call["model"] == "test-model"
    assert "temperature" not in call
    assert call["max_output_tokens"] == 200
    assert call["input"] == [
        {
            "role": "user",
            "content": "Write the section.",
        }
    ]


def test_openai_provider_rejects_nonzero_temperature():
    config = LLMGenerationConfig(
        provider="openai",
        model="test-model",
        temperature=0.2,
        max_tokens=200,
    )

    with pytest.raises(
        ValueError,
        match="temperature=0.0",
    ):
        OpenAIResponsesProvider(
            config,
            client=FakeClient(),
        )


def test_openai_provider_requires_openai_provider_name():
    config = LLMGenerationConfig(
        provider="mock",
        model="test-model",
        temperature=0.0,
        max_tokens=200,
    )

    with pytest.raises(
        ValueError,
        match="provider='openai'",
    ):
        OpenAIResponsesProvider(
            config,
            client=FakeClient(),
        )


def test_openai_provider_rejects_empty_prompt():
    provider = OpenAIResponsesProvider(
        make_config(),
        client=FakeClient(),
    )

    with pytest.raises(
        ValueError,
        match="Prompt must not be empty",
    ):
        provider.generate(
            make_context(),
            "   ",
        )


def test_openai_provider_rejects_empty_model_output():
    class EmptyResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=""
            )

    class EmptyClient:
        responses = EmptyResponses()

    provider = OpenAIResponsesProvider(
        make_config(),
        client=EmptyClient(),
    )

    with pytest.raises(
        ValueError,
        match="no usable text",
    ):
        provider.generate(
            make_context(),
            "Write the section.",
        )


def test_openai_provider_does_not_receive_raw_dataset():
    client = FakeClient()

    provider = OpenAIResponsesProvider(
        make_config(),
        client=client,
    )

    context = make_context()

    provider.generate(
        context,
        "Approved context only.",
    )

    call = client.responses.calls[0]

    assert "csv" not in str(call).lower()
    assert "safetyreportid" not in str(call).lower()