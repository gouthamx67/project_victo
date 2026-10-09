from __future__ import annotations

from pathlib import Path

from src.llm_contract import (
    LLMGenerationConfig,
    SectionContext,
)
from src.llm_provider import (
    MockLLMProvider,
    NoLLMProvider,
    generation_config_hash,
    load_prompt,
    prompt_hash,
    render_prompt,
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


def test_no_llm_provider_is_deterministic():
    context = make_context()
    provider = NoLLMProvider()

    first = provider.generate(
        context,
        "prompt",
    )

    second = provider.generate(
        context,
        "prompt",
    )

    assert first.text == second.text
    assert "[NO-LLM DRAFT]" in first.text


def test_mock_provider_is_deterministic():
    context = make_context()
    provider = MockLLMProvider()

    first = provider.generate(
        context,
        "prompt",
    )

    second = provider.generate(
        context,
        "prompt",
    )

    assert first.text == second.text
    assert "[MOCK LLM]" in first.text


def test_render_prompt_replaces_contract_variables():
    context = make_context()

    template = (
        "TITLE={{SECTION_TITLE}}\n"
        "INSTRUCTIONS={{INSTRUCTIONS}}\n"
        "TONE={{STYLE_TONE}}\n"
        "LENGTH={{STYLE_LENGTH}}\n"
        "DATA={{CONTEXT_JSON}}"
    )

    prompt = render_prompt(
        template,
        context,
    )

    assert "TITLE=Case Volume" in prompt
    assert "INSTRUCTIONS=Describe case volume." in prompt
    assert "TONE=formal" in prompt
    assert "LENGTH=concise" in prompt
    assert '"section_id": "case_volume"' in prompt


def test_prompt_hash_is_stable():
    prompt = "same prompt"

    assert prompt_hash(prompt) == prompt_hash(prompt)


def test_prompt_hash_changes_when_prompt_changes():
    assert prompt_hash("prompt A") != prompt_hash(
        "prompt B"
    )


def test_generation_config_hash_is_stable():
    config = LLMGenerationConfig(
        provider="mock",
        model="test-model",
        temperature=0.0,
        max_tokens=100,
    )

    assert (
        generation_config_hash(config)
        == generation_config_hash(config)
    )


def test_generation_config_hash_changes_when_model_changes():
    config_a = LLMGenerationConfig(
        provider="mock",
        model="model-a",
        temperature=0.0,
        max_tokens=100,
    )

    config_b = LLMGenerationConfig(
        provider="mock",
        model="model-b",
        temperature=0.0,
        max_tokens=100,
    )

    assert (
        generation_config_hash(config_a)
        != generation_config_hash(config_b)
    )


def test_load_prompt_reads_file(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"

    prompt_path.write_text(
        "Hello prompt.",
        encoding="utf-8",
    )

    assert load_prompt(prompt_path) == "Hello prompt."


def test_load_prompt_rejects_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.txt"

    try:
        load_prompt(missing)
    except FileNotFoundError as exc:
        assert "Prompt file not found" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )