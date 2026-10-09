from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.analysis_framework import AnalysisResult
from src.grounding_validator import (
    GroundingValidationError,
    assert_grounded,
)
from src.llm_contract import (
    LLMGenerationConfig,
    LLMNarrative,
    SectionContext,
)


class LLMProvider(ABC):
    """Interface for controlled narrative generation."""

    @abstractmethod
    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        """Generate one narrative response."""
        raise NotImplementedError


class NoLLMProvider(LLMProvider):
    """
    Deterministic fallback provider.

    It deliberately does not imitate an LLM. It produces a clearly marked
    draft containing only approved metric slots.
    """

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        metric_slots = " ".join(
            f"{{{metric.metric_id}}}"
            for metric in context.metrics
        )

        if metric_slots:
            text = (
                f"[NO-LLM DRAFT] "
                f"{context.section_title}: "
                f"{metric_slots}"
            )
        else:
            text = (
                f"[NO-LLM DRAFT] "
                f"{context.section_title}: "
                "No approved metrics are available."
            )

        return LLMNarrative(text=text)


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for tests and local development.

    It emits metric IDs rather than metric values. The actual values remain
    under deterministic application control.
    """

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        metric_ids = ", ".join(
            metric.metric_id
            for metric in context.metrics
        )

        if metric_ids:
            text = (
                f"[MOCK LLM] "
                f"{context.section_title} is based on the approved "
                f"metrics: {metric_ids}."
            )
        else:
            text = (
                f"[MOCK LLM] "
                f"{context.section_title} has no approved metrics."
            )

        return LLMNarrative(text=text)


class OpenAIResponsesProvider(LLMProvider):
    """
    Real OpenAI provider using the Responses API.

    The provider receives only a SectionContext and already-rendered prompt.
    It never receives the raw dataset.

    The OpenAI SDK is imported lazily so the rest of the project remains
    usable in environments where the SDK is not installed.

    Note:
        Newer OpenAI models may reject the `temperature` parameter.
        Therefore temperature remains part of the application's generation
        configuration for reproducibility/audit metadata, but it is not
        sent to the Responses API.
    """

    def __init__(
        self,
        config: LLMGenerationConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if config.provider not in {"openai", "groq"}:
            raise ValueError(
                "OpenAIResponsesProvider requires provider "
                "'openai' or 'groq'."
            )

        if config.temperature != 0.0:
            raise ValueError(
                "OpenAIResponsesProvider requires "
                "temperature=0.0 for reproducible generation."
            )

        if config.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self.config = config

        if client is not None:
            self._client = client
            return

        if api_key is not None:
            resolved_api_key = api_key
        elif config.provider == "groq":
            resolved_api_key = os.getenv("GROQ_API_KEY")
        else:
            resolved_api_key = os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            variable = (
                "GROQ_API_KEY"
                if config.provider == "groq"
                else "OPENAI_API_KEY"
            )
            raise ValueError(
                f"API key is required. Set {variable} "
                "or pass api_key explicitly."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is not installed. "
                "Install it with: pip install openai"
            ) from exc

        client_options = {"api_key": resolved_api_key}
        if config.provider == "groq":
            client_options["base_url"] = "https://api.groq.com/openai/v1"

        self._client = OpenAI(**client_options)

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        if not prompt.strip():
            raise ValueError(
                "Prompt must not be empty."
            )

        response = self._client.responses.create(
            model=self.config.model,
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_output_tokens=self.config.max_tokens,
        )

        text = getattr(
            response,
            "output_text",
            None,
        )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "OpenAI Responses API returned no usable text."
            )

        return LLMNarrative(
            text=text.strip()
        )


def load_prompt(path: str | Path) -> str:
    """
    Load a prompt template from disk.

    Missing prompt files are converted into a clearer application-level
    FileNotFoundError while preserving the standard exception type.
    """
    prompt_path = Path(path)

    try:
        return prompt_path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        ) from exc


def render_prompt(
    template: str,
    context: SectionContext,
) -> str:
    """Render the simple controlled prompt template."""
    context_json = json.dumps(
        context.model_dump(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    replacements = {
        "{{SECTION_TITLE}}": context.section_title,
        "{{INSTRUCTIONS}}": context.instructions,
        "{{STYLE_TONE}}": context.style.get(
            "tone",
            "",
        ),
        "{{STYLE_LENGTH}}": context.style.get(
            "length",
            "",
        ),
        "{{CONTEXT_JSON}}": context_json,
    }

    rendered = template

    for placeholder, value in replacements.items():
        rendered = rendered.replace(
            placeholder,
            value,
        )

    return rendered


def prompt_hash(prompt: str) -> str:
    """Return SHA256 hash of a rendered prompt."""
    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def generation_config_hash(
    config: LLMGenerationConfig,
) -> str:
    """Return stable SHA256 hash of generation configuration."""
    payload = json.dumps(
        config.model_dump(),
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def generate_grounded_narrative(
    provider: LLMProvider,
    context: SectionContext,
    prompt: str,
    results: list[AnalysisResult],
    *,
    required_metric_ids: list[str] | None = None,
    max_retries: int = 1,
    fallback_provider: LLMProvider | None = None,
) -> LLMNarrative:
    """
    Generate and validate a grounded narrative.

    Flow:

        provider.generate()
            ↓
        grounding validation
            ↓
        retry if invalid
            ↓
        deterministic fallback if still invalid
    """
    if max_retries < 0:
        raise ValueError(
            "max_retries must be >= 0."
        )

    if fallback_provider is None:
        fallback_provider = NoLLMProvider()

    attempts = max_retries + 1

    for attempt in range(attempts):
        try:
            narrative = provider.generate(
                context,
                prompt,
            )

            assert_grounded(
                narrative.text,
                results,
                required_metric_ids=required_metric_ids,
            )

            return narrative

        except GroundingValidationError:
            continue

    fallback = fallback_provider.generate(
        context,
        prompt,
    )

    assert_grounded(
        fallback.text,
        results,
        required_metric_ids=None,
    )

    return fallback