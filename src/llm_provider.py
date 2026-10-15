from __future__ import annotations

import hashlib
import json
import os
import re
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


class LLMProviderError(RuntimeError):
    """
    Raised when a provider call fails at the transport or API level.

    This covers authentication failures, rejected requests, connection
    errors, and rate limiting after retries. It deliberately does not
    cover grounding validation, which has its own exception type and
    deterministic fallback.
    """


_ANSI_ESCAPE_PATTERN = re.compile(
    r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-Z\\-_])"
)

_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_api_key(value: str | None) -> str | None:
    """
    Normalize an API key read from the environment.

    Pasted keys frequently carry terminal artifacts: ANSI escape
    sequences (for example ``\\x1b[C`` from an arrow-key press), stray
    control characters, surrounding whitespace, or wrapping quotes.
    Any of these make the key invalid at the API edge while looking
    almost correct, so they are removed here.
    """

    if value is None:
        return None

    cleaned = _ANSI_ESCAPE_PATTERN.sub("", value)
    cleaned = _CONTROL_CHARACTER_PATTERN.sub("", cleaned)
    cleaned = cleaned.strip().strip("'\"")

    return cleaned or None


def read_api_key_from_env(variable: str) -> str | None:
    """Read and sanitize an API key environment variable."""

    return sanitize_api_key(os.getenv(variable))


def _provider_failure_message(
    channel: str,
    exc: Exception,
    *,
    max_length: int = 300,
) -> str:
    """Build a bounded, secret-free provider failure message."""

    detail = str(exc).replace("\n", " ").strip()
    detail = _CONTROL_CHARACTER_PATTERN.sub("", detail)

    if len(detail) > max_length:
        detail = detail[:max_length] + "…"

    return (
        f"{channel} request failed "
        f"({type(exc).__name__}): {detail}"
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
            resolved_api_key = read_api_key_from_env("GROQ_API_KEY")
        else:
            resolved_api_key = read_api_key_from_env("OPENAI_API_KEY")

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

        try:
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
        except Exception as exc:
            raise LLMProviderError(
                _provider_failure_message(
                    "OpenAI Responses API",
                    exc,
                )
            ) from exc

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


class GroqChatProvider(LLMProvider):
    """
    Groq provider using the OpenAI-compatible chat completions API.

    Groq exposes an OpenAI-compatible ``/chat/completions`` endpoint rather
    than OpenAI's ``/responses`` endpoint, so this provider intentionally uses
    ``client.chat.completions.create``. It reuses the OpenAI Python SDK with a
    Groq base URL and requires ``GROQ_API_KEY`` (or an explicit key).

    Like the OpenAI provider, it only ever receives a SectionContext and an
    already-rendered prompt. It never receives the raw dataset.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        config: LLMGenerationConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if config.provider != "groq":
            raise ValueError(
                "GroqChatProvider requires provider 'groq'."
            )

        if config.temperature != 0.0:
            raise ValueError(
                "GroqChatProvider requires temperature=0.0 "
                "for reproducible generation."
            )

        if config.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self.config = config

        if client is not None:
            self._client = client
            return

        resolved_api_key = (
            api_key
            or read_api_key_from_env("GROQ_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "API key is required. Set GROQ_API_KEY "
                "or pass api_key explicitly."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is not installed. "
                "Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=resolved_api_key,
            base_url=self.GROQ_BASE_URL,
        )

    def generate(
        self,
        context: SectionContext,
        prompt: str,
    ) -> LLMNarrative:
        if not prompt.strip():
            raise ValueError(
                "Prompt must not be empty."
            )

        request_options: dict[str, Any] = {}

        # gpt-oss models are reasoning models on Groq: their reasoning
        # tokens count against max_tokens, and at the default effort the
        # budget can be consumed before any content is emitted. Low
        # effort keeps the budget for the section narrative itself.
        if "gpt-oss" in self.config.model.lower():
            request_options["extra_body"] = {
                "reasoning_effort": "low",
            }

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                **request_options,
            )
        except Exception as exc:
            raise LLMProviderError(
                _provider_failure_message(
                    "Groq Chat Completions API",
                    exc,
                )
            ) from exc

        choices = getattr(response, "choices", None)

        if not choices:
            raise LLMProviderError(
                "Groq Chat Completions API returned no choices."
            )

        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None)

        if not isinstance(text, str) or not text.strip():
            raise LLMProviderError(
                "Groq Chat Completions API returned no usable text "
                "(the model may have exhausted its output budget on "
                "reasoning tokens)."
            )

        return LLMNarrative(
            text=text.strip()
        )


def build_provider(
    *,
    provider: str,
    model: str | None = None,
    max_tokens: int = 1000,
    api_key: str | None = None,
) -> LLMProvider | None:
    """
    Build a configured LLM provider from a provider name.

    Returns ``None`` for the deterministic no-LLM mode so callers can pass
    the result straight into ``run_report_pipeline``.
    """
    normalized = (provider or "none").strip().lower()

    if normalized in {"none", ""}:
        return None

    if normalized == "mock":
        return MockLLMProvider()

    default_models = {
        "groq": "openai/gpt-oss-120b",
        "openai": "gpt-5-mini",
    }

    default_model_variables = {
        "groq": "GROQ_MODEL",
        "openai": "OPENAI_MODEL",
    }

    if normalized not in default_models:
        raise ValueError(
            "Unsupported provider: "
            f"{provider!r}. Expected one of "
            "'none', 'mock', 'openai', 'groq'."
        )

    resolved_model = (
        model
        or os.getenv(default_model_variables[normalized])
        or default_models[normalized]
    )

    config = LLMGenerationConfig(
        provider=normalized,
        model=resolved_model,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    if normalized == "groq":
        return GroqChatProvider(config, api_key=api_key)

    return OpenAIResponsesProvider(config, api_key=api_key)


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