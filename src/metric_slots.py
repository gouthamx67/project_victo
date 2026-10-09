from __future__ import annotations

import re
from typing import Any

from src.analysis_framework import AnalysisResult


# A valid metric slot contains only letters, numbers, underscore,
# dot, or hyphen inside braces.
SLOT_PATTERN = re.compile(r"\{([A-Za-z0-9_.-]+)\}")

# Any brace-delimited expression that is not a valid metric slot.
BRACE_PATTERN = re.compile(r"\{([^{}]*)\}")


class MetricSlotError(ValueError):
    """Raised when metric slots cannot be resolved safely."""


def build_metric_map(
    results: list[AnalysisResult],
) -> dict[str, AnalysisResult]:
    """
    Build a lookup of approved deterministic metrics.

    Duplicate metric IDs are rejected rather than silently overwritten.
    """
    metric_map: dict[str, AnalysisResult] = {}

    for result in results:
        if result.metric_id in metric_map:
            raise MetricSlotError(
                f"Duplicate metric_id supplied: {result.metric_id!r}"
            )

        metric_map[result.metric_id] = result

    return metric_map


def find_metric_slots(text: str) -> list[str]:
    """
    Return metric slot IDs appearing in narrative text.

    Duplicate occurrences are preserved in order.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return SLOT_PATTERN.findall(text)


def _find_invalid_brace_expressions(text: str) -> list[str]:
    """
    Find brace expressions that are not valid metric slots.

    For example:

        {case_count + 1}
        {some expression}
        {123 + 456}

    are invalid.

    Valid slots such as {case_count} are ignored.
    """
    invalid: list[str] = []

    for match in BRACE_PATTERN.finditer(text):
        full_expression = match.group(0)
        contents = match.group(1)

        if not re.fullmatch(
            r"[A-Za-z0-9_.-]+",
            contents,
        ):
            invalid.append(full_expression)

    return invalid


def validate_metric_slots(
    text: str,
    results: list[AnalysisResult],
) -> None:
    """
    Ensure every brace expression is a valid, approved metric slot.

    Unknown slots and expression-like content are rejected.
    """
    metric_map = build_metric_map(results)

    invalid_expressions = _find_invalid_brace_expressions(text)

    if invalid_expressions:
        raise MetricSlotError(
            "Narrative contains invalid metric expressions: "
            + ", ".join(repr(item) for item in invalid_expressions)
        )

    slots = find_metric_slots(text)

    unknown = [
        slot
        for slot in slots
        if slot not in metric_map
    ]

    if unknown:
        unique_unknown = list(dict.fromkeys(unknown))

        raise MetricSlotError(
            "Narrative contains unknown metric slots: "
            + ", ".join(repr(slot) for slot in unique_unknown)
        )


def _format_metric_value(value: Any) -> str:
    """
    Convert an approved metric value to deterministic display text.

    This function does not calculate or transform the value.
    """
    if isinstance(value, float):
        return f"{value:g}"

    if isinstance(value, dict):
        parts = []

        for key, item in value.items():
            parts.append(
                f"{key}: {_format_metric_value(item)}"
            )

        return "; ".join(parts)

    if isinstance(value, list):
        return ", ".join(
            _format_metric_value(item)
            for item in value
        )

    return str(value)


def resolve_metric_slots(
    text: str,
    results: list[AnalysisResult],
) -> str:
    """
    Replace approved metric slots with deterministic metric values.

    Unknown or invalid metric expressions raise MetricSlotError.
    """
    metric_map = build_metric_map(results)

    validate_metric_slots(
        text,
        results,
    )

    def replace(match: re.Match[str]) -> str:
        metric_id = match.group(1)
        result = metric_map[metric_id]

        return _format_metric_value(result.value)

    return SLOT_PATTERN.sub(
        replace,
        text,
    )