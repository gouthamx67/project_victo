from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from src.analysis_framework import AnalysisResult
from src.metric_slots import (
    MetricSlotError,
    find_metric_slots,
    validate_metric_slots,
)


# Numeric literals intentionally include integers, decimals, and percentages.
# Scientific notation is also handled.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"-?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
    r"%?"
    r"(?![A-Za-z0-9_.-])"
)


# These phrases represent claims the current evidence model cannot establish.
#
# We deliberately keep this list conservative. This is not a general-purpose
# semantic safety classifier. It is a deterministic guard against claims that
# are explicitly outside this project's evidence boundary.
# Interpretations that require evidence beyond descriptive counts.
UNSUPPORTED_INTERPRETATION_PATTERNS: tuple[str, ...] = (
    r"\btypical month-to-month variability\b",
)


BANNED_CLAIM_PATTERNS: tuple[str, ...] = (
    r"\bsafety signal\b",
    r"\bsignal detected\b",
    r"\bcaused by\b",
    r"\bcaused\b",
    r"\bled to\b",
    r"\bresulted in\b",
    r"\bcausally related\b",
    r"\bcausal relationship\b",
    r"\bexpected adverse reaction\b",
    r"\bunexpected adverse reaction\b",
    r"\blabelled\b",
    r"\bunlabelled\b",
    r"\bunlabeled\b",
    r"\bcontraindicated\b",
    r"\beffective\b",
    r"\bineffective\b",
)


@dataclass(frozen=True)
class GroundingIssue:
    """One deterministic grounding-validation failure."""

    code: str
    message: str


@dataclass(frozen=True)
class GroundingValidationResult:
    """Result of validating one generated narrative."""

    valid: bool
    issues: tuple[GroundingIssue, ...]

    @property
    def messages(self) -> list[str]:
        return [issue.message for issue in self.issues]


class GroundingValidationError(ValueError):
    """Raised when generated narrative fails grounding validation."""

    def __init__(
        self,
        message: str,
        issues: tuple[GroundingIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.issues = issues


def _numeric_value(value: Any) -> list[float]:
    """
    Extract numeric values from an approved metric value.

    This is used only for comparison. It does not create new metrics.
    """
    if isinstance(value, bool):
        return []

    if isinstance(value, int | float):
        number = float(value)

        if math.isfinite(number):
            return [number]

        return []

    if isinstance(value, dict):
        numbers: list[float] = []

        for item in value.values():
            numbers.extend(_numeric_value(item))

        return numbers

    if isinstance(value, list):
        numbers: list[float] = []

        for item in value:
            numbers.extend(_numeric_value(item))

        return numbers

    return []


def _normalise_number(number_text: str) -> float:
    return float(number_text.rstrip("%"))


def _approved_numbers(
    results: list[AnalysisResult],
) -> set[float]:
    numbers: set[float] = set()

    for result in results:
        numbers.update(_numeric_value(result.value))

    return numbers


def _find_unsupported_numbers(
    text: str,
    results: list[AnalysisResult],
) -> list[str]:
    """
    Find numeric literals that do not occur in approved metric values.

    Note that this intentionally checks values globally. CP13/CP15 can later
    make this stricter by validating evidence at the section/metric level.
    """
    approved = _approved_numbers(results)
    unsupported: list[str] = []

    for match in NUMBER_PATTERN.finditer(text):
        literal = match.group(0)

        try:
            number = _normalise_number(literal)
        except ValueError:
            unsupported.append(literal)
            continue

        if number not in approved:
            unsupported.append(literal)

    return unsupported


def _find_banned_claims(text: str) -> list[str]:
    lowered = text.lower()

    # This phrasing raises a causal question rather than asserting causation.
    neutral_causal_discussion = re.compile(
        r"\bdiscusses\s+whether\b[^.!?]*\bcaused\b"
    )
    neutral_spans = [
        match.span()
        for match in neutral_causal_discussion.finditer(lowered)
    ]

    matches: list[str] = []

    for pattern in BANNED_CLAIM_PATTERNS:
        for match in re.finditer(pattern, lowered):
            # Do not flag "caused" when it occurs inside the narrowly
            # recognized neutral discussion phrase.
            if (
                match.group(0) == "caused"
                and any(start <= match.start() and match.end() <= end
                        for start, end in neutral_spans)
            ):
                continue

            matches.append(match.group(0))
            break

    return matches


def _find_unsupported_interpretations(text: str) -> list[str]:
    """Find known interpretations that the current evidence model cannot support."""
    matches: list[str] = []

    for pattern in UNSUPPORTED_INTERPRETATION_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matches.append(match.group(0))

    return matches


def validate_grounding(
    text: str,
    results: list[AnalysisResult],
    *,
    required_metric_ids: list[str] | None = None,
) -> GroundingValidationResult:
    """
    Deterministically validate an LLM narrative against approved evidence.

    Validation covers:
      - approved metric slots
      - unsupported numeric literals
      - banned unsupported claims
      - required metric coverage
    """
    issues: list[GroundingIssue] = []

    try:
        validate_metric_slots(text, results)
    except MetricSlotError as exc:
        issues.append(
            GroundingIssue(
                code="invalid_metric_slot",
                message=str(exc),
            )
        )

    unsupported_numbers = _find_unsupported_numbers(
        text,
        results,
    )

    if unsupported_numbers:
        unique_numbers = list(dict.fromkeys(unsupported_numbers))

        issues.append(
            GroundingIssue(
                code="unsupported_number",
                message=(
                    "Narrative contains unsupported numeric values: "
                    + ", ".join(unique_numbers)
                ),
            )
        )

    banned_claims = _find_banned_claims(text)

    if banned_claims:
        unique_claims = list(dict.fromkeys(banned_claims))

        issues.append(
            GroundingIssue(
                code="banned_claim",
                message=(
                    "Narrative contains unsupported claim language: "
                    + ", ".join(unique_claims)
                ),
            )
        )

    unsupported_interpretations = _find_unsupported_interpretations(text)

    if unsupported_interpretations:
        issues.append(
            GroundingIssue(
                code="unsupported_interpretation",
                message=(
                    "Narrative contains interpretation language requiring "
                    "additional evidence: "
                    + ", ".join(dict.fromkeys(unsupported_interpretations))
                ),
            )
        )

    if required_metric_ids:
        supplied_slots = set(find_metric_slots(text))

        missing = [
            metric_id
            for metric_id in required_metric_ids
            if metric_id not in supplied_slots
        ]

        if missing:
            issues.append(
                GroundingIssue(
                    code="missing_required_metric",
                    message=(
                        "Narrative is missing required metric slots: "
                        + ", ".join(missing)
                    ),
                )
            )

    return GroundingValidationResult(
        valid=not issues,
        issues=tuple(issues),
    )


def assert_grounded(
    text: str,
    results: list[AnalysisResult],
    *,
    required_metric_ids: list[str] | None = None,
) -> None:
    """Raise GroundingValidationError when validation fails."""
    result = validate_grounding(
        text,
        results,
        required_metric_ids=required_metric_ids,
    )

    if result.valid:
        return

    message = "Grounding validation failed: " + " | ".join(
        result.messages
    )

    raise GroundingValidationError(
        message,
        issues=result.issues,
    )