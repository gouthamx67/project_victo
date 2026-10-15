from __future__ import annotations

import re
from dataclasses import dataclass

from src.analysis_framework import AnalysisResult
from src.metric_slots import (
    MetricSlotError,
    SLOT_PATTERN,
    find_metric_slots,
    validate_metric_slots,
)



_CARDINAL_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|trillion)"
)

_QUANTITY_MODIFIER = (
    r"(?:serious|fatal|suspected|confirmed|reported|observed|"
    r"additional|new|unique|distinct|overall|adverse|total)"
)

_QUANTITY_NOUN = (
    r"(?:cases?|reactions?|reports?|patients?|events?|records?|rows?|"
    r"subjects?|deaths?|outcomes?|occurrences?|instances?|participants?|"
    r"exposures?|times|percentages?|samples?|doses?|seconds?|minutes?|"
    r"hours?|days?|weeks?|months?|years?)"
)

SPELLED_OUT_QUANTITY_PATTERN = re.compile(
    rf"\b{_CARDINAL_WORD}(?:[\s-]+(?:and[\s-]+)?{_CARDINAL_WORD})*"
    rf"(?:\s+{_QUANTITY_MODIFIER})?\s+{_QUANTITY_NOUN}\b",
    re.IGNORECASE,
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
    r"\bcause\b",
    r"\bcauses\b",
    r"\bcausing\b",
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


def _find_unsupported_numbers(text: str) -> list[str]:
    """Find digits and common spelled-out quantities outside metric slots."""
    text_without_slots = SLOT_PATTERN.sub("", text)

    unsupported = [
        match.group(0)
        for match in re.finditer(r"\d+", text_without_slots)
    ]
    unsupported.extend(
        match.group(0)
        for match in SPELLED_OUT_QUANTITY_PATTERN.finditer(text_without_slots)
    )
    return unsupported


def _find_banned_claims(text: str) -> list[str]:
    lowered = text.lower()

    # Narrow exceptions for explicit discussion or uncertainty.
    # Non-greedy matching limits the exception to the first causal term.
    neutral_causal_context = re.compile(
        r"\b(?:discusses\s+whether|did\s+not\s+establish\s+that)\s+"
        # Allow only a short subject phrase before the causal term.
        # Do not cross predicates or common clause/conjunction markers.
        r"(?:(?!\b(?:is|are|was|were|be|been|being|has|have|had|"
        r"do|does|did|may|might|could|would|should|when|while|because|"
        r"but|however|yet|whereas|although|and|or|not|if|unless)\b)"
        r"[a-z][a-z-]*\s+){1,3}"
        r"\b(?:caused(?:\s+by)?|cause|causes|causing|led\s+to|resulted\s+in|"
        r"causally\s+related|causal\s+relationship)\b"
    )
    neutral_spans = [
        match.span()
        for match in neutral_causal_context.finditer(lowered)
    ]

    causal_terms = {
        "caused",
        "caused by",
        "cause",
        "causes",
        "causing",
        "led to",
        "resulted in",
        "causally related",
        "causal relationship",
    }

    matches: list[str] = []

    for pattern in BANNED_CLAIM_PATTERNS:
        for match in re.finditer(pattern, lowered):
            is_neutral_mention = (
                match.group(0) in causal_terms
                and any(
                    start <= match.start() and match.end() <= end
                    for start, end in neutral_spans
                )
            )
            if is_neutral_mention:
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

    unsupported_numbers = _find_unsupported_numbers(text)

    if unsupported_numbers:
        unique_numbers = list(dict.fromkeys(unsupported_numbers))

        issues.append(
            GroundingIssue(
                code="unsupported_number",
                message=(
                    "Narrative contains numeric expressions outside approved "
                    "metric slots: "
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