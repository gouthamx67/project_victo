from __future__ import annotations

import re
from html import escape
from typing import Any

from src.analysis_framework import AnalysisResult
from src.evidence_trace import evidence_anchor_id
from src.metric_slots import MetricSlotError


_METRIC_SLOT_PATTERN = re.compile(
    r"\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


def _result_by_id(
    results: list[Any],
) -> dict[str, Any]:
    return {
        result.metric_id: result
        for result in results
    }


def _format_scalar_display_value(
    metric_id: str,
    value: Any,
) -> str:
    """
    Format scalar metric values for deterministic report prose.

    Scalar count metrics use thousands separators.
    Percentage metrics use one decimal place and include '%'.
    """
    if value is None:
        return "Not available"

    if isinstance(value, str):
        if value == "none_provided":
            return "None provided"
        if value.casefold() == "unknown":
            return "Unknown"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        if metric_id.endswith("_percentage"):
            return f"{value:.1f}%"

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:,.2f}".rstrip("0").rstrip(".")

    return str(value)


def _unit_label_for_structured_field(
    metric_id: str,
    key: str,
) -> str | None:
    """
    Return the semantic unit for numeric fields inside structured metrics.

    This prevents values such as '68' or '25' from appearing without
    explaining whether they represent cases or reaction occurrences.
    """
    if key == "count":
        if metric_id == "reaction_outcome_distribution":
            return "reaction occurrences"

        if metric_id == "case_outcome_distribution":
            return "cases"

        if metric_id == "monthly_case_counts":
            return "cases"

        if metric_id == "monthly_case_deviations":
            return "cases"

    if key == "occurrences":
        if metric_id in {
            "top_reactions",
            "monthly_top_reactions",
        }:
            return "reaction occurrences"

    if key == "distinct_cases":
        if metric_id == "top_reactions":
            return "cases"

    if key == "case_count":
        return "cases"

    return None


def _format_structured_value(
    metric_id: str,
    value: Any,
) -> str:
    """
    Format structured deterministic metric values.

    Field labels are retained and numeric fields receive their semantic
    units where the metric contract defines them.
    """
    if value is None:
        return "Not available"

    if isinstance(value, dict):
        if not value:
            return "None"

        parts: list[str] = []

        for key, item in value.items():
            key_text = str(key).replace(
                "_",
                " ",
            )

            key_text = (
                key_text[:1].upper()
                + key_text[1:]
            )

            if isinstance(item, (int, float)) and not isinstance(
                item,
                bool,
            ):
                formatted_item = _format_scalar_display_value(
                    "",
                    item,
                )
            elif isinstance(item, str):
                formatted_item = _format_scalar_display_value(
                    "",
                    item,
                )
            else:
                formatted_item = _format_structured_value(
                    metric_id,
                    item,
                )

            if isinstance(item, (int, float, str)) and not isinstance(
                item,
                bool,
            ):
                unit_label = _unit_label_for_structured_field(
                    metric_id,
                    str(key),
                )

                if unit_label:
                    formatted_item = (
                        f"{formatted_item} "
                        f"{unit_label}"
                    )
            else:
                formatted_item = _format_structured_value(
                    metric_id,
                    item,
                )

            parts.append(
                f"{key_text}: {formatted_item}"
            )

        return "; ".join(parts)

    if isinstance(value, (list, tuple)):
        if not value:
            return "None"

        return ", ".join(
            _format_structured_value(
                metric_id,
                item,
            )
            for item in value
        )

    return _format_scalar_display_value(
        metric_id,
        value,
    )


def _format_display_value(
    value: Any,
    metric_id: str = "",
) -> str:
    """
    Format an approved metric for deterministic narrative rendering.
    """
    return _format_structured_value(
        metric_id,
        value,
    )


def _format_value_for_evidence(
    metric_id: str,
    value: Any,
) -> str:
    """
    Produce readable evidence-link text.

    This is presentation-only and does not alter the underlying metric.
    """
    return _format_structured_value(
        metric_id,
        value,
    )


def format_metric_value(
    metric_id: str,
    value: Any,
) -> str:
    """
    Format metric values for readable report output.

    Structured metrics receive metric-specific formatting instead of
    being dumped as Python dictionaries or lists.
    """
    if metric_id == "case_listing":
        if isinstance(value, (list, tuple)):
            return (
                f"{len(value):,} case records are available "
                "in the deterministic case listing."
            )

    if metric_id == "monthly_case_counts":
        if isinstance(value, (list, tuple)):
            if not value:
                return "None"

            rows: list[str] = []

            for item in value:
                if not isinstance(item, dict):
                    return _format_display_value(
                        value,
                        metric_id=metric_id,
                    )

                month = item.get("reporting_month", "Unknown")
                count = item.get("case_count")

                formatted_count = _format_scalar_display_value(
                    "case_count",
                    count,
                )

                rows.append(
                    f"{month}: {formatted_count} cases"
                )

            return "; ".join(rows)

    if metric_id == "monthly_case_deviations":
        if isinstance(value, (list, tuple)):
            if not all(isinstance(item, dict) for item in value):
                return _format_display_value(
                    value,
                    metric_id=metric_id,
                )

            if not value:
                return "no monthly deviation records available"

            flagged = [
                item
                for item in value
                if item.get("deviation_flag") is True
            ]

            if not flagged:
                return (
                    "the absence of monthly deviations "
                    "above the configured threshold"
                )

            months = [
                str(item.get("reporting_month", "Unknown"))
                for item in flagged
            ]

            return (
                "monthly deviations flagged for human review in "
                + ", ".join(months)
                + " (descriptive flags, not confirmed signals)"
            )

    return _format_display_value(
        value,
        metric_id=metric_id,
    )


def render_narrative(
    narrative: str,
    results: list[AnalysisResult],
) -> str:
    """
    Resolve approved metric slots.

    Scalar counts use thousands separators.
    Percentages use one decimal place.
    Structured numeric fields retain explicit semantic units.
    """
    if not isinstance(narrative, str):
        raise TypeError(
            "narrative must be a string."
        )

    if not narrative.strip():
        raise ValueError(
            "narrative must not be empty."
        )

    results_by_id = _result_by_id(
        results
    )

    output: list[str] = []
    cursor = 0

    for match in _METRIC_SLOT_PATTERN.finditer(
        narrative
    ):
        output.append(
            narrative[
                cursor:match.start()
            ]
        )

        metric_id = match.group(1)

        if metric_id not in results_by_id:
            raise MetricSlotError(
                f"Unknown metric slot: {metric_id}"
            )

        result = results_by_id[
            metric_id
        ]

        output.append(
            format_metric_value(
                metric_id,
                result.value,
            )
        )

        cursor = match.end()

    output.append(
        narrative[cursor:]
    )

    return "".join(output)


def render_section(
    narrative: str,
    results: list[AnalysisResult],
) -> str:
    """
    Backwards-compatible section-level renderer.
    """
    return render_narrative(
        narrative=narrative,
        results=results,
    )


def render_narrative_with_evidence(
    narrative: str,
    results: list[Any],
) -> str:
    """
    Resolve metric slots and make each resolved value clickable
    to its deterministic evidence block.
    """
    if not isinstance(narrative, str):
        raise TypeError(
            "narrative must be a string."
        )

    if not narrative.strip():
        raise ValueError(
            "narrative must not be empty."
        )

    results_by_id = _result_by_id(
        results
    )

    output: list[str] = []
    cursor = 0

    for match in _METRIC_SLOT_PATTERN.finditer(
        narrative
    ):
        output.append(
            escape(
                narrative[
                    cursor:match.start()
                ]
            )
        )

        metric_id = match.group(1)

        if metric_id not in results_by_id:
            raise MetricSlotError(
                f"Unknown metric slot: {metric_id}"
            )

        result = results_by_id[
            metric_id
        ]

        value_text = _format_value_for_evidence(
            metric_id,
            result.value,
        )

        href = (
            "#"
            + evidence_anchor_id(
                metric_id
            )
        )

        output.append(
            f'<a class="metric-evidence-link" '
            f'href="{escape(href, quote=True)}" '
            f'data-metric-id="'
            f'{escape(metric_id, quote=True)}">'
            f'{escape(value_text)}'
            f'</a>'
        )

        cursor = match.end()

    output.append(
        escape(
            narrative[cursor:]
        )
    )

    return "".join(output)