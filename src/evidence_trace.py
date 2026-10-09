from __future__ import annotations

from html import escape

from pydantic import BaseModel, ConfigDict, Field

from src.analysis_framework import AnalysisResult


class EvidenceTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    metric_ids: list[str]
    supporting_ids: dict[str, list[str]]


def build_evidence_trace(
    *,
    section_id: str,
    metric_ids: list[str],
    analyses: list[AnalysisResult],
) -> EvidenceTrace:
    results_by_id = {
        result.metric_id: result
        for result in analyses
    }

    missing = [
        metric_id
        for metric_id in metric_ids
        if metric_id not in results_by_id
    ]

    if missing:
        raise ValueError(
            "Missing analyses for evidence trace: "
            + ", ".join(missing)
        )

    supporting_ids = {
        metric_id: list(results_by_id[metric_id].supporting_ids)
        for metric_id in metric_ids
    }

    return EvidenceTrace(
        section_id=section_id,
        metric_ids=list(metric_ids),
        supporting_ids=supporting_ids,
    )


def evidence_anchor_id(metric_id: str) -> str:
    """
    Return the deterministic HTML anchor ID for one metric.

    The metric ID itself is not treated as trusted HTML.
    """
    return f"evidence-{metric_id}"


def render_evidence_html(
    trace: EvidenceTrace,
) -> str:
    """
    Render a deterministic, reviewer-facing evidence block.

    The evidence structure is:
        section -> metric -> supporting IDs

    Each metric receives a stable HTML anchor so narrative
    metric references can link directly to its evidence.

    All values are HTML-escaped before insertion.
    """
    parts: list[str] = [
        '<div class="evidence-block">',
        "<h3>Evidence Trace</h3>",
        '<div class="evidence-metrics">',
    ]

    for metric_id in trace.metric_ids:
        safe_metric_id = escape(metric_id)
        anchor_id = escape(
            evidence_anchor_id(metric_id),
            quote=True,
        )

        parts.append(
            '<div class="evidence-metric" '
            f'id="{anchor_id}">'
        )

        parts.append(
            f"<strong>Metric:</strong> "
            f"<code>{safe_metric_id}</code>"
        )

        supporting_ids = trace.supporting_ids.get(
            metric_id,
            [],
        )

        if supporting_ids:
            count = len(supporting_ids)
            parts.append(
                f"<div><strong>Supporting IDs:</strong> {count:,}</div>"
            )

            collapsed = count > 20
            if collapsed:
                parts.append(
                    "<details><summary>"
                    f"Show all {count:,} supporting IDs"
                    "</summary>"
                )

            parts.append("<ul>")
            for supporting_id in supporting_ids:
                parts.append(
                    f"<li><code>{escape(supporting_id)}</code></li>"
                )
            parts.append("</ul>")

            if collapsed:
                parts.append("</details>")
        else:
            parts.append(
                "<div><strong>Supporting IDs:</strong> "
                "None</div>"
            )

        parts.append("</div>")

    parts.extend(
        [
            "</div>",
            "</div>",
        ]
    )

    return "\n".join(parts)