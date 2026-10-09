from __future__ import annotations
from src.run_manifest import sha256_file
from dataclasses import dataclass
from enum import Enum
from html import escape
from pathlib import Path

from src.analysis_framework import (
    AnalysisResult,
    run_all_analyses,
)
from src.evidence_trace import (
    EvidenceTrace,
    build_evidence_trace,
    render_evidence_html,
)
from src.llm_contract import (
    LLMNarrative,
    MetricContext,
    SectionContext,
)
from src.llm_provider import (
    LLMProvider,
    MockLLMProvider,
    NoLLMProvider,
    generate_grounded_narrative,
    load_prompt,
    prompt_hash,
    render_prompt,
)
from src.normalize import (
    NormalizedData,
    normalize_dataset,
)
from src.report_config import (
    ReportConfig,
    SectionConfig,
    load_report_config,
)
from src.report_renderer import (
    render_narrative,
    render_narrative_with_evidence,
)
from src.review import ReviewSet


class ReportStatus(str, Enum):
    DRAFT = "DRAFT"
    FINAL = "FINAL"


@dataclass(frozen=True)
class PipelineSection:
    section_id: str
    title: str
    ai_enabled: bool
    context: SectionContext
    narrative: LLMNarrative
    narrative_template: str
    evidence: EvidenceTrace
    provider_id: str = "unknown"
    model_id: str = "not_applicable"
    rendered_prompt_hash: str | None = None
    grounding_status: str = "not_recorded"
    generation_source: str = "unknown"


@dataclass(frozen=True)
class ReportPipelineResult:
    config: ReportConfig
    normalized: NormalizedData
    analyses: list[AnalysisResult]
    sections: list[PipelineSection]
    review_set: ReviewSet
    status: ReportStatus
    dataset_sha256: str = ""


def _ensure_builtin_analyses_loaded() -> None:
    import src.analyses  # noqa: F401


def _get_sections(
    config: ReportConfig,
) -> list[SectionConfig]:
    sections = getattr(
        config.report,
        "sections",
        None,
    )

    if sections is None:
        raise AttributeError(
            "ReportDefinition does not expose a "
            "'sections' field. "
            f"Available fields: "
            f"{list(config.report.__class__.model_fields)}"
        )

    return sorted(
        sections,
        key=lambda section: section.order,
    )


def _analysis_results_for_section(
    section: SectionConfig,
    results: list[AnalysisResult],
) -> list[AnalysisResult]:
    result_by_id = {
        result.metric_id: result
        for result in results
    }

    missing = [
        metric_id
        for metric_id in section.needs
        if metric_id not in result_by_id
    ]

    if missing:
        raise ValueError(
            f"Section {section.id!r} is missing "
            f"required analyses: "
            f"{', '.join(missing)}"
        )

    return [
        result_by_id[metric_id]
        for metric_id in section.needs
    ]


def _analysis_to_metric_context(
    result: AnalysisResult,
) -> MetricContext:
    return MetricContext(
        metric_id=result.metric_id,
        value=result.value,
        definition=result.definition,
        population=result.population,
        supporting_ids=[],
        version=result.version,
    )

def _build_section_context(
    config: ReportConfig,
    section: SectionConfig,
    results: list[AnalysisResult],
) -> SectionContext:
    section_results = _analysis_results_for_section(
        section,
        results,
    )

    metrics = [
        _analysis_to_metric_context(result)
        for result in section_results
    ]

    return SectionContext(
        report_id=config.report.id,
        report_version=config.report.version,
        section_id=section.id,
        section_title=section.title,
        metrics=metrics,
        instructions=section.instructions,
        style=section.style.model_dump(),
    )


def _approved_distribution_summary(
    context: SectionContext,
) -> str:
    metrics = {
        metric.metric_id: metric.value
        for metric in context.metrics
    }

    parts = []

    reactions = metrics.get("top_reactions")

    if isinstance(reactions, list) and reactions:
        first = reactions[0]

        if isinstance(first, dict) and "reaction_pt" in first:
            parts.append(f"Top reaction: {first['reaction_pt']}")

    ages = metrics.get("age_distribution")

    if isinstance(ages, list) and ages:
        known_ages = [
            item for item in ages
            if isinstance(item, dict)
            and item.get("age_bucket") != "unknown"
        ]

        if known_ages:
            dominant_age = max(
                known_ages,
                key=lambda item: item.get("count", 0),
            )
            parts.append(
                f"Most represented age group: {dominant_age['age_bucket']}"
            )

    sexes = metrics.get("sex_distribution")

    if isinstance(sexes, list) and sexes:
        counts = {
            item.get("sex"): item.get("count", 0)
            for item in sexes if isinstance(item, dict)
        }
        parts.append(
            f"Sex distribution: female {counts.get('female', 0)}, "
            f"male {counts.get('male', 0)}, "
            f"unknown {counts.get('unknown', 0)}"
        )

    return "; ".join(parts)


def _deterministic_section_narrative(
    section: SectionConfig,
    context: SectionContext,
) -> LLMNarrative:
    sentence_templates: dict[str, str] = {
        "case_count": (
            "The reporting period contains "
            "{case_count} unique safety cases."
        ),
        "serious_case_count": (
            "There are {serious_case_count} serious cases."
        ),
        "serious_case_percentage": (
            "Serious cases represent "
            "{serious_case_percentage} of the cases."
        ),
        "expedited_case_count": (
            "There are {expedited_case_count} expedited cases."
        ),
        "expedited_case_percentage": (
            "Expedited cases represent "
            "{expedited_case_percentage} of the cases."
        ),
        "fatal_case_count": (
            "There are {fatal_case_count} cases with a fatal case-level outcome."
        ),
        "not_recovered_case_count": (
            "The not-recovered case count is "
            "{not_recovered_case_count}."
        ),
        "recovered_case_count": (
            "The recovered case count is "
            "{recovered_case_count}."
        ),
        "unique_reaction_count": (
            "There are {unique_reaction_count} "
            "unique reported reaction terms."
        ),
        "reaction_count": (
            "There are {reaction_count} "
            "reported reaction records."
        ),
        "distinct_case_reaction_pairs": (
            "There are {distinct_case_reaction_pairs} distinct case–reaction pairs, "
            "counting each reaction term once per case."
        ),
        "top_reactions": (
            "The most frequently reported reactions are: "
            "{top_reactions}."
        ),
        "reaction_outcome_distribution": (
            "The reaction outcome distribution is: "
            "{reaction_outcome_distribution}."
        ),
        "case_outcome_distribution": (
            "The case outcome distribution is: "
            "{case_outcome_distribution}."
        ),
        "monthly_case_counts": (
            "Monthly case counts were: "
            "{monthly_case_counts}."
        ),
        "monthly_case_deviations": (
            "Monthly case-count review: {monthly_case_deviations}. "
"The first and last observed calendar months are excluded "
"from the full-month baseline when partial. Deviations are "
"descriptive flags for human review, not confirmed safety signals."
        ),
        "monthly_top_reactions": (
            "Monthly leading reaction terms were: "
            "{monthly_top_reactions}."
        ),
        "case_listing": (
            "{case_listing}"
        ),
    }

    sentences: list[str] = []

    for metric in context.metrics:
        template = sentence_templates.get(
            metric.metric_id
        )

        if template is None:
            display_name = (
                metric.metric_id
                .replace("_", " ")
                .capitalize()
            )

            template = (
                f"{display_name}: "
                f"{{{metric.metric_id}}}."
            )

        sentences.append(template)

    if not sentences:
        sentences.append(
            f"{section.title}: "
            "No approved metrics are available."
        )

    summary = _approved_distribution_summary(context)
    if summary:
        sentences.append(summary)

    return LLMNarrative(text=" ".join(sentences))


def _render_final_narrative(
    narrative: LLMNarrative,
    results: list[AnalysisResult],
) -> LLMNarrative:
    """
    Resolve the slot-bearing narrative into deterministic text.

    Important: LLMNarrative is a model object; render_narrative()
    accepts its text field, not the object itself.
    """
    rendered_text = render_narrative(
        narrative=narrative.text,
        results=results,
    )

    return LLMNarrative(
        text=rendered_text
    )


def run_report_pipeline(
    csv_path: str | Path,
    config_path: str | Path,
    prompt_path: str | Path,
    *,
    provider: LLMProvider | None = None,
    use_llm: bool = False,
    max_retries: int = 1,
) -> ReportPipelineResult:
    _ensure_builtin_analyses_loaded()

    config = load_report_config(
        config_path
    )

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    prompt_text = load_prompt(
        Path(prompt_path)
    )

    from src.ingest import ingest_and_validate

    dataframe, validation = ingest_and_validate(
        csv_path
    )

    if not validation.is_valid:
        errors = "; ".join(
            validation.errors
        )

        raise ValueError(
            f"Dataset validation failed: {errors}"
        )

    normalized = normalize_dataset(
        dataframe
    )

    analyses = run_all_analyses(
        normalized
    )

    if not analyses:
        raise RuntimeError(
            "Analysis registry produced no results. "
            "Expected deterministic analysis modules "
            "to be registered."
        )

    if use_llm:
        selected_provider: LLMProvider = (
            provider
            if provider is not None
            else MockLLMProvider()
        )
    else:
        selected_provider = NoLLMProvider()

    pipeline_sections: list[
        PipelineSection
    ] = []

    for section in _get_sections(config):
        context = _build_section_context(
            config=config,
            section=section,
            results=analyses,
        )

        evidence = build_evidence_trace(
            section_id=section.id,
            metric_ids=list(section.needs),
            analyses=analyses,
        )

        rendered_prompt = None
        provider_id = type(selected_provider).__name__
        model_id = getattr(
            getattr(selected_provider, "config", None),
            "model",
            "not_applicable",
        )
        grounding_status = "not_applicable_deterministic"
        generation_source = "deterministic_template"

        if not section.ai:
            narrative = _deterministic_section_narrative(
                section=section,
                context=context,
            )

        elif use_llm:
            rendered_prompt = render_prompt(
                prompt_text,
                context,
            )

            provider_id = type(selected_provider).__name__
            generation_source = "provider"
            narrative = generate_grounded_narrative(
                provider=selected_provider,
                context=context,
                prompt=rendered_prompt,
                results=analyses,
                required_metric_ids=list(
                    section.needs
                ),
                max_retries=max_retries,
                fallback_provider=NoLLMProvider(),
            )

        else:
            rendered_prompt = None
            provider_id = "NoLLMProvider"
            generation_source = "no_llm_deterministic"
            narrative = _deterministic_section_narrative(
                section=section,
                context=context,
            )

        if narrative is None:
            raise RuntimeError(
                f"Narrative generation returned None "
                f"for section {section.id!r}."
            )

        narrative_template = narrative.text
        if section.ai and generation_source == "provider":
            grounding_status = "passed"
            if narrative.text.startswith("[NO-LLM DRAFT]"):
                generation_source = "deterministic_fallback"
                provider_id = "NoLLMProvider"
                model_id = "not_applicable"
                grounding_status = "not_applicable_deterministic"

        rendered = _render_final_narrative(
            narrative=narrative,
            results=analyses,
        )

        pipeline_sections.append(
            PipelineSection(
                section_id=section.id,
                title=section.title,
                ai_enabled=section.ai,
                context=context,
                narrative=rendered,
                narrative_template=narrative_template,
                evidence=evidence,
                provider_id=provider_id,
                model_id=model_id,
                rendered_prompt_hash=(
                    prompt_hash(rendered_prompt)
                    if rendered_prompt is not None
                    else None
                ),
                grounding_status=grounding_status,
                generation_source=generation_source,
            )
        )

    from src.report_review import (
        make_initial_review_set,
    )

    review_set = make_initial_review_set(
        config
    )

    return ReportPipelineResult(
        config=config,
        normalized=normalized,
        analyses=analyses,
        sections=pipeline_sections,
        review_set=review_set,
        status=ReportStatus.DRAFT,
        dataset_sha256=sha256_file(csv_path),
    )


def _render_case_listing_table(section: PipelineSection) -> str:
    """Render deterministic case-listing records as an escaped HTML table."""
    metric = next(
        (
            item
            for item in section.context.metrics
            if item.metric_id == "case_listing"
        ),
        None,
    )

    if metric is None or not isinstance(metric.value, (list, tuple)):
        return (
            '<p class="case-listing-empty">'
            "Case listing data is unavailable."
            "</p>"
        )

    headers = (
        ("case_id", "Case ID"),
        ("reactions", "Reaction(s)"),
        ("serious", "Seriousness"),
        ("receivedate", "Receive date"),
        ("country", "Country"),
        ("outcome", "Outcome"),
    )

    def display_value(key: str, value: object) -> str:
        if key == "reactions":
            if isinstance(value, (list, tuple)):
                items = [str(item) for item in value]
                shown = items[:5]
                if len(items) > 5:
                    shown.append(f"+{len(items) - 5} more")
                value = "; ".join(shown)
            else:
                value = str(value) if value is not None else ""

        elif key == "serious":
            value = (
                "Yes" if value is True
                else "No" if value is False
                else "Unknown"
            )

        elif value is None or not str(value).strip():
            value = "Not reported"

        return escape(str(value))

    table_rows = []

    for record in metric.value:
        if not isinstance(record, dict):
            continue

        cells = "".join(
            "<td>"
            + display_value(key, record.get(key))
            + "</td>"
            for key, _ in headers
        )
        table_rows.append(f"<tr>{cells}</tr>")

    header_html = "".join(
        f"<th scope=\"col\">{escape(label)}</th>"
        for _, label in headers
    )

    return (
        '<div class="case-listing-table-wrapper">'
        '<table class="case-listing-table">'
        "<thead><tr>"
        + header_html
        + "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
    )



_STRUCTURED_TABLE_METRICS = {
    "age_distribution",
    "sex_distribution",
    "country_distribution",
    "top_reactions",
    "top_serious_reactions",
    "reaction_outcome_distribution",
    "case_outcome_distribution",
    "monthly_case_counts",
    "monthly_case_deviations",
    "monthly_top_reactions",
}

_TABLE_COLUMN_LABELS = {
    "age_bucket": "Age group",
    "sex": "Sex",
    "country_or_region": "Country / region",
    "reaction_pt": "Reaction Preferred Term",
    "occurrences": "Reaction occurrences",
    "distinct_cases": "Distinct cases",
    "count": "Count",
    "percentage": "Percentage (%)",
    "case_count": "Cases",
    "reporting_month": "Reporting month",
    "reaction_count": "Reaction occurrences",
    "outcome_category": "Outcome",
    "case_outcome": "Case outcome",
    "baseline_eligible": "Full-month baseline eligible",
    "mean_full_month_case_count": "Mean full-month cases",
    "mean_monthly_case_count": "Mean monthly cases",
    "deviation_threshold": "Review threshold (cases)",
    "deviation_flag": "Flagged for human review",
}


def _render_structured_metric_tables(section: PipelineSection) -> str:
    """Render supported structured metrics as accessible HTML tables."""
    tables = []

    for metric in section.context.metrics:
        if metric.metric_id not in _STRUCTURED_TABLE_METRICS:
            continue
        if not isinstance(metric.value, (list, tuple)) or not metric.value:
            continue
        records = [row for row in metric.value if isinstance(row, dict)]
        if not records:
            continue

        keys = list(dict.fromkeys(
            key for record in records for key in record
        ))
        if not keys:
            continue

        display_records = records
        percentage_metrics = {
            "age_distribution",
            "sex_distribution",
            "country_distribution",
            "case_outcome_distribution",
        }
        if (
            metric.metric_id in percentage_metrics
            and all(
                isinstance(row.get("count"), (int, float))
                and not isinstance(row.get("count"), bool)
                for row in records
            )
        ):
            total = sum(row["count"] for row in records)
            display_records = [
                {
                    **row,
                    "percentage": (
                        round(row["count"] * 100 / total, 1)
                        if total else 0.0
                    ),
                }
                for row in records
            ]
            if "percentage" not in keys:
                keys.append("percentage")

        def cell_value(key, value):
            if value is None or value == "":
                return "Not reported"
            if isinstance(value, bool):
                return "Yes" if value else "No"
            if isinstance(value, int):
                return f"{value:,}"
            if isinstance(value, float):
                return f"{value:,.1f}"
            if isinstance(value, str):
                if value == "none_provided":
                    return "None provided"
                if value.casefold() == "unknown":
                    return "Unknown"
                if key == "country_or_region":
                    if value.casefold() == "european union (region)":
                        return "European Union (region)"
                    return value.title()
            return str(value)

        headers = "".join(
            f'<th scope="col">{escape(_TABLE_COLUMN_LABELS.get(key, key.replace("_", " ").title()))}</th>'
            for key in keys
        )
        rows = []
        for record in display_records:
            cells = "".join(
                f"<td>{escape(cell_value(key, record.get(key)))}</td>"
                for key in keys
            )
            rows.append(f"<tr>{cells}</tr>")

        title = escape(metric.metric_id.replace("_", " ").title())
        tables.append(
            f'<div class="structured-table-wrapper">'
            f'<h3>{title}</h3>'
            f'<table class="structured-data-table">'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )

    return "".join(tables)


def _remove_table_metric_sentences(
    narrative: str,
    section: PipelineSection,
) -> str:
    """Avoid repeating structured tables as long inline prose dumps."""
    import re

    table_ids = {
        metric.metric_id
        for metric in section.context.metrics
        if metric.metric_id in _STRUCTURED_TABLE_METRICS
        and isinstance(metric.value, (list, tuple))
        and metric.value
    }
    for metric_id in table_ids:
        narrative = re.sub(
            rf"[^.]*\{{{re.escape(metric_id)}\}}[^.]*\.?\s*",
            "",
            narrative,
        )
    return narrative.strip()


def render_html_report(
    result: ReportPipelineResult,
    template_path: str | Path,
) -> str:
    template_path = Path(
        template_path
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Report template not found: "
            f"{template_path}"
        )

    template = template_path.read_text(
        encoding="utf-8"
    )

    section_html: list[str] = []

    for section in result.sections:
        if section.section_id in {"case_listing", "case_index_listing"}:
            narrative_html = _render_case_listing_table(section)
        else:
            table_html = _render_structured_metric_tables(section)
            narrative_template = _remove_table_metric_sentences(
                section.narrative_template,
                section,
            )
            if narrative_template.strip():
                narrative_html = render_narrative_with_evidence(
                    narrative=narrative_template,
                    results=section.context.metrics,
                )
            else:
                narrative_html = ""
            narrative_html += table_html

            # Keep the trend-method explanation even when the metric
            # sentence is removed to avoid duplicating the data table.
            if section.section_id == "trends_and_important_observations":
                narrative_html += (
                    '<p class="trend-method-note">'
                    "Trend review method: the first and last observed "
                    "calendar months are excluded from the full-month "
                    "baseline when partial. A month is flagged only when its "
                    "case count is strictly greater than 1.5 times the eligible "
                    "full-month mean. Flags are descriptive and intended for "
                    "human review; they are not confirmed safety signals."
                    "</p>"
                )

            # Keep evidence links available without repeating a long
            # inline list below every structured table.
            missing_metric_links = []
            for metric in section.context.metrics:
                anchor = f'href="#evidence-{escape(metric.metric_id)}"'
                if anchor in narrative_html:
                    continue
                link_html = render_narrative_with_evidence(
                    narrative="{" + metric.metric_id + "}",
                    results=section.context.metrics,
                )
                if metric.metric_id == "fatal_case_count":
                    value = (
                        f"{metric.value:,}"
                        if isinstance(metric.value, int)
                        else str(metric.value)
                    )
                    link_html = link_html.replace(
                        f">{escape(value)}</a>",
                        f">{escape(value)} cases with a fatal case-level outcome</a>",
                    )
                missing_metric_links.append(
                    '<span class="table-metric-evidence">'
                    + escape(metric.metric_id.replace("_", " ").title())
                    + ": "
                    + link_html
                    + "</span>"
                )
            if missing_metric_links:
                narrative_html += (
                    '<details class="table-metric-evidence-links">'
                    '<summary>Additional metric evidence links</summary>'
                    + " ".join(missing_metric_links)
                    + "</details>"
                )

        identification_html = ""
        if section.section_id == "reporting_period":
            report = result.config.report
            identification_html = (
                '<dl class="report-identification">'
                f'<dt>Product name</dt><dd>{escape(report.product_name)}</dd>'
                f'<dt>Application identifier</dt><dd>{escape(report.application_identifier)}</dd>'
                f'<dt>Report type</dt><dd>{escape(report.report_type)}</dd>'
                f'<dt>Data-source SHA-256</dt><dd><code>{escape(result.dataset_sha256)}</code></dd>'
                '</dl>'
            )

        evidence_html = render_evidence_html(
            section.evidence
        )

        section_html.append(
            f"""
<section class="report-section"
         id="section-{escape(section.section_id)}">
    <h2>{escape(section.title)}</h2>
    <details class="generation-metadata">
        <summary>Generation and validation metadata</summary>
        <dl>
            <dt>Provider ID</dt><dd>{escape(section.provider_id)}</dd>
            <dt>Model ID</dt><dd>{escape(section.model_id)}</dd>
            <dt>Rendered prompt SHA-256</dt>
            <dd>{escape(section.rendered_prompt_hash or "Not applicable")}</dd>
            <dt>Grounding validation</dt><dd>{escape(section.grounding_status)}</dd>
            <dt>Generation source</dt><dd>{escape(section.generation_source)}</dd>
        </dl>
    </details>
    <div class="narrative">
        {narrative_html}
    </div>
    {identification_html}
    {evidence_html}
</section>
""".strip()
        )

    if result.status == ReportStatus.FINAL:
        status_html = (
            '<div class="report-status report-status-final">'
            "FINAL — HUMAN REVIEW APPROVED"
            "</div>"
        )
    else:
        status_html = (
            '<div class="report-status report-status-draft">'
            "DRAFT — HUMAN REVIEW REQUIRED"
            "</div>"
        )

    html = template.replace(
        "{{REPORT_TITLE}}",
        escape(
            result.config.report.title
        ),
    )

    html = html.replace(
        "{{REPORT_ID}}",
        escape(
            result.config.report.id
        ),
    )

    html = html.replace(
        "{{REPORT_VERSION}}",
        escape(
            result.config.report.version
        ),
    )

    html = html.replace(
        "{{REPORT_STATUS}}",
        escape(
            result.status.value
        ),
    )

    html = html.replace(
        "{{REPORT_STATUS_BANNER}}",
        status_html,
    )

    html = html.replace(
        "{{SECTION_CONTENT}}",
        "\n".join(section_html),
    )

    return html


def write_html_report(
    result: ReportPipelineResult,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html = render_html_report(
        result=result,
        template_path=template_path,
    )

    output_path.write_text(
        html,
        encoding="utf-8",
    )

    return output_path