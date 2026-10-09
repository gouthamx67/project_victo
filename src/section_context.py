from __future__ import annotations

from src.analysis_framework import (
    AnalysisResult,
    get_registered_analysis,
)
from src.llm_contract import MetricContext, SectionContext
from src.report_config import ReportConfig, SectionConfig


def _ensure_analysis_registry_loaded() -> None:
    """
    Ensure built-in analysis modules have registered their metrics.

    Registration is import-driven, so section-context construction must not
    depend on an unrelated caller having imported src.analyses first.
    """
    import src.analyses  # noqa: F401


def build_metric_context(result: AnalysisResult) -> MetricContext:
    """Convert a deterministic analysis result into LLM-approved context."""
    return MetricContext(
        metric_id=result.metric_id,
        value=result.value,
        definition=result.definition,
        population=result.population,
        supporting_ids=list(result.supporting_ids),
        version=result.version,
    )


def build_section_context(
    config: ReportConfig,
    section: SectionConfig,
    results: list[AnalysisResult],
) -> SectionContext:
    """
    Build the minimal evidence context required by one report section.

    Only metrics explicitly declared in section.needs are included.
    Raw dataframes are never accepted by this interface.
    """
    _ensure_analysis_registry_loaded()

    result_by_id = {result.metric_id: result for result in results}

    missing_metrics = [
        metric_id
        for metric_id in section.needs
        if metric_id not in result_by_id
    ]

    if missing_metrics:
        raise ValueError(
            f"Section {section.id!r} is missing required analyses: "
            f"{', '.join(missing_metrics)}"
        )

    metrics: list[MetricContext] = []

    for metric_id in section.needs:
        result = result_by_id[metric_id]
        registered = get_registered_analysis(metric_id)

        if registered.population != result.population:
            raise ValueError(
                f"Analysis population mismatch for {metric_id!r}: "
                f"registered={registered.population!r}, "
                f"result={result.population!r}"
            )

        metrics.append(build_metric_context(result))

    return SectionContext(
        report_id=config.report.id,
        report_version=config.report.version,
        section_id=section.id,
        section_title=section.title,
        instructions=section.instructions,
        style={
            "tone": section.style.tone,
            "length": section.style.length,
        },
        metrics=metrics,
    )