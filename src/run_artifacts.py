from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from src.llm_provider import LLMGenerationConfig
from src.report_pipeline import (
    ReportPipelineResult,
    render_html_report,
)
from src.report_review import save_review_set
from src.run_manifest import (
    build_run_manifest,
    write_manifest,
)


def _jsonable(value: Any) -> Any:
    """
    Convert supported project objects into JSON-compatible values.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    raise TypeError(
        f"Unsupported JSON value type: "
        f"{type(value).__name__}"
    )


def _write_json(
    value: Any,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def _section_metric_ids(
    result: ReportPipelineResult,
) -> dict[str, list[str]]:
    return {
        section.section_id: [
            metric.metric_id
            for metric in section.context.metrics
        ]
        for section in result.sections
    }


def _evidence_payload(
    result: ReportPipelineResult,
) -> list[Any]:
    return [
        section.evidence
        for section in result.sections
    ]


def _narratives_payload(
    result: ReportPipelineResult,
) -> dict[str, dict[str, str]]:
    """
    Persist the generated narrative templates so a reviewed run can be
    finalized without regenerating the LLM output.
    """
    return {
        section.section_id: {
            "narrative": section.narrative,
            "narrative_template": section.narrative_template,
        }
        for section in result.sections
    }


def write_run_artifacts(
    *,
    result: ReportPipelineResult,
    run_id: str,
    dataset_path: Path,
    report_config_path: Path,
    prompt_path: Path,
    generation_config: LLMGenerationConfig,
    template_path: Path,
    output_root: Path,
) -> Path:
    """
    Persist the complete artifact bundle for one pipeline run.

    The target directory must not already exist. This prevents an
    accidental run from silently overwriting a previous run.
    """
    output_root = Path(output_root)

    run_directory = output_root / run_id

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = build_run_manifest(
        run_id=run_id,
        dataset_path=dataset_path,
        report_config_path=report_config_path,
        prompt_path=prompt_path,
        generation_config=generation_config,
        report_id=result.config.report.id,
        report_version=result.config.report.version,
        analyses=result.analyses,
        section_metric_ids=_section_metric_ids(
            result
        ),
    )

    write_manifest(
        manifest,
        run_directory / "manifest.json",
    )

    save_review_set(
        result.review_set,
        run_directory / "review.json",
    )

    _write_json(
        result.analyses,
        run_directory / "analyses.json",
    )

    _write_json(
        _evidence_payload(result),
        run_directory / "evidence.json",
    )

    _write_json(
        _narratives_payload(result),
        run_directory / "narratives.json",
    )

    html = render_html_report(
        result,
        template_path,
    )

    (run_directory / "report.html").write_text(
        html,
        encoding="utf-8",
    )

    return run_directory