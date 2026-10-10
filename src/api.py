from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from src.analysis_framework import AnalysisResult
from src.evidence_trace import build_evidence_trace
from src.llm_provider import (
    LLMGenerationConfig,
    LLMProvider,
    build_provider,
)
from src.report_config import load_report_config
from src.report_pipeline import (
    PipelineSection,
    ReportPipelineResult,
    ReportStatus,
    _build_section_context,
    render_html_report,
    run_report_pipeline,
)
from src.report_review import (
    ReportFinalizationError,
    assert_report_can_be_finalized,
    finalize_report,
    load_review_set,
    validate_report_review,
)
from src.review import (
    ReviewSet,
    ReviewStatus,
    approve_section,
    flag_section,
)
from src.run_artifacts import write_run_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pader.yaml"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "sections" / "default.txt"
DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "report_template.html"

API_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "api_runs"


class ReviewActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)

    status: Literal[
        ReviewStatus.APPROVED,
        ReviewStatus.FLAGGED,
    ]

    reviewer: str = Field(min_length=1)
    comment: str = ""


app = FastAPI(
    title="Regulatory Safety Reporting API",
    version="1.0.0",
    description=(
        "Backend API for the deterministic safety-reporting pipeline. "
        "The API is a thin transport layer over the existing report pipeline."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run_directory(run_id: str) -> Path:
    return API_OUTPUT_ROOT / run_id


def _require_run(run_id: str) -> Path:
    run_directory = _run_directory(run_id)

    if not run_directory.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' was not found.",
        )

    return run_directory


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{path.name}' was not found.",
        )

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Artifact '{path.name}' is invalid JSON.",
        ) from exc


def _build_generation_config() -> LLMGenerationConfig:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in {"openai", "groq"}:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'groq'.")

    default_model = (
        "openai/gpt-oss-120b"
        if provider == "groq"
        else "gpt-5-mini"
    )
    model_variable = (
        "GROQ_MODEL"
        if provider == "groq"
        else "OPENAI_MODEL"
    )

    return LLMGenerationConfig(
        provider=provider,
        model=os.getenv(model_variable, default_model),
        temperature=0.0,
        max_tokens=1000,
    )


def _build_llm_provider() -> LLMProvider:
    generation_config = _build_generation_config()

    provider = build_provider(
        provider=generation_config.provider,
        model=generation_config.model,
        max_tokens=generation_config.max_tokens,
    )

    if provider is None:
        raise ValueError("LLM provider is not configured.")

    return provider


def _load_review_set_for_run(
    run_directory: Path,
) -> ReviewSet:
    review_path = run_directory / "review.json"

    if not review_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Review artifact was not found.",
        )

    try:
        return load_review_set(review_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Review artifact is invalid: {exc}",
        ) from exc


def _load_report_config():
    try:
        return load_report_config(DEFAULT_CONFIG_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Report configuration is invalid: {exc}",
        ) from exc


def _load_analysis_results(
    run_directory: Path,
) -> list[AnalysisResult]:
    payload = _read_json(
        run_directory / "analyses.json",
    )

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=500,
            detail="Analysis artifact must contain a JSON list.",
        )

    try:
        return [
            AnalysisResult.model_validate(item)
            for item in payload
        ]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis artifact is invalid: {exc}",
        ) from exc


def _extract_persisted_narrative(
    value: Any,
    *,
    section_id: str,
) -> str:
    """
    Normalize persisted narrative payloads.

    Current artifacts may contain either:

        "narrative": "plain text"

    or:

        "narrative": {
            "text": "plain text"
        }

    The API exposes the normalized text to the frontend and uses
    the same normalized text when reconstructing a report for
    finalization.
    """

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        text = value.get("text")

        if isinstance(text, str):
            return text

    raise HTTPException(
        status_code=500,
        detail=(
            f"Narrative entry for '{section_id}' has an unsupported "
            "narrative format. Expected a string or an object "
            "containing a string 'text' field."
        ),
    )


def _load_narratives(
    run_directory: Path,
) -> dict[str, dict[str, str]]:
    payload = _read_json(
        run_directory / "narratives.json",
    )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="Narrative artifact must contain a JSON object.",
        )

    normalized: dict[str, dict[str, str]] = {}

    for section_id, value in payload.items():
        if not isinstance(section_id, str):
            raise HTTPException(
                status_code=500,
                detail="Narrative section IDs must be strings.",
            )

        if not isinstance(value, dict):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Narrative entry for '{section_id}' "
                    "must be an object."
                ),
            )

        narrative = _extract_persisted_narrative(
            value.get("narrative"),
            section_id=section_id,
        )

        narrative_template = value.get("narrative_template")

        if not isinstance(narrative_template, str):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Narrative entry for '{section_id}' "
                    "is missing a valid 'narrative_template'."
                ),
            )

        normalized[section_id] = {
            "narrative": narrative,
            "narrative_template": narrative_template,
        }

    return normalized


def _reconstruct_result_for_finalization(
    run_directory: Path,
    review_set: ReviewSet,
) -> ReportPipelineResult:
    config = _load_report_config()

    analyses = _load_analysis_results(
        run_directory,
    )

    narratives = _load_narratives(
        run_directory,
    )

    configured_sections = sorted(
        config.report.sections,
        key=lambda section: section.order,
    )

    sections: list[PipelineSection] = []

    for section_config in configured_sections:
        narrative_data = narratives.get(
            section_config.id,
        )

        if narrative_data is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Narrative artifact is missing section "
                    f"'{section_config.id}'."
                ),
            )

        context = _build_section_context(
            config=config,
            section=section_config,
            results=analyses,
        )

        evidence = build_evidence_trace(
            section_id=section_config.id,
            metric_ids=list(section_config.needs),
            analyses=analyses,
        )

        sections.append(
            PipelineSection(
                section_id=section_config.id,
                title=section_config.title,
                narrative=narrative_data["narrative"],
                narrative_template=(
                    narrative_data["narrative_template"]
                ),
                context=context,
                evidence=evidence,
            )
        )

    return ReportPipelineResult(
        config=config,
        normalized=None,
        analyses=analyses,
        sections=sections,
        review_set=review_set,
        status=ReportStatus.DRAFT,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "regulatory-safety-reporting-api",
    }


@app.post("/runs", status_code=201)
async def create_run(
    file: UploadFile = File(...),
    use_llm: bool = Form(False),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A CSV file is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    run_id = uuid.uuid4().hex

    API_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_directory = API_OUTPUT_ROOT / "_inputs"

    input_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = input_directory / f"{run_id}.csv"

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty.",
            )

        dataset_path.write_bytes(content)

        if use_llm:
            generation_config = _build_generation_config()
            provider = _build_llm_provider()
        else:
            generation_config = LLMGenerationConfig(
                provider="none",
                model="none",
                temperature=0.0,
                max_tokens=1000,
            )
            provider = None

        result: ReportPipelineResult = run_report_pipeline(
            csv_path=dataset_path,
            config_path=DEFAULT_CONFIG_PATH,
            prompt_path=DEFAULT_PROMPT_PATH,
            provider=provider,
            use_llm=use_llm,
        )

        run_directory = write_run_artifacts(
            result=result,
            run_id=run_id,
            dataset_path=dataset_path,
            report_config_path=DEFAULT_CONFIG_PATH,
            prompt_path=DEFAULT_PROMPT_PATH,
            generation_config=generation_config,
            template_path=DEFAULT_TEMPLATE_PATH,
            output_root=API_OUTPUT_ROOT,
        )

    except HTTPException:
        raise

    except Exception as exc:
        if dataset_path.exists():
            dataset_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "run_id": run_id,
        "status": result.status.value,
        "report_id": result.config.report.id,
        "report_version": result.config.report.version,
        "llm_enabled": use_llm,
        "artifact_directory": str(run_directory),
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run_directory = _require_run(run_id)

    manifest = _read_json(
        run_directory / "manifest.json",
    )

    report_path = run_directory / "report.html"

    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Report artifact was not found.",
        )

    report_html = report_path.read_text(
        encoding="utf-8",
    )

    if "FINAL — HUMAN REVIEW APPROVED" in report_html:
        report_status = "FINAL"
    else:
        report_status = "DRAFT"

    return {
        "run_id": run_id,
        "status": report_status,
        "report_id": manifest.get("report_id"),
        "report_version": manifest.get("report_version"),
        "created_at_utc": manifest.get("created_at_utc"),
        "dataset_sha256": manifest.get("dataset_sha256"),
        "analysis_registry_version": manifest.get(
            "analysis_registry_version"
        ),
        "artifact_files": sorted(
            path.name
            for path in run_directory.iterdir()
            if path.is_file()
        ),
    }


@app.get(
    "/runs/{run_id}/report",
    response_class=HTMLResponse,
)
def get_report(run_id: str) -> HTMLResponse:
    run_directory = _require_run(run_id)

    report_path = run_directory / "report.html"

    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Report artifact was not found.",
        )

    return HTMLResponse(
        content=report_path.read_text(
            encoding="utf-8",
        )
    )


@app.get("/runs/{run_id}/manifest")
def get_manifest(run_id: str) -> dict[str, Any]:
    run_directory = _require_run(run_id)

    return _read_json(
        run_directory / "manifest.json",
    )


@app.get("/runs/{run_id}/review")
def get_review(run_id: str) -> dict[str, Any]:
    run_directory = _require_run(run_id)

    return _read_json(
        run_directory / "review.json",
    )


@app.get("/runs/{run_id}/evidence")
def get_evidence(run_id: str) -> Any:
    run_directory = _require_run(run_id)

    return _read_json(
        run_directory / "evidence.json",
    )


@app.get("/runs/{run_id}/narratives")
def get_narratives(run_id: str) -> dict[str, Any]:
    """
    Return persisted section narratives for the reviewer UI.

    This endpoint intentionally exposes the narrative artifact,
    not the raw CSV and not the full analysis support IDs.
    """

    run_directory = _require_run(run_id)

    narratives = _read_json(
        run_directory / "narratives.json",
    )

    if not isinstance(narratives, dict):
        raise HTTPException(
            status_code=500,
            detail="Narrative artifact must contain a JSON object.",
        )

    return narratives


@app.post("/runs/{run_id}/review")
def update_review(
    run_id: str,
    request: ReviewActionRequest,
) -> dict[str, Any]:
    run_directory = _require_run(run_id)

    review_set = _load_review_set_for_run(
        run_directory,
    )

    section_ids = {
        review.section_id
        for review in review_set.reviews
    }

    if request.section_id not in section_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown section '{request.section_id}'."
            ),
        )

    try:
        if request.status == ReviewStatus.APPROVED:
            updated_review = approve_section(
                section_id=request.section_id,
                reviewer=request.reviewer,
                comment=request.comment,
            )
        else:
            updated_review = flag_section(
                section_id=request.section_id,
                reviewer=request.reviewer,
                comment=request.comment,
            )

        updated_reviews = [
            updated_review
            if review.section_id == request.section_id
            else review
            for review in review_set.reviews
        ]

        updated_review_set = ReviewSet(
            reviews=updated_reviews,
        )

        review_path = run_directory / "review.json"

        review_path.write_text(
            updated_review_set.model_dump_json(
                indent=2,
            ),
            encoding="utf-8",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "run_id": run_id,
        "section_id": request.section_id,
        "status": updated_review.status,
        "reviewer": updated_review.reviewer,
        "comment": updated_review.comment,
        "reviewed_at": updated_review.reviewed_at,
    }


@app.post("/runs/{run_id}/finalize")
def finalize_run(
    run_id: str,
) -> dict[str, Any]:
    run_directory = _require_run(run_id)

    review_set = _load_review_set_for_run(
        run_directory,
    )

    # Validate the human-review gate FIRST.
    # An unreviewed/flagged run must return 409 even if other
    # persisted report artifacts are unavailable.
    config = _load_report_config()

    try:
        validate_report_review(
            config=config,
            review_set=review_set,
        )

        assert_report_can_be_finalized(
            config=config,
            review_set=review_set,
        )

    except ReportFinalizationError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    # Only after the review gate passes do we reconstruct the
    # persisted report and render the FINAL artifact.
    result = _reconstruct_result_for_finalization(
        run_directory=run_directory,
        review_set=review_set,
    )

    final_result = finalize_report(
        result,
        review_set,
    )

    html = render_html_report(
        final_result,
        DEFAULT_TEMPLATE_PATH,
    )

    (run_directory / "report.html").write_text(
        html,
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "status": final_result.status.value,
        "message": (
            "All configured sections are approved and the "
            "report artifact has been finalized."
        ),
    }