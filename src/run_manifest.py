from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.analysis_framework import ANALYSIS_REGISTRY_VERSION, AnalysisResult
from src.llm_contract import LLMGenerationConfig


MANIFEST_VERSION = "1.0.0"


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = MANIFEST_VERSION

    run_id: str = Field(min_length=1)
    created_at_utc: str = Field(min_length=1)

    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(min_length=64, max_length=64)

    report_config_path: str = Field(min_length=1)
    report_config_sha256: str = Field(min_length=64, max_length=64)

    prompt_path: str = Field(min_length=1)
    prompt_sha256: str = Field(min_length=64, max_length=64)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float
    max_tokens: int

    analysis_registry_version: str = Field(min_length=1)
    analysis_versions: dict[str, str]

    report_id: str = Field(min_length=1)
    report_version: str = Field(min_length=1)

    section_metric_ids: dict[str, list[str]]


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot hash missing file: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def build_run_manifest(
    *,
    run_id: str,
    dataset_path: str | Path,
    report_config_path: str | Path,
    prompt_path: str | Path,
    generation_config: LLMGenerationConfig,
    report_id: str,
    report_version: str,
    section_metric_ids: dict[str, list[str]],
    analyses: list[AnalysisResult],
) -> RunManifest:
    analysis_versions = {
        result.metric_id: result.version
        for result in analyses
    }

    return RunManifest(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_path=str(dataset_path),
        dataset_sha256=sha256_file(dataset_path),
        report_config_path=str(report_config_path),
        report_config_sha256=sha256_file(report_config_path),
        prompt_path=str(prompt_path),
        prompt_sha256=sha256_file(prompt_path),
        provider=generation_config.provider,
        model=generation_config.model,
        temperature=generation_config.temperature,
        max_tokens=generation_config.max_tokens,
        analysis_registry_version=ANALYSIS_REGISTRY_VERSION,
        analysis_versions=analysis_versions,
        report_id=report_id,
        report_version=report_version,
        section_metric_ids={
            section_id: list(metric_ids)
            for section_id, metric_ids in section_metric_ids.items()
        },
    )


def write_manifest(
    manifest: RunManifest,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            manifest.model_dump(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return output_path