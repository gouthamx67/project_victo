from pathlib import Path

from src.analysis_framework import AnalysisResult
from src.llm_contract import LLMGenerationConfig
from src.run_manifest import (
    MANIFEST_VERSION,
    build_run_manifest,
    canonical_json_hash,
    sha256_file,
    write_manifest,
)


def make_result() -> AnalysisResult:
    return AnalysisResult(
        metric_id="case_count",
        value=1024,
        definition="Number of unique cases.",
        population="case",
        supporting_ids=["CASE-1", "CASE-2"],
        version="1.0.0",
    )


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("hello", encoding="utf-8")

    first = sha256_file(path)
    second = sha256_file(path)

    assert first == second
    assert len(first) == 64


def test_canonical_json_hash_is_order_independent() -> None:
    first = canonical_json_hash(
        {
            "b": 2,
            "a": 1,
        }
    )

    second = canonical_json_hash(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert first == second


def test_build_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    config = tmp_path / "config.yaml"
    prompt = tmp_path / "prompt.txt"

    dataset.write_text("a,b\n1,2\n", encoding="utf-8")
    config.write_text("report: test\n", encoding="utf-8")
    prompt.write_text("Use approved metrics.", encoding="utf-8")

    generation_config = LLMGenerationConfig(
        provider="mock",
        model="mock-model",
        temperature=0.0,
        max_tokens=1000,
    )

    manifest = build_run_manifest(
        run_id="run-001",
        dataset_path=dataset,
        report_config_path=config,
        prompt_path=prompt,
        generation_config=generation_config,
        report_id="pader",
        report_version="1.0.0",
        section_metric_ids={
            "executive_summary": [
                "case_count",
            ],
        },
        analyses=[make_result()],
    )

    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.run_id == "run-001"
    assert len(manifest.dataset_sha256) == 64
    assert len(manifest.report_config_sha256) == 64
    assert len(manifest.prompt_sha256) == 64
    assert manifest.provider == "mock"
    assert manifest.model == "mock-model"
    assert manifest.temperature == 0.0
    assert manifest.analysis_registry_version
    assert manifest.analysis_versions == {
        "case_count": "1.0.0",
    }


def test_manifest_changes_when_dataset_changes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("a,b\n1,2\n", encoding="utf-8")

    first = sha256_file(dataset)

    dataset.write_text("a,b\n1,3\n", encoding="utf-8")

    second = sha256_file(dataset)

    assert first != second


def test_write_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    config = tmp_path / "config.yaml"
    prompt = tmp_path / "prompt.txt"

    dataset.write_text("data", encoding="utf-8")
    config.write_text("config", encoding="utf-8")
    prompt.write_text("prompt", encoding="utf-8")

    manifest = build_run_manifest(
        run_id="run-002",
        dataset_path=dataset,
        report_config_path=config,
        prompt_path=prompt,
        generation_config=LLMGenerationConfig(),
        report_id="pader",
        report_version="1.0.0",
        section_metric_ids={},
        analyses=[],
    )

    output = tmp_path / "nested" / "manifest.json"

    written = write_manifest(
        manifest,
        output,
    )

    assert written == output
    assert output.exists()

    text = output.read_text(
        encoding="utf-8"
    )

    assert '"run_id": "run-002"' in text