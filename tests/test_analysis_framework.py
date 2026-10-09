from __future__ import annotations

import pytest

# IMPORTANT:
# Import built-in analyses before taking the registry snapshot.
# Their decorators populate the central analysis registry.
import src.analyses  # noqa: F401

from src.analysis_framework import (
    AnalysisResult,
    RegisteredAnalysis,
    clear_analysis_registry,
    get_registered_analysis,
    list_registered_analyses,
    register_analysis,
    run_all_analyses,
    run_analysis,
)
from src.normalize import NormalizedData


# Snapshot the built-in registrations after src.analyses has populated them.
BUILTIN_ANALYSES: tuple[RegisteredAnalysis, ...] = tuple(
    list_registered_analyses()
)


def restore_builtin_registry() -> None:
    """Restore the built-in analysis registrations."""
    clear_analysis_registry()

    for registered in BUILTIN_ANALYSES:
        register_analysis(
            metric_id=registered.metric_id,
            population=registered.population,
            version=registered.version,
        )(registered.function)


@pytest.fixture
def isolated_registry():
    """
    Give a test an empty registry and restore built-ins afterward.
    """
    clear_analysis_registry()

    yield

    restore_builtin_registry()


def make_empty_data() -> NormalizedData:
    import pandas as pd

    return NormalizedData(
        cases=pd.DataFrame(),
        reactions=pd.DataFrame(),
    )


def test_register_analysis_registers_function(isolated_registry):
    @register_analysis(
        metric_id="test_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_test_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="test_metric",
            value=1,
            definition="Test metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    registered = get_registered_analysis("test_metric")

    assert registered.metric_id == "test_metric"
    assert registered.population == "case"
    assert registered.version == "1.0.0"
    assert registered.function is calculate_test_metric


def test_duplicate_metric_id_is_rejected(isolated_registry):
    @register_analysis(
        metric_id="duplicate_metric",
        population="case",
        version="1.0.0",
    )
    def first_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="duplicate_metric",
            value=1,
            definition="First metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    with pytest.raises(ValueError, match="already registered"):

        @register_analysis(
            metric_id="duplicate_metric",
            population="case",
            version="1.0.0",
        )
        def second_metric(data: NormalizedData) -> AnalysisResult:
            return AnalysisResult(
                metric_id="duplicate_metric",
                value=2,
                definition="Second metric.",
                population="case",
                unit="cases",
                supporting_ids=[],
                version="1.0.0",
            )


def test_run_analysis_validates_registered_result(isolated_registry):
    @register_analysis(
        metric_id="valid_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_valid_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="valid_metric",
            value=42,
            definition="Valid metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    result = run_analysis(
        "valid_metric",
        make_empty_data(),
    )

    assert result.metric_id == "valid_metric"
    assert result.value == 42
    assert result.population == "case"
    assert result.unit == "cases"


def test_list_registered_analyses_is_sorted(isolated_registry):
    @register_analysis(
        metric_id="z_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_z_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="z_metric",
            value=1,
            definition="Z metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    @register_analysis(
        metric_id="a_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_a_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="a_metric",
            value=1,
            definition="A metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    assert [
        analysis.metric_id
        for analysis in list_registered_analyses()
    ] == [
        "a_metric",
        "z_metric",
    ]


def test_run_all_analyses_runs_registered_metrics_in_order(isolated_registry):
    @register_analysis(
        metric_id="b_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_b_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="b_metric",
            value=2,
            definition="B metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    @register_analysis(
        metric_id="a_metric",
        population="case",
        version="1.0.0",
    )
    def calculate_a_metric(data: NormalizedData) -> AnalysisResult:
        return AnalysisResult(
            metric_id="a_metric",
            value=1,
            definition="A metric.",
            population="case",
            unit="cases",
            supporting_ids=[],
            version="1.0.0",
        )

    results = run_all_analyses(make_empty_data())

    assert [result.metric_id for result in results] == [
        "a_metric",
        "b_metric",
    ]

    assert all(result.unit == "cases" for result in results)


def test_run_analysis_requires_declared_unit(isolated_registry):
    @register_analysis(
        metric_id="metric_without_unit",
        population="case",
        version="1.0.0",
    )
    def calculate_metric_without_unit(
        data: NormalizedData,
    ) -> AnalysisResult:
        return AnalysisResult(
            metric_id="metric_without_unit",
            value=1,
            definition="Test metric.",
            population="case",
            supporting_ids=[],
            version="1.0.0",
        )

    with pytest.raises(
        ValueError,
        match="did not declare a reporting unit",
    ):
        run_analysis(
            "metric_without_unit",
            make_empty_data(),
        )