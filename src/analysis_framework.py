from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.normalize import NormalizedData


# ---------------------------------------------------------------------------
# Framework version
# ---------------------------------------------------------------------------

ANALYSIS_REGISTRY_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Population = Literal["case", "reaction"]

AnalysisUnit = Literal[
    "cases",
    "case_percentage",
    "reaction_occurrences",
    "reaction_distinct_terms",
    "distinct_case_reaction_pairs",
    "case_outcome_distribution",
    "reaction_outcome_distribution",
    "monthly_case_counts",
    "monthly_reaction_occurrences",
    "monthly_case_deviation",
    "case_listing",
]


class AnalysisResult(BaseModel):
    """
    Standard output contract for every deterministic safety analysis.

    The result is intentionally independent of any LLM or report type.

    ``population`` identifies the underlying analysis population, while
    ``unit`` identifies what the metric's primary numeric/listing value
    represents.
    """

    model_config = ConfigDict(extra="forbid")

    metric_id: str = Field(
        min_length=1,
        description="Stable identifier for the analysis metric.",
    )

    value: object = Field(
        description="Deterministically calculated metric value.",
    )

    definition: str = Field(
        min_length=1,
        description="Human-readable definition of what the metric measures.",
    )

    population: Population = Field(
        description="Unit of analysis: case or reaction.",
    )

    unit: AnalysisUnit | None = Field(
        default=None,
        description=(
            "Primary reporting unit of the metric. "
            "Production analyses must declare this explicitly."
        ),
    )

    supporting_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs supporting the result. "
            "Case analyses use case_id; reaction analyses may use "
            "reaction_id or distinct case_id depending on the metric."
        ),
    )

    version: str = Field(
        min_length=1,
        description="Version of the analysis implementation.",
    )


AnalysisFunction = Callable[[NormalizedData], AnalysisResult]


@dataclass(frozen=True)
class RegisteredAnalysis:
    """
    Metadata describing one registered analysis function.
    """

    metric_id: str
    population: Population
    version: str
    function: AnalysisFunction


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ANALYSIS_REGISTRY: dict[str, RegisteredAnalysis] = {}


def register_analysis(
    *,
    metric_id: str,
    population: Population,
    version: str = ANALYSIS_REGISTRY_VERSION,
) -> Callable[[AnalysisFunction], AnalysisFunction]:
    """
    Decorator used to register a deterministic analysis.

    Example:

        @register_analysis(
            metric_id="case_count",
            population="case",
            version="1.0.0",
        )
        def calculate_case_count(data: NormalizedData) -> AnalysisResult:
            ...

    Registration is explicit and keyed by metric_id.
    Duplicate metric IDs are rejected to avoid silently replacing analyses.
    """

    if not metric_id.strip():
        raise ValueError("metric_id must not be empty.")

    if not version.strip():
        raise ValueError("version must not be empty.")

    def decorator(function: AnalysisFunction) -> AnalysisFunction:
        if metric_id in _ANALYSIS_REGISTRY:
            existing = _ANALYSIS_REGISTRY[metric_id]

            raise ValueError(
                "Analysis metric_id already registered: "
                f"{metric_id!r}. Existing version: {existing.version!r}."
            )

        _ANALYSIS_REGISTRY[metric_id] = RegisteredAnalysis(
            metric_id=metric_id,
            population=population,
            version=version,
            function=function,
        )

        return function

    return decorator


def get_registered_analysis(metric_id: str) -> RegisteredAnalysis:
    """
    Return one registered analysis.

    Raises:
        KeyError: if metric_id is not registered.
    """

    try:
        return _ANALYSIS_REGISTRY[metric_id]
    except KeyError as exc:
        raise KeyError(
            f"Analysis metric_id is not registered: {metric_id!r}"
        ) from exc


def list_registered_analyses() -> list[RegisteredAnalysis]:
    """
    Return all registered analyses in stable metric_id order.
    """

    return [
        _ANALYSIS_REGISTRY[metric_id]
        for metric_id in sorted(_ANALYSIS_REGISTRY)
    ]


def clear_analysis_registry() -> None:
    """
    Clear the registry.

    This exists primarily for isolated tests.

    Production code should normally rely on module import-time registration.
    """

    _ANALYSIS_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_analysis(
    metric_id: str,
    data: NormalizedData,
) -> AnalysisResult:
    """
    Execute one registered analysis.

    The framework validates that the returned result agrees with the
    registered metric metadata and that the production analysis declares
    a reporting unit.
    """

    registered = get_registered_analysis(metric_id)

    result = registered.function(data)

    if not isinstance(result, AnalysisResult):
        raise TypeError(
            f"Analysis {metric_id!r} returned "
            f"{type(result).__name__}, expected AnalysisResult."
        )

    if result.metric_id != registered.metric_id:
        raise ValueError(
            f"Analysis {metric_id!r} returned metric_id "
            f"{result.metric_id!r}."
        )

    if result.population != registered.population:
        raise ValueError(
            f"Analysis {metric_id!r} returned population "
            f"{result.population!r}; expected "
            f"{registered.population!r}."
        )

    if result.version != registered.version:
        raise ValueError(
            f"Analysis {metric_id!r} returned version "
            f"{result.version!r}; expected "
            f"{registered.version!r}."
        )

    if result.unit is None:
        raise ValueError(
            f"Analysis {metric_id!r} did not declare a reporting unit."
        )

    return result

def run_all_analyses(
    data: NormalizedData,
) -> list[AnalysisResult]:
    """
    Execute every registered analysis in stable metric_id order.
    """

    return [
        run_analysis(registered.metric_id, data)
        for registered in list_registered_analyses()
    ]