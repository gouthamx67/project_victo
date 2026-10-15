from __future__ import annotations

import pandas as pd
import pytest

import src.analyses

from src.analyses.core import calculate_case_count
from src.analyses.trends import calculate_monthly_case_counts
from src.llm_contract import MetricContext
from src.normalize import NormalizedData
from src.report_config import get_section, load_report_config
from src.section_context import (
    build_metric_context,
    build_section_context,
)


CONFIG_PATH = "configs/pader.yaml"


def make_data() -> NormalizedData:
    cases = pd.DataFrame(
        {
            "case_id": [
                "CASE-001",
                "CASE-002",
            ],
            "receivedate": [
                "2025-01-01",
                "2025-01-02",
            ],
            "serious": [
                True,
                False,
            ],
            "expedited": [
                True,
                False,
            ],
        }
    )

    reactions = pd.DataFrame(
        {
            "reaction_id": [
                "CASE-001:0:0",
                "CASE-002:0:0",
            ],
            "case_id": [
                "CASE-001",
                "CASE-002",
            ],
            "reaction_pt": [
                "Headache",
                "Nausea",
            ],
            "outcome": [
                "unknown",
                "recovered/resolved",
            ],
        }
    )

    return NormalizedData(
        cases=cases,
        reactions=reactions,
    )


def test_build_metric_context_preserves_analysis_result():
    data = make_data()

    result = calculate_case_count(data)

    context = build_metric_context(result)

    assert isinstance(context, MetricContext)
    assert context.metric_id == "case_count"
    assert context.value == 2
    assert context.population == "case"
    assert context.version == "1.0.0"


def test_section_context_contains_only_requested_metrics():
    config = load_report_config(CONFIG_PATH)
    section = get_section(
        config,
        "case_volume",
    )

    data = make_data()

    case_count = calculate_case_count(data)
    monthly_case_counts = calculate_monthly_case_counts(data)

    context = build_section_context(
        config,
        section,
        [
            case_count,
            monthly_case_counts,
        ],
    )

    assert [
        metric.metric_id
        for metric in context.metrics
    ] == [
        "case_count",
        "monthly_case_counts",
    ]


def test_section_context_rejects_missing_metric():
    config = load_report_config(CONFIG_PATH)
    section = get_section(
        config,
        "case_volume",
    )

    data = make_data()

    case_count = calculate_case_count(data)

    with pytest.raises(
        ValueError,
        match="missing required analyses",
    ):
        build_section_context(
            config,
            section,
            [case_count],
        )


def test_section_context_preserves_section_metadata():
    config = load_report_config(CONFIG_PATH)
    section = get_section(
        config,
        "seriousness",
    )

    data = make_data()

    from src.analyses.core import (
        calculate_non_serious_case_count,
        calculate_serious_case_count,
        calculate_serious_case_percentage,
    )

    results = [
        calculate_serious_case_count(data),
        calculate_non_serious_case_count(data),
        calculate_serious_case_percentage(data),
    ]

    context = build_section_context(
        config,
        section,
        results,
    )

    assert context.report_id == "pader"
    assert context.report_version == "1.0.0"
    assert context.section_id == "seriousness"
    assert context.section_title == "Seriousness"
    assert context.style == {
        "tone": "formal",
        "length": "concise",
    }


def test_section_context_preserves_metric_order_from_config():
    config = load_report_config(CONFIG_PATH)
    section = get_section(
        config,
        "seriousness",
    )

    data = make_data()

    from src.analyses.core import (
        calculate_non_serious_case_count,
        calculate_serious_case_count,
        calculate_serious_case_percentage,
    )

    # Deliberately pass results in reverse order.
    results = [
        calculate_serious_case_percentage(data),
        calculate_non_serious_case_count(data),
        calculate_serious_case_count(data),
    ]

    context = build_section_context(
        config,
        section,
        results,
    )

    assert [
        metric.metric_id
        for metric in context.metrics
    ] == [
        "serious_case_count",
        "non_serious_case_count",
        "serious_case_percentage",
    ]