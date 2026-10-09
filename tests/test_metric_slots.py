from __future__ import annotations

import pytest

from src.analysis_framework import AnalysisResult
from src.metric_slots import (
    MetricSlotError,
    build_metric_map,
    find_metric_slots,
    resolve_metric_slots,
    validate_metric_slots,
)


def make_result(
    metric_id: str,
    value,
    population: str = "case",
) -> AnalysisResult:
    return AnalysisResult(
        metric_id=metric_id,
        value=value,
        definition=f"Definition of {metric_id}.",
        population=population,
        supporting_ids=[],
        version="1.0.0",
    )


def test_find_metric_slots():
    text = (
        "There were {case_count} cases and "
        "{serious_case_percentage}% were serious."
    )

    assert find_metric_slots(text) == [
        "case_count",
        "serious_case_percentage",
    ]


def test_find_metric_slots_preserves_duplicates():
    text = "{case_count} cases were observed; {case_count} in total."

    assert find_metric_slots(text) == [
        "case_count",
        "case_count",
    ]


def test_build_metric_map():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_count", 1023),
    ]

    metric_map = build_metric_map(results)

    assert metric_map["case_count"].value == 1024
    assert metric_map["serious_case_count"].value == 1023


def test_duplicate_metric_ids_are_rejected():
    results = [
        make_result("case_count", 1024),
        make_result("case_count", 999),
    ]

    with pytest.raises(
        MetricSlotError,
        match="Duplicate metric_id",
    ):
        build_metric_map(results)


def test_validate_metric_slots_accepts_approved_metrics():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_percentage", 99.9),
    ]

    validate_metric_slots(
        "There were {case_count} cases and {serious_case_percentage}% were serious.",
        results,
    )


def test_unknown_metric_slot_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    with pytest.raises(
        MetricSlotError,
        match="unknown metric slots",
    ):
        validate_metric_slots(
            "There were {case_count} cases and {invented_count} events.",
            results,
        )


def test_resolve_integer_metric():
    results = [
        make_result("case_count", 1024),
    ]

    rendered = resolve_metric_slots(
        "The analysis identified {case_count} cases.",
        results,
    )

    assert rendered == "The analysis identified 1024 cases."


def test_resolve_float_metric_without_recalculation():
    results = [
        make_result("serious_case_percentage", 99.9),
    ]

    rendered = resolve_metric_slots(
        "{serious_case_percentage}% of cases were serious.",
        results,
    )

    assert rendered == "99.9% of cases were serious."


def test_resolve_dictionary_metric_deterministically():
    results = [
        make_result(
            "reaction_outcome_distribution",
            {
                "fatal": 3,
                "recovered": 20,
            },
            population="reaction",
        )
    ]

    rendered = resolve_metric_slots(
        "Outcomes: {reaction_outcome_distribution}.",
        results,
    )

    assert rendered == (
        "Outcomes: fatal: 3; recovered: 20."
    )


def test_resolve_list_metric_deterministically():
    results = [
        make_result(
            "example_list",
            ["a", "b", "c"],
        )
    ]

    rendered = resolve_metric_slots(
        "Items: {example_list}.",
        results,
    )

    assert rendered == "Items: a, b, c."


def test_no_slots_leaves_text_unchanged():
    results = [
        make_result("case_count", 1024),
    ]

    text = "No metric reference was required."

    assert resolve_metric_slots(text, results) == text


def test_slot_replacement_does_not_execute_expressions():
    results = [
        make_result("case_count", 1024),
    ]

    text = "{case_count + 1}"

    with pytest.raises(MetricSlotError):
        resolve_metric_slots(text, results)


def test_unknown_slot_cannot_inject_a_number():
    results = [
        make_result("case_count", 1024),
    ]

    malicious = (
        "There were {case_count} cases and "
        "{fake_metric_999999} additional cases."
    )

    with pytest.raises(MetricSlotError):
        resolve_metric_slots(malicious, results)