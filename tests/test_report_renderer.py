from __future__ import annotations

import pytest

from src.analysis_framework import AnalysisResult
from src.metric_slots import MetricSlotError
from src.report_renderer import (
    render_narrative,
    render_narrative_with_evidence,
    render_section,
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
        unit=(
            "case_percentage"
            if metric_id.endswith("_percentage")
            else "cases"
        ),
    )


def test_render_narrative_resolves_only_approved_metrics():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_percentage", 99.9),
    ]

    rendered = render_narrative(
        "There were {case_count} cases. "
        "{serious_case_percentage} were serious.",
        results,
    )

    assert rendered == (
        "There were 1,024 cases. "
        "99.9% were serious."
    )


def test_render_section_matches_narrative_renderer():
    results = [
        make_result("case_count", 1024),
    ]

    narrative = "The report contains {case_count} cases."

    assert render_section(narrative, results) == render_narrative(
        narrative,
        results,
    )


def test_unknown_metric_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    with pytest.raises(MetricSlotError):
        render_narrative(
            "The report contains {case_count} cases and "
            "{unsupported_metric} other events.",
            results,
        )


def test_empty_narrative_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    with pytest.raises(ValueError, match="must not be empty"):
        render_narrative("", results)


def test_whitespace_only_narrative_is_rejected():
    results = [
        make_result("case_count", 1024),
    ]

    with pytest.raises(ValueError, match="must not be empty"):
        render_narrative("   ", results)


def test_renderer_does_not_change_non_slot_text():
    results = [
        make_result("case_count", 1024),
    ]

    narrative = (
        "The reporting period was reviewed. "
        "No causal conclusion is made here."
    )

    assert render_narrative(narrative, results) == narrative


def test_multiple_occurrences_are_all_resolved():
    results = [
        make_result("case_count", 1024),
    ]

    rendered = render_narrative(
        "The dataset contained {case_count} cases. "
        "The denominator was {case_count} cases.",
        results,
    )

    assert rendered == (
        "The dataset contained 1,024 cases. "
        "The denominator was 1,024 cases."
    )


def test_percentage_is_formatted_to_one_decimal_place():
    results = [
        make_result("serious_case_percentage", 99.987),
    ]

    rendered = render_narrative(
        "Serious cases represented {serious_case_percentage} of all cases.",
        results,
    )

    assert rendered == (
        "Serious cases represented 100.0% of all cases."
    )


def test_integer_counts_use_thousands_separators():
    results = [
        make_result("case_count", 10234),
    ]

    rendered = render_narrative(
        "The analysis included {case_count} cases.",
        results,
    )

    assert rendered == (
        "The analysis included 10,234 cases."
    )


def test_structured_top_reactions_keep_field_labels():
    results = [
        make_result(
            "top_reactions",
            [
                {
                    "reaction_pt": "Headache",
                    "occurrences": 25,
                    "distinct_cases": 20,
                }
            ],
            population="reaction",
        )
    ]

    rendered = render_narrative(
        "Top reaction evidence: {top_reactions}.",
        results,
    )

    assert rendered == (
        "Top reaction evidence: "
        "Reaction pt: Headache; "
        "Occurrences: 25 reaction occurrences; "
        "Distinct cases: 20 cases."
    )


def test_structured_outcome_counts_keep_field_labels():
    results = [
        make_result(
            "case_outcome_distribution",
            [
                {
                    "case_outcome": "fatal",
                    "count": 68,
                }
            ],
        )
    ]

    rendered = render_narrative(
        "Outcome distribution: {case_outcome_distribution}.",
        results,
    )

    assert rendered == (
        "Outcome distribution: "
        "Case outcome: fatal; "
        "Count: 68 cases."
    )


def test_evidence_renderer_formats_counts_and_percentages():
    results = [
        make_result("case_count", 1024),
        make_result("serious_case_percentage", 99.9),
    ]

    rendered = render_narrative_with_evidence(
        "{case_count} cases; {serious_case_percentage} serious.",
        results,
    )

    assert "1,024" in rendered
    assert "99.9%" in rendered
    assert 'data-metric-id="case_count"' in rendered
    assert 'data-metric-id="serious_case_percentage"' in rendered
    assert "#evidence-case_count" in rendered
    assert "#evidence-serious_case_percentage" in rendered


# ---------------------------------------------------------------------------
# Unit-label rendering
# ---------------------------------------------------------------------------


def test_case_outcome_distribution_labels_count_as_cases():
    results = [
        make_result(
            "case_outcome_distribution",
            [
                {
                    "case_outcome": "fatal",
                    "count": 68,
                }
            ],
        )
    ]

    rendered = render_narrative(
        "{case_outcome_distribution}",
        results,
    )

    assert rendered == (
        "Case outcome: fatal; Count: 68 cases"
    )


def test_reaction_outcome_distribution_labels_count_as_reaction_occurrences():
    results = [
        make_result(
            "reaction_outcome_distribution",
            [
                {
                    "outcome_category": "fatal",
                    "count": 137,
                }
            ],
            population="reaction",
        )
    ]

    rendered = render_narrative(
        "{reaction_outcome_distribution}",
        results,
    )

    assert rendered == (
        "Outcome category: fatal; "
        "Count: 137 reaction occurrences"
    )


def test_top_reactions_label_occurrences_and_distinct_cases():
    results = [
        make_result(
            "top_reactions",
            [
                {
                    "reaction_pt": "Headache",
                    "occurrences": 25,
                    "distinct_cases": 20,
                }
            ],
            population="reaction",
        )
    ]

    rendered = render_narrative(
        "{top_reactions}",
        results,
    )

    assert rendered == (
        "Reaction pt: Headache; "
        "Occurrences: 25 reaction occurrences; "
        "Distinct cases: 20 cases"
    )


def test_monthly_case_count_is_explicitly_labeled_as_cases():
    results = [
        make_result(
            "monthly_case_counts",
            [
                {
                    "reporting_month": "2025-01",
                    "case_count": 75,
                }
            ],
        )
    ]

    rendered = render_narrative(
        "{monthly_case_counts}",
        results,
    )

    assert rendered == "2025-01: 75 cases"