from pathlib import Path

import pytest

from src.report_pipeline import (
    ReportStatus,
    render_html_report,
    run_report_pipeline,
)
from src.report_review import (
    ReportFinalizationError,
    finalize_report,
)
from src.review import (
    ReviewSet,
    approve_section,
    flag_section,
)


DATASET = Path("tests/fixtures/synthetic_icsr.csv")

CONFIG = Path(
    "configs/pader.yaml"
)

PROMPT = Path(
    "prompts/sections/default.txt"
)

TEMPLATE = Path(
    "prompts/report_template.html"
)


def _run_draft():
    return run_report_pipeline(
        csv_path=DATASET,
        config_path=CONFIG,
        prompt_path=PROMPT,
        use_llm=False,
    )


def _approve_all(result):
    return ReviewSet(
        reviews=[
            approve_section(
                section_id=review.section_id,
                reviewer="reviewer-1",
                comment="Reviewed and approved.",
            )
            for review in result.review_set.reviews
        ]
    )


def test_unreviewed_report_cannot_be_finalized():
    result = _run_draft()

    with pytest.raises(
        ReportFinalizationError,
        match="must be APPROVED",
    ):
        finalize_report(result)


def test_flagged_report_cannot_be_finalized():
    result = _run_draft()

    reviews = []

    for review in result.review_set.reviews:
        if review.section_id == "executive_summary":
            reviews.append(
                flag_section(
                    section_id=review.section_id,
                    reviewer="reviewer-1",
                    comment="Please verify this section.",
                )
            )
        else:
            reviews.append(
                approve_section(
                    section_id=review.section_id,
                    reviewer="reviewer-1",
                    comment="Reviewed.",
                )
            )

    review_set = ReviewSet(reviews=reviews)

    with pytest.raises(
        ReportFinalizationError,
        match="must be APPROVED",
    ):
        finalize_report(
            result,
            review_set=review_set,
        )


def test_all_approved_report_becomes_final():
    result = _run_draft()

    approved_reviews = _approve_all(result)

    final_result = finalize_report(
        result,
        review_set=approved_reviews,
    )

    assert result.status == ReportStatus.DRAFT
    assert final_result.status == ReportStatus.FINAL
    assert final_result.review_set == approved_reviews


def test_finalization_does_not_mutate_original_result():
    result = _run_draft()

    approved_reviews = _approve_all(result)

    final_result = finalize_report(
        result,
        review_set=approved_reviews,
    )

    assert result.status == ReportStatus.DRAFT
    assert final_result.status == ReportStatus.FINAL
    assert result is not final_result


def test_draft_html_shows_review_required():
    result = _run_draft()

    html = render_html_report(
        result=result,
        template_path=TEMPLATE,
    )

    assert "DRAFT — HUMAN REVIEW REQUIRED" in html
    assert "FINAL — HUMAN REVIEW APPROVED" not in html
    assert 'content="DRAFT"' in html


def test_final_html_shows_approval():
    result = _run_draft()

    final_result = finalize_report(
        result,
        review_set=_approve_all(result),
    )

    html = render_html_report(
        result=final_result,
        template_path=TEMPLATE,
    )

    assert "FINAL — HUMAN REVIEW APPROVED" in html
    assert "DRAFT — HUMAN REVIEW REQUIRED" not in html
    assert 'content="FINAL"' in html


def test_finalization_preserves_report_content():
    result = _run_draft()

    final_result = finalize_report(
        result,
        review_set=_approve_all(result),
    )

    assert final_result.config == result.config
    assert final_result.normalized.cases.equals(
        result.normalized.cases
    )
    assert final_result.normalized.reactions.equals(
        result.normalized.reactions
    )
    assert final_result.analyses == result.analyses
    assert final_result.sections == result.sections