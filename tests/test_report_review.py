from pathlib import Path

import pytest

from src.report_config import load_report_config
from src.report_review import (
    ReportFinalizationError,
    assert_report_can_be_finalized,
    can_finalize_report,
    configured_section_ids,
    load_review_set,
    make_initial_review_set,
    save_review_set,
)
from src.review import (
    ReviewSet,
    approve_section,
    flag_section,
)


CONFIG = Path("configs/pader.yaml")


@pytest.fixture
def config():
    return load_report_config(CONFIG)


def test_configured_section_ids_follow_report_order(config):
    section_ids = configured_section_ids(config)

    assert section_ids == [
        "executive_summary",
        "case_volume",
        "seriousness",
        "expedited_cases",
        "reaction_profile",
        "outcomes",
        "trend_review",
        "case_listing",
    ]


def test_initial_review_set_contains_every_section_as_unreviewed(config):
    review_set = make_initial_review_set(config)

    assert [
        review.section_id
        for review in review_set.reviews
    ] == configured_section_ids(config)

    assert all(
        review.status.value == "UNREVIEWED"
        for review in review_set.reviews
    )


def test_initial_review_set_cannot_finalize(config):
    review_set = make_initial_review_set(config)

    assert not can_finalize_report(
        config=config,
        review_set=review_set,
    )


def test_flagged_section_cannot_finalize(config):
    review_set = make_initial_review_set(config)

    review_set.reviews[0] = flag_section(
        section_id="executive_summary",
        reviewer="reviewer-1",
        comment="Please verify the wording.",
    )

    assert not can_finalize_report(
        config=config,
        review_set=review_set,
    )

    with pytest.raises(
        ReportFinalizationError,
        match="must be APPROVED",
    ):
        assert_report_can_be_finalized(
            config=config,
            review_set=review_set,
        )


def test_all_approved_sections_can_finalize(config):
    section_ids = configured_section_ids(config)

    review_set = ReviewSet(
        reviews=[
            approve_section(
                section_id=section_id,
                reviewer="reviewer-1",
                comment="Reviewed.",
            )
            for section_id in section_ids
        ]
    )

    assert can_finalize_report(
        config=config,
        review_set=review_set,
    )

    assert_report_can_be_finalized(
        config=config,
        review_set=review_set,
    )


def test_review_set_round_trip_json(config, tmp_path):
    review_set = make_initial_review_set(config)

    review_set.reviews[0] = approve_section(
        section_id="executive_summary",
        reviewer="reviewer-1",
        comment="Reviewed.",
    )

    output_path = tmp_path / "review.json"

    save_review_set(
        review_set=review_set,
        output_path=output_path,
    )

    loaded = load_review_set(output_path)

    assert loaded == review_set