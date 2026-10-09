import pytest

from src.review import (
    ReviewSet,
    ReviewStatus,
    SectionReview,
    all_sections_approved,
    approve_section,
    flag_section,
    get_review,
    make_unreviewed_review,
    validate_review_set,
)


def test_new_section_is_unreviewed() -> None:
    review = make_unreviewed_review("executive_summary")

    assert review.section_id == "executive_summary"
    assert review.status == ReviewStatus.UNREVIEWED
    assert review.reviewer is None
    assert review.reviewed_at is None


def test_approve_section_records_reviewer() -> None:
    review = approve_section(
        section_id="executive_summary",
        reviewer="reviewer-1",
        comment="Checked against approved metrics.",
    )

    assert review.status == ReviewStatus.APPROVED
    assert review.reviewer == "reviewer-1"
    assert review.comment == "Checked against approved metrics."
    assert review.reviewed_at is not None


def test_flag_requires_comment() -> None:
    with pytest.raises(ValueError, match="comment is required"):
        flag_section(
            section_id="trend_review",
            reviewer="reviewer-1",
            comment="   ",
        )


def test_flag_section_records_comment() -> None:
    review = flag_section(
        section_id="trend_review",
        reviewer="reviewer-1",
        comment="Please verify the deviation wording.",
    )

    assert review.status == ReviewStatus.FLAGGED
    assert review.reviewer == "reviewer-1"
    assert review.comment == "Please verify the deviation wording."
    assert review.reviewed_at is not None


def test_empty_reviewer_is_rejected() -> None:
    with pytest.raises(ValueError, match="reviewer"):
        approve_section(
            section_id="case_volume",
            reviewer="   ",
        )


def test_review_set_requires_every_section() -> None:
    review_set = ReviewSet(
        reviews=[
            make_unreviewed_review("executive_summary"),
        ]
    )

    with pytest.raises(ValueError, match="missing sections"):
        validate_review_set(
            section_ids=[
                "executive_summary",
                "case_volume",
            ],
            review_set=review_set,
        )


def test_review_set_rejects_unknown_section() -> None:
    review_set = ReviewSet(
        reviews=[
            make_unreviewed_review("unknown_section"),
        ]
    )

    with pytest.raises(ValueError, match="unknown section"):
        validate_review_set(
            section_ids=["executive_summary"],
            review_set=review_set,
        )


def test_duplicate_section_reviews_are_rejected() -> None:
    review_set = ReviewSet(
        reviews=[
            make_unreviewed_review("executive_summary"),
            make_unreviewed_review("executive_summary"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="exactly one review record",
    ):
        validate_review_set(
            section_ids=["executive_summary"],
            review_set=review_set,
        )


def test_all_sections_approved() -> None:
    review_set = ReviewSet(
        reviews=[
            approve_section(
                section_id="executive_summary",
                reviewer="reviewer-1",
            ),
            approve_section(
                section_id="case_volume",
                reviewer="reviewer-1",
            ),
        ]
    )

    assert all_sections_approved(
        section_ids=[
            "executive_summary",
            "case_volume",
        ],
        review_set=review_set,
    )


def test_flagged_section_prevents_full_approval() -> None:
    review_set = ReviewSet(
        reviews=[
            approve_section(
                section_id="executive_summary",
                reviewer="reviewer-1",
            ),
            flag_section(
                section_id="case_volume",
                reviewer="reviewer-1",
                comment="Needs review.",
            ),
        ]
    )

    assert not all_sections_approved(
        section_ids=[
            "executive_summary",
            "case_volume",
        ],
        review_set=review_set,
    )


def test_unreviewed_section_prevents_full_approval() -> None:
    review_set = ReviewSet(
        reviews=[
            approve_section(
                section_id="executive_summary",
                reviewer="reviewer-1",
            ),
            make_unreviewed_review("case_volume"),
        ]
    )

    assert not all_sections_approved(
        section_ids=[
            "executive_summary",
            "case_volume",
        ],
        review_set=review_set,
    )


def test_get_review_returns_requested_section() -> None:
    review_set = ReviewSet(
        reviews=[
            approve_section(
                section_id="executive_summary",
                reviewer="reviewer-1",
            ),
            make_unreviewed_review("case_volume"),
        ]
    )

    review = get_review(
        review_set,
        "case_volume",
    )

    assert review.section_id == "case_volume"
    assert review.status == ReviewStatus.UNREVIEWED


def test_get_review_rejects_missing_section() -> None:
    review_set = ReviewSet(reviews=[])

    with pytest.raises(ValueError, match="found 0"):
        get_review(
            review_set,
            "case_volume",
        )