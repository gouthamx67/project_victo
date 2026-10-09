from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ReviewStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    APPROVED = "APPROVED"
    FLAGGED = "FLAGGED"


class SectionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer: str | None = None
    comment: str | None = None
    reviewed_at: str | None = None


class ReviewSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[SectionReview]


def make_unreviewed_review(section_id: str) -> SectionReview:
    return SectionReview(
        section_id=section_id,
        status=ReviewStatus.UNREVIEWED,
    )


def approve_section(
    section_id: str,
    reviewer: str,
    comment: str | None = None,
) -> SectionReview:
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty.")

    return SectionReview(
        section_id=section_id,
        status=ReviewStatus.APPROVED,
        reviewer=reviewer,
        comment=comment,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )


def flag_section(
    section_id: str,
    reviewer: str,
    comment: str,
) -> SectionReview:
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty.")

    if not comment.strip():
        raise ValueError(
            "A comment is required when flagging a section."
        )

    return SectionReview(
        section_id=section_id,
        status=ReviewStatus.FLAGGED,
        reviewer=reviewer,
        comment=comment,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )


def validate_review_set(
    section_ids: list[str],
    review_set: ReviewSet,
) -> None:
    expected = set(section_ids)

    actual = {
        review.section_id
        for review in review_set.reviews
    }

    unknown = actual - expected

    if unknown:
        raise ValueError(
            "Review contains unknown section IDs: "
            + ", ".join(sorted(unknown))
        )

    missing = expected - actual

    if missing:
        raise ValueError(
            "Review is missing sections: "
            + ", ".join(sorted(missing))
        )

    if len(actual) != len(review_set.reviews):
        raise ValueError(
            "Each section must have exactly one review record."
        )


def all_sections_approved(
    section_ids: list[str],
    review_set: ReviewSet,
) -> bool:
    validate_review_set(
        section_ids=section_ids,
        review_set=review_set,
    )

    return all(
        review.status == ReviewStatus.APPROVED
        for review in review_set.reviews
    )


def get_review(
    review_set: ReviewSet,
    section_id: str,
) -> SectionReview:
    matches = [
        review
        for review in review_set.reviews
        if review.section_id == section_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one review for section "
            f"{section_id!r}, found {len(matches)}."
        )

    return matches[0]