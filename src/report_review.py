from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.report_config import ReportConfig
from src.review import (
    ReviewSet,
    all_sections_approved,
    make_unreviewed_review,
    validate_review_set,
)


class ReportFinalizationError(ValueError):
    """Raised when a report cannot be finalized."""


def configured_section_ids(config: ReportConfig) -> list[str]:
    sections = config.report.sections
    return [
        section.id
        for section in sorted(sections, key=lambda section: section.order)
    ]


def make_initial_review_set(config: ReportConfig | list[str]) -> ReviewSet:
    if isinstance(config, ReportConfig):
        section_ids = configured_section_ids(config)
    else:
        section_ids = list(config)

    if len(section_ids) != len(set(section_ids)):
        raise ValueError("Section IDs must be unique.")

    return ReviewSet(
        reviews=[
            make_unreviewed_review(section_id)
            for section_id in section_ids
        ]
    )


def validate_report_review(
    result=None,
    review_set: ReviewSet | None = None,
    *,
    config: ReportConfig | None = None,
) -> None:
    if config is None:
        if result is None:
            raise TypeError("Either result or config must be provided.")
        config = result.config

    if review_set is None:
        if result is None:
            raise TypeError("review_set is required.")
        review_set = result.review_set

    section_ids = configured_section_ids(config)

    validate_review_set(
        section_ids=section_ids,
        review_set=review_set,
    )


def can_finalize_report(
    result=None,
    review_set: ReviewSet | None = None,
    *,
    config: ReportConfig | None = None,
) -> bool:
    if config is None:
        if result is None:
            raise TypeError("Either result or config must be provided.")
        config = result.config

    if review_set is None:
        if result is None:
            raise TypeError("review_set is required.")
        review_set = result.review_set

    section_ids = configured_section_ids(config)

    try:
        validate_review_set(
            section_ids=section_ids,
            review_set=review_set,
        )
    except ValueError:
        return False

    return all_sections_approved(
        section_ids=section_ids,
        review_set=review_set,
    )


def assert_report_can_be_finalized(
    result=None,
    review_set: ReviewSet | None = None,
    *,
    config: ReportConfig | None = None,
) -> None:
    if config is None:
        if result is None:
            raise TypeError("Either result or config must be provided.")
        config = result.config

    if review_set is None:
        if result is None:
            raise TypeError("review_set is required.")
        review_set = result.review_set

    if not can_finalize_report(
        config=config,
        review_set=review_set,
    ):
        raise ReportFinalizationError(
            "Every configured section must be APPROVED before "
            "the report can be finalized."
        )


def finalize_report(
    result,
    review_set: ReviewSet | None = None,
):
    candidate = review_set if review_set is not None else result.review_set

    assert_report_can_be_finalized(
        config=result.config,
        review_set=candidate,
    )

    from src.report_pipeline import ReportStatus

    return replace(
        result,
        review_set=candidate,
        status=ReportStatus.FINAL,
    )


def save_review_set(
    review_set: ReviewSet,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = review_set.model_dump(
        mode="json",
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def load_review_set(
    input_path: str | Path,
) -> ReviewSet:
    input_path = Path(input_path)

    payload = json.loads(
        input_path.read_text(
            encoding="utf-8",
        )
    )

    return ReviewSet.model_validate(payload)