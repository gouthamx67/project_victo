"""
Generate the example PADER-style report shipped with this repository.

This runs the same pipeline the API uses:

    CSV -> ingest/validate -> normalize -> registered analyses
        -> grounded narratives (optional LLM) -> review gate -> finalize

Every configured section is approved through the same review functions
the API exposes, then the report is finalized and rendered to HTML.

Usage:
    python -m scripts.generate_example_report
    python -m scripts.generate_example_report --provider groq
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.env_loader import load_dotenv_file
from src.llm_provider import build_provider
from src.report_pipeline import (
    run_report_pipeline,
    write_html_report,
)
from src.report_review import (
    finalize_report,
    make_initial_review_set,
)
from src.review import (
    ReviewSet,
    approve_section,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "Bisoprolol_icsr_sample_1068rows.csv"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "psur_lite.yaml"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "sections" / "default.txt"
DEFAULT_TEMPLATE = PROJECT_ROOT / "prompts" / "report_template.html"
DEFAULT_OUTPUT = PROJECT_ROOT / "report_output.html"

DEFAULT_REVIEW_COMMENT = (
    "Example submission artifact approved through the review gate."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate, review, and finalize the example report."
        )
    )

    parser.add_argument(
        "--csv",
        default=str(DEFAULT_DATASET),
        help="Path to the safety CSV dataset.",
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the report YAML configuration.",
    )

    parser.add_argument(
        "--provider",
        choices=["none", "mock", "groq", "openai"],
        default="none",
        help=(
            "'none' (default) uses deterministic templates; "
            "'groq'/'openai' enable LLM-authored narratives; "
            "'mock' is an offline test double."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model id for the selected provider.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help=(
            "Grounding retries per LLM section before the "
            "deterministic fallback is used."
        ),
    )

    parser.add_argument(
        "--reviewer",
        default="Example Reviewer",
        help="Reviewer name recorded on each approved section.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the finalized HTML report.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv_file(PROJECT_ROOT / ".env")

    csv_path = Path(args.csv)

    if not csv_path.is_file():
        print(
            "Dataset not found: "
            f"{csv_path}\n"
            "Place the exercise CSV at data/ or pass --csv."
        )
        return 1

    provider = build_provider(
        provider=args.provider,
        model=args.model,
    )

    use_llm = provider is not None

    result = run_report_pipeline(
        csv_path=csv_path,
        config_path=Path(args.config),
        prompt_path=DEFAULT_PROMPT,
        provider=provider,
        use_llm=use_llm,
        max_retries=args.max_retries,
    )

    review_set = make_initial_review_set(result.config)

    approved_reviews = [
        approve_section(
            section_id=review.section_id,
            reviewer=args.reviewer,
            comment=DEFAULT_REVIEW_COMMENT,
        )
        for review in review_set.reviews
    ]

    final_result = finalize_report(
        result,
        ReviewSet(reviews=approved_reviews),
    )

    output_path = write_html_report(
        result=final_result,
        template_path=DEFAULT_TEMPLATE,
        output_path=Path(args.output),
    )

    print("=" * 72)
    print("EXAMPLE REPORT")
    print("=" * 72)
    print(f"config:   {args.config}")
    print(f"provider: {args.provider}")
    print(f"status:   {final_result.status.value}")
    print(f"output:   {output_path}")
    print("-" * 72)

    for section in final_result.sections:
        print(
            f"{section.section_id:32s} "
            f"{section.generation_source:22s} "
            f"{section.grounding_status}"
        )

    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
