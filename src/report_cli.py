from __future__ import annotations

import argparse
from pathlib import Path

from src.report_pipeline import (
    run_report_pipeline,
    write_html_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a controlled safety report."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the safety CSV dataset.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the report YAML configuration.",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Path to the section-generation prompt.",
    )

    parser.add_argument(
        "--template",
        required=True,
        help="Path to the HTML report template.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated HTML report.",
    )

    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use the configured LLM provider. "
            "Without this flag, deterministic no-LLM mode is used."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum grounding retries for LLM sections.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = run_report_pipeline(
        csv_path=Path(args.csv),
        config_path=Path(args.config),
        prompt_path=Path(args.prompt),
        use_llm=args.llm,
        max_retries=args.max_retries,
    )

    output_path = write_html_report(
        result=result,
        template_path=Path(args.template),
        output_path=Path(args.output),
    )

    print("=" * 72)
    print("REPORT PIPELINE")
    print("=" * 72)
    print(f"report_id: {result.config.report.id}")
    print(f"report_version: {result.config.report.version}")
    print(f"cases: {len(result.normalized.cases)}")
    print(f"reactions: {len(result.normalized.reactions)}")
    print(f"analyses: {len(result.analyses)}")
    print(f"sections: {len(result.sections)}")
    print(f"mode: {'llm' if args.llm else 'no-llm'}")
    print(f"output: {output_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())