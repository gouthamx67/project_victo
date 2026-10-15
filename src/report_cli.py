from __future__ import annotations

import argparse
from pathlib import Path

from src.env_loader import load_dotenv_file
from src.llm_provider import build_provider
from src.report_pipeline import (
    run_report_pipeline,
    write_html_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv_file(PROJECT_ROOT / ".env")


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
            "Deprecated alias for --provider mock. Prefer --provider."
        ),
    )

    parser.add_argument(
        "--provider",
        choices=["none", "mock", "openai", "groq"],
        default=None,
        help=(
            "LLM provider for narrative sections. "
            "'none' (default) uses deterministic templates; "
            "'mock' is an offline test double; "
            "'groq' uses the Groq chat/completions API (GROQ_API_KEY); "
            "'openai' uses the OpenAI Responses API (OPENAI_API_KEY)."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id for the selected provider. Defaults to the "
            "provider's environment variable (GROQ_MODEL / OPENAI_MODEL) "
            "or a built-in default."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum output tokens per generated section.",
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

    provider_name = args.provider
    if provider_name is None:
        provider_name = "mock" if args.llm else "none"

    provider = build_provider(
        provider=provider_name,
        model=args.model,
        max_tokens=args.max_tokens,
    )

    use_llm = provider is not None

    result = run_report_pipeline(
        csv_path=Path(args.csv),
        config_path=Path(args.config),
        prompt_path=Path(args.prompt),
        provider=provider,
        use_llm=use_llm,
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
    print(f"provider: {provider_name}")
    print(f"output: {output_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())