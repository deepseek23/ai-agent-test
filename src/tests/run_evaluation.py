"""
Evaluation CLI — run full agent evaluation suite and print per-case results.

Usage:
    python -m src.tests.run_evaluation
    python -m src.tests.run_evaluation --visible-only
    python -m src.tests.run_evaluation --json evaluation/results.json
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.evaluation.report import format_full_report, write_json_report
from src.evaluation.runner import load_all_cases, load_cases, run_all_cases, VISIBLE_CASES_FILE
from src.llm import get_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aster & Row agent evaluation suite")
    parser.add_argument(
        "--visible-only",
        action="store_true",
        help="Run only evaluation/visible-cases.json",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default="evaluation/results.json",
        help="Write JSON results to this path (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON report",
    )
    args = parser.parse_args(argv)

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set. Add it to .env before running evaluation.")
        return 1

    print("Initializing agent...")
    get_agent()

    cases = load_cases(VISIBLE_CASES_FILE) if args.visible_only else load_all_cases()
    print(f"Running {len(cases)} evaluation case(s)...")

    results = run_all_cases(cases)
    report = format_full_report(results)
    print(report)

    if not args.no_json:
        from pathlib import Path

        from src.config import PROJECT_ROOT

        output_path = Path(args.json)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        written = write_json_report(results, path=output_path)
        print(f"\nJSON report written to: {written}")

    failed = sum(1 for r in results if not r.passed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
