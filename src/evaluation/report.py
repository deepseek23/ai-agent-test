import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.evaluation.categories import REPORT_CATEGORIES
from src.evaluation.runner import CaseResult


def _pct(passed: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{round(100 * passed / total, 1)}%"


def format_case_line(result: CaseResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    line = f"[{status}] {result.case_id} ({result.category}) — {result.elapsed_ms}ms"
    if result.failures:
        line += "\n    " + "\n    ".join(result.failures)
    return line


def format_category_summary(results: list[CaseResult]) -> str:
    buckets: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        buckets[result.report_category].append(result)

    lines = ["", "Category summary:", "=" * 60]
    for category in REPORT_CATEGORIES:
        bucket = buckets.get(category, [])
        passed = sum(1 for r in bucket if r.passed)
        total = len(bucket)
        lines.append(f"  {category:14} {passed}/{total} ({_pct(passed, total)})")

    other = [r for r in results if r.report_category not in REPORT_CATEGORIES]
    if other:
        passed = sum(1 for r in other if r.passed)
        lines.append(f"  other          {passed}/{len(other)} ({_pct(passed, len(other))})")

    overall_passed = sum(1 for r in results if r.passed)
    lines.append("=" * 60)
    lines.append(f"  overall        {overall_passed}/{len(results)} ({_pct(overall_passed, len(results))})")
    return "\n".join(lines)


def format_full_report(results: list[CaseResult]) -> str:
    lines = ["Evaluation results", "=" * 60]
    for result in results:
        lines.append(format_case_line(result))
    lines.append(format_category_summary(results))
    return "\n".join(lines)


def write_json_report(results: list[CaseResult], path: Path | None = None) -> Path:
    output_path = path or (PROJECT_ROOT / "evaluation" / "results.json")
    payload: dict[str, Any] = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "cases": [asdict(r) for r in results],
        "categories": {},
    }

    buckets: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        buckets[result.report_category].append(result)

    for category, bucket in buckets.items():
        payload["categories"][category] = {
            "total": len(bucket),
            "passed": sum(1 for r in bucket if r.passed),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
