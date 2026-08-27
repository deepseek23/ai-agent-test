import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent import run_agent
from src.config import PROJECT_ROOT
from src.evaluation.assertions import check_expectations
from src.evaluation.categories import report_category

CASE_DELAY_SECONDS = float(os.getenv("EVAL_CASE_DELAY", "6"))


VISIBLE_CASES_FILE = PROJECT_ROOT / "evaluation" / "visible-cases.json"
CUSTOM_CASES_FILE = PROJECT_ROOT / "evaluation" / "custom-cases.json"


@dataclass
class CaseResult:
    case_id: str
    category: str
    report_category: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    response: str = ""
    all_responses: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data["cases"]


def load_all_cases() -> list[dict[str, Any]]:
    visible = load_cases(VISIBLE_CASES_FILE)
    custom = load_cases(CUSTOM_CASES_FILE)
    return visible + custom


def run_case(case: dict[str, Any]) -> CaseResult:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    category = case.get("category", "unknown")
    expect = case.get("expect", {})
    all_responses: list[str] = []
    full_trace: list[dict[str, Any]] = []

    started = time.perf_counter()
    try:
        for message in case.get("messages", []):
            if message.get("role") != "user":
                continue
            response, trace = run_agent(
                message["content"],
                thread_id=thread_id,
                include_trace=True,
            )
            all_responses.append(response)
            full_trace.extend(trace)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return CaseResult(
            case_id=case["id"],
            category=category,
            report_category=report_category(category),
            passed=False,
            failures=[f"runtime error: {exc}"],
            response=all_responses[-1] if all_responses else "",
            all_responses=all_responses,
            trace=full_trace,
            elapsed_ms=elapsed_ms,
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    final_response = all_responses[-1] if all_responses else ""

    failures = check_expectations(
        final_response,
        expect,
        full_trace,
        all_responses,
    )

    return CaseResult(
        case_id=case["id"],
        category=category,
        report_category=report_category(category),
        passed=not failures,
        failures=failures,
        response=final_response,
        all_responses=all_responses,
        trace=full_trace,
        elapsed_ms=elapsed_ms,
    )


def run_all_cases(cases: list[dict[str, Any]] | None = None) -> list[CaseResult]:
    if cases is None:
        cases = load_all_cases()
    results: list[CaseResult] = []
    for index, case in enumerate(cases):
        if index > 0 and CASE_DELAY_SECONDS > 0:
            time.sleep(CASE_DELAY_SECONDS)
        results.append(run_case(case))
    return results
