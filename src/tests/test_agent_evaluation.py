import os

import pytest

from src.evaluation.runner import load_all_cases, run_case


pytestmark = pytest.mark.integration

ALL_CASES = load_all_cases()


def _skip_if_no_credentials():
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set — skipping live agent evaluation")


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c["id"])
def test_agent_evaluation_case(case):
    _skip_if_no_credentials()
    result = run_case(case)
    if not result.passed:
        detail = "\n".join(result.failures)
        pytest.fail(f"Case {result.case_id} failed:\n{detail}\n\nResponse:\n{result.response}")
