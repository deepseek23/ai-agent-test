import os

from dotenv import load_dotenv
import pytest

load_dotenv()

from src.evaluation.runner import load_all_cases, load_cases, VISIBLE_CASES_FILE, CUSTOM_CASES_FILE


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: live agent evaluation requiring API key and chroma db")


def _has_live_credentials() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


@pytest.fixture(scope="session")
def all_eval_cases():
    return load_all_cases()


@pytest.fixture(scope="session")
def visible_eval_cases():
    return load_cases(VISIBLE_CASES_FILE)


@pytest.fixture(scope="session")
def custom_eval_cases():
    return load_cases(CUSTOM_CASES_FILE)


@pytest.fixture(scope="session", autouse=True)
def _init_agent_for_integration(request):
    """Warm up agent once when integration tests run."""
    session = request.session
    if not session.items:
        return
    has_integration = any(
        item.get_closest_marker("integration") for item in session.items
    )
    if has_integration and _has_live_credentials():
        from src.llm import get_agent

        get_agent()
