"""Map evaluation case categories to report buckets."""

REPORT_CATEGORIES = [
    "retrieval",
    "groundedness",
    "tool_use",
    "privacy",
    "multi_turn",
]

CATEGORY_TO_REPORT: dict[str, str] = {
    "retrieval": "retrieval",
    "multi-source-grounding": "groundedness",
    "groundedness": "groundedness",
    "abstention": "groundedness",
    "source-conflict": "groundedness",
    "conversation": "multi_turn",
    "tool-use": "tool_use",
    "tool-reliability": "tool_use",
    "privacy": "privacy",
    "prompt-security": "privacy",
}


def report_category(case_category: str) -> str:
    return CATEGORY_TO_REPORT.get(case_category, case_category)
