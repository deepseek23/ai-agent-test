"""Unit tests for deterministic evaluation assertions (no live LLM)."""

from src.evaluation.assertions import (
    check_expectations,
    contains_concept,
    detect_handoff,
    extract_cited_sources,
    extract_tool_calls,
)


def test_extract_cited_sources():
    text = "Policy is 30 days. [Source: 01-returns-policy-current.md › Standard return window]"
    assert extract_cited_sources(text) == ["01-returns-policy-current.md"]


def test_tool_call_expectation_order_lookup():
    trace = [
        {"event": "tool_call", "tool": "order_lookup", "args": {"order_id": "ORD-1007"}},
    ]
    failures = check_expectations(
        "Shipped via UPS. [Source: orders]",
        {
            "tool": "order_lookup",
            "tool_arguments": {"order_id": "ORD-1007"},
            "must_include": ["UPS"],
            "handoff": False,
        },
        trace,
        ["Shipped via UPS. [Source: orders]"],
    )
    assert failures == []


def test_tool_not_called_fails_when_tool_used():
    trace = [{"event": "tool_call", "tool": "order_lookup", "args": {"order_id": "ORD-1007"}}]
    failures = check_expectations(
        "Answer",
        {"tool": "not_called", "handoff": False},
        trace,
        ["Answer"],
    )
    assert any("tool should not be called" in f for f in failures)


def test_must_not_invent_without_lookup():
    failures = check_expectations(
        "Your order is shipped with UPS.",
        {
            "must_not_invent": ["order status"],
            "tool": "not_called_without_id",
            "handoff": False,
        },
        [],
        ["Your order is shipped with UPS."],
    )
    assert any("must_not_invent" in f for f in failures)


def test_ask_for_order_id_passes_without_invention():
    failures = check_expectations(
        "Please provide your order ID so I can check the status.",
        {
            "must_ask_for": ["order ID"],
            "must_not_invent": ["order status", "tracking number"],
            "tool": "not_called_without_id",
            "handoff": False,
        },
        [],
        ["Please provide your order ID so I can check the status."],
    )
    assert failures == []


def test_handoff_detection():
    assert detect_handoff("Please contact support for help.")
    assert not detect_handoff("Your return window is 30 days.")


def test_concept_matching():
    assert contains_concept("30 calendar days", "Customers may return within 30 calendar days of delivery.")
