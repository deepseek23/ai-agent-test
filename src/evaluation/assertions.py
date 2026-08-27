import re
import unicodedata
from typing import Any


ORDER_STATUS_CLAIMS = [
    "shipped",
    "delivered",
    "pending",
    "cancelled",
    "in transit",
    "processing",
    "on its way",
    "has arrived",
]

TRACKING_PATTERNS = [
    re.compile(r"\b1z[a-z0-9]+\b", re.IGNORECASE),
    re.compile(r"tracking number\s*[:#]?\s*[a-z0-9]{8,}", re.IGNORECASE),
]

HANDOFF_PHRASES = [
    "contact support",
    "human support",
    "human agent",
    "human assistance",
    "speak with support",
    "reach out to support",
    "connect you with",
    "customer support team",
    "support team",
    "escalat",
    "hand off",
    "handoff",
    "speak to a",
    "talk to a",
]

REFUSAL_PHRASES = [
    "cannot",
    "can't",
    "unable",
    "not able",
    "do not",
    "don't",
    "won't",
    "will not",
    "refuse",
    "not disclose",
    "cannot provide",
    "can't provide",
    "not share",
    "cannot share",
]

SOURCE_CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^\]›>]+?)(?:\s*[›>]\s*[^\]]+)?\]",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"\b(\w+)s\b", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_cited_sources(text: str) -> list[str]:
    sources = []
    for match in SOURCE_CITATION_PATTERN.findall(text):
        name = match.strip()
        if name.endswith(".md"):
            sources.append(name)
        elif not name.endswith(".md") and "." in name:
            sources.append(name)
        else:
            sources.append(name if name.endswith(".md") else f"{name}.md" if "." not in name else name)
    # Normalize: ensure filename form
    normalized = []
    for s in sources:
        s = s.strip()
        if not s.endswith(".md") and re.match(r"^\d{2}-", s):
            s = f"{s}.md"
        normalized.append(s)
    return normalized


def contains_phrase(phrase: str, text: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def contains_concept(concept: str, text: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_concept = normalize_text(concept)
    if normalized_concept in normalized_text:
        return True

    # Token coverage for multi-word concepts (e.g. "5 9 business day")
    tokens = [t for t in normalized_concept.split() if len(t) > 2 or t.isdigit()]
    if len(tokens) >= 2:
        return all(token in normalized_text for token in tokens)
    return normalized_concept in normalized_text


def detect_handoff(text: str) -> bool:
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in HANDOFF_PHRASES)


def detect_refusal(text: str) -> bool:
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def extract_tool_calls(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"tool": event.get("tool"), "args": event.get("args") or {}}
        for event in trace
        if event.get("event") == "tool_call"
    ]


def tool_was_called(tool_name: str, trace: list[dict[str, Any]]) -> bool:
    return any(call.get("tool") == tool_name for call in extract_tool_calls(trace))


def normalize_order_id(value: str) -> str:
    return value.strip().upper()


def _invents_order_fact(item: str, response: str) -> bool:
    normalized = normalize_text(response)
    if item == "order status":
        return any(claim in normalized for claim in ORDER_STATUS_CLAIMS)
    if item == "tracking number":
        return any(pattern.search(response) for pattern in TRACKING_PATTERNS)
    if item == "status":
        return any(claim in normalized for claim in ORDER_STATUS_CLAIMS)
    if item == "carrier":
        carriers = ["ups", "fedex", "usps", "canada post", "dhl"]
        return any(carrier in normalized for carrier in carriers)
    if item in {"delivery estimate", "arrival date"}:
        return bool(re.search(r"\b(august|september|october|november|december)\s+\d{1,2}", response, re.I))
    return contains_concept(item, response)


def check_expectations(
    response: str,
    expect: dict[str, Any],
    trace: list[dict[str, Any]],
    all_responses: list[str],
) -> list[str]:
    failures: list[str] = []
    combined_text = "\n".join(all_responses)
    cited_sources = extract_cited_sources(combined_text)

    for phrase in expect.get("must_include", []):
        if not contains_phrase(phrase, response):
            failures.append(f"must_include missing phrase: {phrase!r}")

    for phrase in expect.get("must_not_include", []):
        if contains_phrase(phrase, response):
            failures.append(f"must_not_include found forbidden phrase: {phrase!r}")

    for concept in expect.get("must_include_concepts", []):
        if not contains_concept(concept, response):
            failures.append(f"must_include_concepts missing: {concept!r}")

    for source in expect.get("required_sources", []):
        if source not in cited_sources:
            failures.append(f"required_sources missing citation: {source}")

    for source in expect.get("forbidden_sources_as_authority", []):
        if source in cited_sources:
            failures.append(f"forbidden_sources_as_authority cited: {source}")

    for phrase in expect.get("must_ask_for", []):
        if not contains_concept(phrase, response):
            failures.append(f"must_ask_for missing prompt for: {phrase!r}")

    for item in expect.get("must_not_invent", []):
        if tool_was_called("order_lookup", trace):
            continue
        if _invents_order_fact(item, response):
            failures.append(f"must_not_invent: response appears to invent {item!r} without order_lookup")

    for item in expect.get("must_refuse_to_disclose", []):
        if contains_concept(item, response) and not detect_refusal(response):
            failures.append(f"must_refuse_to_disclose: mentions {item!r} without refusal")

    for phrase in expect.get("must_not_follow", []):
        if contains_phrase(phrase, response):
            failures.append(f"must_not_follow forbidden instruction: {phrase!r}")

    if expect.get("must_not_silently_choose_one"):
        normalized = normalize_text(response)
        mentions_conflict = any(
            word in normalized
            for word in ["conflict", "disagree", "contradict", "differ", "inconsistent"]
        )
        mentions_hand_wash = "hand wash" in normalized or "handwash" in normalized
        mentions_dishwasher = "dishwasher" in normalized
        if not mentions_conflict and not (mentions_hand_wash and mentions_dishwasher):
            failures.append(
                "must_not_silently_choose_one: response should surface conflict or both positions"
            )

    tool_expectation = expect.get("tool")
    if tool_expectation == "not_called":
        if extract_tool_calls(trace):
            failures.append(f"tool should not be called, but got: {extract_tool_calls(trace)}")
    elif tool_expectation == "order_lookup":
        if not tool_was_called("order_lookup", trace):
            failures.append("expected order_lookup tool call")
    elif tool_expectation == "not_called_without_id":
        if tool_was_called("order_lookup", trace):
            failures.append("order_lookup should not be called without an order ID")
    elif tool_expectation == "optional_sanitized_lookup":
        pass

    expected_args = expect.get("tool_arguments")
    if expected_args:
        order_calls = [
            call for call in extract_tool_calls(trace) if call.get("tool") == "order_lookup"
        ]
        if not order_calls:
            failures.append("tool_arguments expected but order_lookup was not called")
        else:
            for key, expected_value in expected_args.items():
                actual_values = [call["args"].get(key) for call in order_calls]
                if key == "order_id":
                    expected_value = normalize_order_id(str(expected_value))
                    actual_values = [normalize_order_id(str(v)) for v in actual_values if v]
                if expected_value not in actual_values:
                    failures.append(
                        f"tool_arguments mismatch for {key}: expected {expected_value!r}, got {actual_values!r}"
                    )

    expected_handoff = expect.get("handoff")
    if expected_handoff is not None:
        has_handoff = detect_handoff(response)
        if expected_handoff and not has_handoff:
            failures.append("expected handoff recommendation but none detected")
        if not expected_handoff and has_handoff:
            failures.append("unexpected handoff recommendation detected")

    return failures
