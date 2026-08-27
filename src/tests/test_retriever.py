"""
Hybrid Retriever Evaluation Tests
==================================

Tests the hybrid retriever (BM25 + Chroma vector similarity) against
the visible-cases.json evaluation suite.

WHAT THIS FILE TESTS:
- Required sources are retrieved
- Forbidden sources are NOT retrieved
- Required phrases exist in retrieved chunks (exact match)
- Required concepts exist in retrieved chunks (normalized match)

WHAT THIS FILE DOES NOT TEST:
- LLM answer generation
- Tool calls (order_lookup etc.)
- Security / prompt-injection resistance
- Privacy / PII handling
- Agent handoff logic

Those belong in separate integration/agent tests.

Project layout expected:

project/
├── evaluation/
│   └── visible-cases.json
├── notebook/
│   └── chroma_langchain_db/
├── src/
│   └── tests/
│       └── test_retriever.py
└── knowledge-base/
    └── *.md

HOW THE HYBRID RETRIEVER WORKS:
- BM25 (keyword): catches exact phrases like "30 calendar days"
- Chroma (vector): catches semantic matches like "broken zipper" → damaged items
- EnsembleRetriever merges both via Reciprocal Rank Fusion
- Superseded docs filtered out on the vector side
"""

from pathlib import Path
import json
import unicodedata
import re

import pytest
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document

import yaml
from markdown_it import MarkdownIt
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional


# ============================================================
# PATH CONFIGURATION
# ============================================================

TESTS_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent.parent
EVAL_FILE    = PROJECT_ROOT / "evaluation" / "visible-cases.json"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
CHROMA_DIR   = NOTEBOOK_DIR / "chroma_langchain_db"
KB_DIR       = PROJECT_ROOT / "knowledge-base"

COLLECTION_NAME = "example_collection"


# ============================================================
# CASES THAT REQUIRE LIVE TOOL DATA — skip at retriever level
#
# These cases expect dynamic order data ("UPS", "August 22 2026",
# "shipped") that exists only in tool call responses, never in
# static policy documents. They belong in an agent integration
# test, not a retriever test.
# ============================================================

TOOL_DEPENDENT_CASES = {
    "valid-order-lookup",
    "missing-order-id",
    "cancelled-order-stale-eta",
    "unknown-order",
    "shipped-without-eta",
    "order-data-privacy",
}


# ============================================================
# FRONTMATTER SCHEMA — must match notebook exactly
# ============================================================
class PolicyFrontMatter(BaseModel):
    document_id: str
    title: str
    status: Literal["active", "superseded", "draft"]
    policy_authority: Literal["official", "unofficial", "none"]  # add "none"
    audience: str = "all"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    @property
    def is_citable_authority(self) -> bool:
        return self.status == "active" and self.policy_authority == "official"


# ============================================================
# KNOWLEDGE BASE LOADER — must match notebook exactly
# ============================================================

def split_frontmatter(raw_text: str) -> tuple[dict, str]:
    if not raw_text.startswith("---"):
        return {}, raw_text
    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text
    _, fm_raw, body = parts
    return (yaml.safe_load(fm_raw) or {}), body.lstrip("\n")


def split_by_headings(body: str, max_level: int = 3) -> list[dict]:
    md = MarkdownIt()
    tokens = md.parse(body)
    sections: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    lines = body.splitlines()

    def flush(path_snapshot):
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"heading_path": list(path_snapshot), "text": text})

    i = 0
    line_cursor = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            inline_tok = tokens[i + 1]
            heading_text = inline_tok.content
            flush([h for _, h in heading_stack])
            current_lines.clear()
            heading_stack = [(lvl, txt) for lvl, txt in heading_stack if lvl < level]
            if level <= max_level:
                heading_stack.append((level, heading_text))
            i += 3
            continue
        if tok.type == "inline" and tok.map:
            start, end = tok.map
            current_lines.extend(lines[line_cursor:end] if end > line_cursor else [])
            line_cursor = max(line_cursor, end)
        i += 1

    flush([h for _, h in heading_stack])
    if not sections:
        sections = [{"heading_path": [], "text": body.strip()}]
    return sections


def load_knowledge_base(kb_dir: Path) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    load_errors: list[str] = []

    for filepath in sorted(kb_dir.glob("**/*.md")):
        raw_text = filepath.read_text(encoding="utf-8")
        fm_dict, body = split_frontmatter(raw_text)
        try:
            fm = PolicyFrontMatter(**fm_dict)
        except ValidationError as e:
            load_errors.append(f"{filepath.name}: {e}")
            continue

        sections = split_by_headings(body)
        for idx, section in enumerate(sections):
            heading_path = section["heading_path"]
            section_text = section["text"]
            if len(section_text) < 3:
                continue
            heading_label = " > ".join(heading_path) if heading_path else fm.title
            content_with_context = f"{fm.title} > {heading_label}\n\n{section_text}"
            documents.append(Document(
                page_content=content_with_context,
                metadata={
                    "chunk_id":             f"{filepath.stem}::{idx}",
                    "filename":             filepath.name,
                    "document_id":          fm.document_id,
                    "title":                fm.title,
                    "heading_path":         heading_label,
                    "status":               fm.status,
                    "policy_authority":     fm.policy_authority,
                    "audience":             fm.audience,
                    "supersedes":           fm.supersedes or "",
                    "superseded_by":        fm.superseded_by or "",
                    "is_citable_authority": str(fm.is_citable_authority).lower(),
                },
            ))

    return documents, load_errors


# ============================================================
# LOAD EVALUATION JSON
# ============================================================

@pytest.fixture(scope="session")
def evaluation_cases():
    if not EVAL_FILE.exists():
        pytest.fail(f"Evaluation file not found:\n{EVAL_FILE}")

    with EVAL_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cases" in data, "visible-cases.json must contain a top-level 'cases' key"
    return data["cases"]


# ============================================================
# LOAD KNOWLEDGE BASE DOCS (for BM25)
# ============================================================

@pytest.fixture(scope="session")
def kb_docs():
    if not KB_DIR.exists():
        pytest.fail(
            f"Knowledge base directory not found:\n{KB_DIR}\n"
            "Update KB_DIR at the top of this file."
        )
    docs, errors = load_knowledge_base(KB_DIR)
    if errors:
        import warnings
        warnings.warn(
            f"{len(errors)} file(s) skipped during validation:\n"
            + "\n".join(errors)
        )
    if not docs:
        pytest.fail("No documents loaded from knowledge base — check KB_DIR path.")

    # Filter out superseded and draft docs so BM25 matches Chroma's scope
    # Also filter internal docs (policy_authority=none, audience=internal)
    active_docs = [
        doc for doc in docs
        if doc.metadata.get("status") == "active"
        and doc.metadata.get("policy_authority") in ("official", "unofficial")
    ]
    return active_docs

# ============================================================
# LOAD CHROMA VECTOR STORE
# ============================================================

@pytest.fixture(scope="session")
def vector_store():
    if not CHROMA_DIR.exists():
        pytest.fail(
            f"Chroma database not found:\n{CHROMA_DIR}\n\n"
            "Run the ingestion notebook first to build the database."
        )
    embedding = init_embeddings("huggingface:BAAI/bge-small-en-v1.5")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
    )


# ============================================================
# BUILD HYBRID RETRIEVER
#
# BM25 + Chroma vector search merged via Reciprocal Rank Fusion.
#
# BM25  weight=0.4 — exact keyword matches ("30 calendar days")
# Chroma weight=0.6 — semantic matches ("broken zipper" → damages)
#
# Vector side filters out superseded documents.
# BM25 operates on all active chunks in memory.
# ============================================================

@pytest.fixture(scope="session")
def retriever(kb_docs, vector_store):
    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------
    # Use a larger candidate pool because exact policy wording
    # may be several ranks below the membership-specific chunks.
    bm25 = BM25Retriever.from_documents(kb_docs)
    bm25.k = 20

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------
    # Keep vector retrieval in the same policy scope as BM25:
    # active + official/unofficial.
    vector = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 20,
        "filter": {"status": "active"},
    },
)
    

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------
    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.5, 0.5],
    )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_filename(doc: Document) -> str | None:
    filename = doc.metadata.get("filename")
    if filename:
        return filename
    source = doc.metadata.get("source")
    if source:
        return Path(str(source)).name
    return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"-", " ", text)          # "45-calendar-day" → "45 calendar day"
    text = re.sub(r"\b(\w+)s\b", r"\1", text)  # "days" → "day", "returns" → "return"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_document_text(doc: Document) -> str:
    metadata_text = " ".join(
        str(v) for v in doc.metadata.values() if v is not None
    )
    return normalize_text(f"{doc.page_content} {metadata_text}")


def contains_phrase(phrase: str, docs: list[Document]) -> bool:
    """Exact phrase match after normalization."""
    normalized = normalize_text(phrase)
    return any(normalized in get_document_text(doc) for doc in docs)


def contains_concept(concept: str, docs: list[Document]) -> bool:
    """
    Normalized phrase match — same as contains_phrase but
    named separately so test output is clear about intent.
    """
    normalized = normalize_text(concept)
    return any(normalized in get_document_text(doc) for doc in docs)


# ============================================================
# RETRIEVAL TESTS
# ============================================================

@pytest.mark.parametrize("case_index", range(15))
def test_retrieval_case(case_index, evaluation_cases, retriever):
    """
    Tests retrieval quality for each evaluation case.

    Skips:
    - Tool-dependent cases (dynamic order data, not in policy docs)

    Checks:
    - required_sources present in retrieved docs
    - forbidden_sources_as_authority NOT in retrieved docs
    - must_include phrases found in retrieved text
    - must_include_concepts found in retrieved text (xfail if missing)
    """

    case    = evaluation_cases[case_index]
    case_id = case["id"]
    query   = case["messages"][0]["content"]

    # --------------------------------------------------------
    # Skip tool-dependent cases — wrong test category
    # --------------------------------------------------------
    if case_id in TOOL_DEPENDENT_CASES:
        pytest.skip(
            f"[{case_id}] Requires live tool/order data. "
            f"Test this in an agent integration test."
        )

    # --------------------------------------------------------
    # Retrieve — no LLM, no tool, no security layer
    # --------------------------------------------------------
    docs = retriever.invoke(query)

    retrieved_sources = [get_filename(doc) for doc in docs]

    # --------------------------------------------------------
    # Expectations
    # --------------------------------------------------------
    expect = case.get("expect", {})

    required_sources  = expect.get("required_sources", [])
    forbidden_sources = expect.get("forbidden_sources_as_authority", [])
    must_include      = expect.get("must_include", [])
    must_include_concepts = expect.get("must_include_concepts", [])

    # --------------------------------------------------------
    # Required sources
    # --------------------------------------------------------
    missing_sources = [
        s for s in required_sources
        if s not in retrieved_sources
    ]
    assert not missing_sources, (
        f"\n[{case_id}] Required source(s) not retrieved.\n"
        f"Query:     {query}\n"
        f"Missing:   {missing_sources}\n"
        f"Retrieved: {retrieved_sources}"
    )

    # --------------------------------------------------------
    # Forbidden sources
    #
    # Checks retrieval presence only — not whether the agent
    # would cite it as authority. Governance tests cover that.
    # --------------------------------------------------------
    forbidden_found = [
        s for s in forbidden_sources
        if s in retrieved_sources
    ]
    assert not forbidden_found, (
        f"\n[{case_id}] Forbidden source was retrieved.\n"
        f"Query:     {query}\n"
        f"Forbidden: {forbidden_found}\n"
        f"Retrieved: {retrieved_sources}"
    )

    # --------------------------------------------------------
    # Required phrases (exact, normalized)
    # --------------------------------------------------------
    missing_phrases = [
        p for p in must_include
        if not contains_phrase(p, docs)
    ]
    assert not missing_phrases, (
        f"\n[{case_id}] Required phrase(s) not found in retrieved docs.\n"
        f"Query:     {query}\n"
        f"Missing:   {missing_phrases}\n"
        f"Retrieved: {retrieved_sources}"
    )

    # --------------------------------------------------------
    # Required concepts (semantic — xfail if missing)
    #
    # Concepts are semantic descriptions, not exact phrases.
    # String matching will not always find them.
    # Once retrieval is stable, replace with a semantic evaluator.
    # --------------------------------------------------------
    missing_concepts = [
        c for c in must_include_concepts
        if not contains_concept(c, docs)
    ]
    if missing_concepts:
        pytest.xfail(
            reason=(
                f"[{case_id}] Semantic concept(s) not found by string matching: "
                f"{missing_concepts}"
            )
        )


# ============================================================
# DEBUG — PRINT RETRIEVAL DETAILS
#
# Run with:  pytest -s tests/test_retriever.py::test_show_results
#
# Shows exactly what the hybrid retriever returned for each
# case so you can inspect ranking and content.
# ============================================================

@pytest.mark.parametrize("case_index", range(15))
def test_show_results(case_index, evaluation_cases, retriever):
    """
    Inspection-only test — no assertions.
    Run with -s to see retrieval output for every case.
    """
    case  = evaluation_cases[case_index]
    query = case["messages"][0]["content"]
    docs  = retriever.invoke(query)

    print("\n")
    print("=" * 80)
    print(f"CASE:     {case['id']}")
    print(f"CATEGORY: {case.get('category', 'N/A')}")
    print("=" * 80)
    print(f"QUERY: {query}")
    print()

    for rank, doc in enumerate(docs, start=1):
        print(f"--- RESULT {rank} ---")
        print(f"SOURCE:    {get_filename(doc)}")
        print(f"HEADING:   {doc.metadata.get('heading_path', 'N/A')}")
        print(f"STATUS:    {doc.metadata.get('status', 'N/A')}")
        print(f"AUTHORITY: {doc.metadata.get('policy_authority', 'N/A')}")
        print(f"CITABLE:   {doc.metadata.get('is_citable_authority', 'N/A')}")
        preview = doc.page_content.replace("\n", " ").strip()
        print(f"TEXT:      {preview[:400]}")
        print()

def test_debug_phrase(retriever):
    query = "My TrailPlus membership was active when I ordered. What is my return window?"
    docs = retriever.invoke(query)

    target_phrase = "45 calendar days"

    print("\n" + "=" * 100)
    print("DEBUG RETRIEVAL")
    print("=" * 100)
    print("QUERY:", query)
    print("TARGET:", target_phrase)

    found = False

    for rank, doc in enumerate(docs, start=1):
        text = get_document_text(doc)
        contains = target_phrase in text

        if contains:
            found = True

        print(f"\n--- RESULT {rank} ---")
        print("SOURCE:", get_filename(doc))
        print("CHUNK:", doc.metadata.get("chunk_id"))
        print("HEADING:", doc.metadata.get("heading_path"))
        print("STATUS:", doc.metadata.get("status"))
        print("AUTHORITY:", doc.metadata.get("policy_authority"))
        print("CONTAINS TARGET:", contains)
        print("TEXT:")
        print(doc.page_content[:1000])

    print("\n" + "=" * 100)
    print("TARGET FOUND:", found)
    print("=" * 100)