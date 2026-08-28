from pathlib import Path
from typing import Literal, Optional

import yaml
from markdown_it import MarkdownIt
from pydantic import BaseModel, ValidationError
from langchain_core.documents import Document

from src.config import KB_DIR


class PolicyFrontMatter(BaseModel):
    document_id: str
    title: str
    status: Literal["active", "superseded", "draft"]
    policy_authority: Literal["official", "unofficial", "none"]
    audience: str = "all"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    @property
    def is_citable_authority(self) -> bool:
        return self.status == "active" and self.policy_authority == "official"


def split_frontmatter(raw_text: str) -> tuple[dict, str]:
    if not raw_text.startswith("---"):
        return {}, raw_text
    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text
    return (yaml.safe_load(parts[1]) or {}), parts[2].lstrip("\n")


def split_by_headings(body: str, max_level: int = 3) -> list[dict]:
    md = MarkdownIt()
    tokens = md.parse(body)
    sections, heading_stack, current_lines = [], [], []
    lines = body.splitlines()

    def flush(path_snapshot):
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"heading_path": list(path_snapshot), "text": text})

    i, line_cursor = 0, 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            heading_text = tokens[i + 1].content
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
    return sections or [{"heading_path": [], "text": body.strip()}]


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    documents: list[Document] = []
    for filepath in sorted(kb_dir.glob("**/*.md")):
        raw_text = filepath.read_text(encoding="utf-8")
        fm_dict, body = split_frontmatter(raw_text)
        try:
            fm = PolicyFrontMatter(**fm_dict)
        except ValidationError:
            continue

        for idx, section in enumerate(split_by_headings(body)):
            if len(section["text"]) < 3:
                continue
            heading_label = (
                " > ".join(section["heading_path"]) if section["heading_path"] else fm.title
            )
            documents.append(
                Document(
                    page_content=f"{fm.title} > {heading_label}\n\n{section['text']}",
                    metadata={
                        "document_id": fm.document_id,
                        "title": fm.title,
                        "filename": filepath.name,
                        "status": fm.status,
                        "policy_authority": fm.policy_authority,
                        "audience": fm.audience,
                        "supersedes": fm.supersedes or "",
                        "superseded_by": fm.superseded_by or "",
                        "heading": heading_label,
                        "heading_path": " > ".join(section["heading_path"]),
                    },
                )
            )
    return documents


def get_active_documents(documents: list[Document]) -> list[Document]:
    return [
        d
        for d in documents
        if d.metadata.get("status") == "active"
        and d.metadata.get("policy_authority") in ("official", "unofficial")
    ]


def build_vector_store(*, force: bool = False) -> int:
    """Build or rebuild the persisted Chroma index from active KB chunks."""
    import shutil

    from dotenv import load_dotenv
    from langchain.embeddings import init_embeddings
    from langchain_chroma import Chroma

    from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

    load_dotenv()

    active_docs = get_active_documents(load_documents())
    if not active_docs:
        raise ValueError("No active documents found to index.")

    if force and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=active_docs,
        embedding=init_embeddings(EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return len(active_docs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the Chroma vector index.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing index before rebuilding.",
    )
    args = parser.parse_args()

    count = build_vector_store(force=args.force)
    from src.config import CHROMA_DIR

    print(f"Indexed {count} active chunks into {CHROMA_DIR}")
