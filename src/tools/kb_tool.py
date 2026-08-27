from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool


class KBQueryInput(BaseModel):
    query: str = Field(description="Natural-language question to look up in the knowledge base.")


def format_chunks_for_llm(chunks: list[Document]) -> str:
    """Serialize retrieved LangChain documents into a labelled context block."""
    if not chunks:
        return "[No relevant passages found in the knowledge base.]"

    lines = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata
        status = metadata.get("status", "unknown")
        status_label = (
            "SUPERSEDED - do not use as authoritative source"
            if status == "superseded"
            else status.upper()
        )
        source_ref = metadata.get("filename", metadata.get("source", "unknown"))
        doc_type = metadata.get("policy_authority", "unknown")
        lines.append(
            f"--- PASSAGE {i} ---\n"
            f"Source : {source_ref}\n"
            f"Status : {status_label}\n"
            f"Type   : {doc_type}\n"
            f"Content:\n{chunk.page_content}\n"
        )
    return "\n".join(lines)


def make_kb_tool(retriever) -> StructuredTool:
    def _run_kb_query(query: str) -> str:
        chunks = retriever.invoke(query)
        return format_chunks_for_llm(chunks)

    return StructuredTool.from_function(
        func=_run_kb_query,
        name="knowledge_base_search",
        description=(
            "Search Aster & Row's knowledge base for policies, shipping info, "
            "product details, FAQs, and procedures. "
            "Use this for ANY company-specific question before answering from general knowledge. "
            "Returns ranked, source-cited passages."
        ),
        args_schema=KBQueryInput,
    )
