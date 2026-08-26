from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE = BASE_DIR / "knowledge-base"

loader = DirectoryLoader(
    str(KNOWLEDGE_BASE),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.lazy_load_documents()

print(f"Loaded {len(documents)} documents")