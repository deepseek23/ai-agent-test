from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from src.ingest import get_active_documents, load_documents


def build_hybrid_retriever(
    documents: list[Document] | None = None,
    bm25_k: int = 10,
    vector_k: int = 10,
    weights: tuple[float, float] = (0.5, 0.5),
) -> EnsembleRetriever:
    if documents is None:
        documents = load_documents()

    active_docs = get_active_documents(documents)

    bm25_retriever = BM25Retriever.from_documents(active_docs)
    bm25_retriever.k = bm25_k

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=init_embeddings(EMBEDDING_MODEL),
    )
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": vector_k, "filter": {"status": "active"}},
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights),
    )
