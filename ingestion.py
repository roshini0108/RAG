"""PDF ingestion pipeline for the customer support RAG assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import AppConfig, get_config


def get_embeddings(config: AppConfig):
    """Create the Ollama embedding model used for vector search."""

    return OllamaEmbeddings(model=config.embedding_model)


def find_pdf_files(data_dir: Path) -> List[Path]:
    """Return all PDF files found in the configured data directory."""

    # Create the data folder on first run so users know where to place PDFs.
    data_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.pdf"))


def load_pdf_documents(pdf_paths: Iterable[Path]) -> List[Document]:
    """Load all PDF pages from the provided file paths."""

    documents: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


def split_documents(documents: List[Document], config: AppConfig) -> List[Document]:
    """Split loaded documents into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document], config: AppConfig) -> Chroma:
    """Create or update the local Chroma vector store with document chunks."""

    # Ensure the persistence folder exists before Chroma writes local files.
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings(config)
    vector_store = Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.chroma_dir),
    )
    vector_store.add_documents(chunks)
    return vector_store


def ingest_documents(config: AppConfig | None = None) -> None:
    """Run the full PDF ingestion pipeline and persist vectors to Chroma."""

    config = config or get_config()
    pdf_files = find_pdf_files(config.data_dir)

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{config.data_dir.resolve()}'. Add PDFs and try again."
        )

    documents = load_pdf_documents(pdf_files)
    if not documents:
        raise ValueError("PDF files were found, but no readable content was extracted.")

    chunks = split_documents(documents, config)
    build_vector_store(chunks, config)

    print(f"Ingestion complete. Indexed {len(pdf_files)} PDF(s) into ChromaDB.")


if __name__ == "__main__":
    ingest_documents()
