"""Retrieval helpers for the customer support RAG assistant."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma

from config import AppConfig, get_config
from ingestion import get_embeddings


def vector_store_exists(config: AppConfig) -> bool:
    """Check whether the persisted Chroma directory appears to exist."""

    return config.chroma_dir.exists() and any(config.chroma_dir.iterdir())


def load_vector_store(config: AppConfig | None = None) -> Chroma:
    """Load the persisted Chroma vector store from disk."""

    config = config or get_config()

    if not vector_store_exists(config):
        raise FileNotFoundError(
            f"Chroma database not found in '{config.chroma_dir.resolve()}'. Run ingestion first."
        )

    return Chroma(
        collection_name=config.collection_name,
        embedding_function=get_embeddings(config),
        persist_directory=str(config.chroma_dir),
    )


def extract_sources(documents: List[Any]) -> List[str]:
    """Collect readable source names from retrieved document metadata."""

    sources: List[str] = []
    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        if source not in sources:
            sources.append(source)
    return sources


def expand_query(query: str) -> List[str]:
    """Expand short support queries with related phrases to improve recall."""

    expansions = {
        "delivery": ["shipping", "delivery time"],
        "refund": ["return policy", "money back"],
        "complaint": ["issue", "problem", "support"],
    }

    expanded = [query]

    for key, values in expansions.items():
        if key in query:
            expanded.extend(values)

    return expanded


def deduplicate_results(results: List[Tuple[Any, float]], top_k: int) -> List[Tuple[Any, float]]:
    """Keep the best-scoring unique documents after multi-query retrieval."""

    unique_results: List[Tuple[Any, float]] = []
    seen_keys = set()

    for document, score in sorted(results, key=lambda item: item[1]):
        doc_key = (
            document.metadata.get("source", "unknown"),
            document.metadata.get("page", -1),
            document.page_content,
        )
        if doc_key in seen_keys:
            continue

        seen_keys.add(doc_key)
        unique_results.append((document, score))

        if len(unique_results) >= top_k:
            break

    return unique_results


def retrieve_documents(query: str, config: AppConfig | None = None) -> Dict:
    """Run similarity search and return documents, distances, and source metadata."""

    config = config or get_config()
    vector_store = load_vector_store(config)
    # Normalize the query before expansion so retrieval is more consistent.
    query = query.lower().strip()
    queries = expand_query(query)

    results: List[Tuple[Any, float]] = []
    for expanded_query in queries:
        results.extend(vector_store.similarity_search_with_score(expanded_query, k=2))

    results = deduplicate_results(results, config.top_k)

    if not results:
        # No retrieval results means we should force HITL in the graph.
        return {
            "documents": [],
            "scores": [],
            "average_confidence": 0.0,
            "best_score": 0.0,
            "has_matches": False,
            "sources": [],
        }

    documents = [doc for doc, _ in results]
    distances = [score for _, score in results]
    # Chroma returns distance scores, so lower values mean higher similarity.
    average_confidence = round(sum(distances) / len(distances), 3)
    best_score = round(min(distances), 3)
    sources = extract_sources(documents)

    print(f"[DEBUG] Expanded Queries: {queries}")
    print(f"[DEBUG] Retrieved {len(documents)} chunks")
    print(f"[DEBUG] Best Score: {best_score}")
    print(f"[DEBUG] Distance Score (lower is better): {average_confidence}")

    return {
        "documents": documents,
        "scores": distances,
        "average_confidence": average_confidence,
        "best_score": best_score,
        "has_matches": True,
        "sources": sources,
    }
