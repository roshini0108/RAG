"""CLI entrypoint for the customer support RAG assistant."""

from __future__ import annotations

from graph import build_graph
from ingestion import ingest_documents
from retrieval import vector_store_exists
from config import get_config


def ensure_knowledge_base() -> None:
    """Ensure the vector database exists before accepting user questions."""

    config = get_config()
    # Make the data directory available on first run for easy PDF drop-in.
    config.data_dir.mkdir(parents=True, exist_ok=True)
    if vector_store_exists(config):
        return

    print("No Chroma database found. Starting first-time ingestion...")
    ingest_documents(config)


def run_cli() -> None:
    """Run a simple command-line chat loop for customer support questions."""

    ensure_knowledge_base()
    app = build_graph()

    print("\nCustomer Support Assistant is ready.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Customer Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            result = app.invoke({"query": query})
            print("\nAssistant Response:")
            print(
                {
                    "answer": result.get("answer", "No answer was produced."),
                    "sources": result.get("sources", []),
                }
            )
            print()
        except FileNotFoundError as exc:
            print(f"Setup error: {exc}")
            break
        except Exception as exc:
            print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    run_cli()
