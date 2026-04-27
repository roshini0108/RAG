from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Store all configurable values used across the project."""

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "support_docs")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    # Higher overlap helps preserve context across neighboring chunks.
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "300"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # Only escalate when the best Chroma distance is clearly poor.
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def get_config() -> AppConfig:
    """Return the shared application configuration object."""

    return AppConfig()
