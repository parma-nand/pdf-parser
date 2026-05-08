"""
rag/vector_store.py
====================
Builds and manages the ChromaDB vector store from resume chunks.

Flow
----
1. Load embed_schema.json (chunks produced by Steps 1-5)
2. Use HuggingFace sentence-transformers for embeddings
3. Store in ChromaDB with full metadata
4. Return a LangChain retriever for use in the RAG chain
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings


BASE_DIR = Path(__file__).resolve().parent.parent


def _chunks_to_documents(embed_schema_path: Path) -> list[Document]:
    """
    Convert embed_schema.json chunks into LangChain Document objects.
    Each Document carries the chunk text + metadata for filtering.
    """
    data   = json.loads(embed_schema_path.read_text(encoding="utf-8"))
    docs   = []

    for chunk in data.get("chunks", []):
        meta = chunk.get("metadata", {})
        # Flatten metadata — ChromaDB requires str/int/float/bool values only
        clean_meta = {
            k: (str(v) if v is not None else "")
            for k, v in meta.items()
        }
        docs.append(Document(
            page_content = chunk["text"],
            metadata     = clean_meta,
        ))
    return docs


def build_vector_store(
    embed_schema_path: str | Path | None = None,
    persist_dir:       str | None        = None,
    collection_name:   str | None        = None,
) -> Chroma:
    """
    Build (or load) a ChromaDB vector store from embed_schema.json.

    Parameters
    ----------
    embed_schema_path : path to schema/embed_schema.json
                        (defaults to BASE_DIR/schema/embed_schema.json)
    persist_dir       : where ChromaDB stores data on disk
    collection_name   : ChromaDB collection name

    Returns
    -------
    Chroma vector store instance
    """
    embed_schema_path = Path(embed_schema_path or (BASE_DIR / "schema" / "embed_schema.json"))
    persist_dir       = persist_dir       or settings.chroma_persist_dir
    collection_name   = collection_name   or settings.chroma_collection

    if not embed_schema_path.exists():
        raise FileNotFoundError(
            f"embed_schema.json not found at {embed_schema_path}.\n"
            "Run the pipeline first:  python run_pipeline.py <resume.pdf>"
        )

    print(f"  Building vector store from {embed_schema_path.name} ...")
    docs       = _chunks_to_documents(embed_schema_path)
    embeddings = settings.get_embeddings()

    vector_store = Chroma.from_documents(
        documents       = docs,
        embedding       = embeddings,
        collection_name = collection_name,
        persist_directory = persist_dir,
    )

    print(f"  ChromaDB ready — {len(docs)} chunks in '{collection_name}'")
    return vector_store


def load_vector_store(
    persist_dir:     str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    Use this instead of build_vector_store() when the DB already exists.
    """
    persist_dir     = persist_dir     or settings.chroma_persist_dir
    collection_name = collection_name or settings.chroma_collection
    embeddings      = settings.get_embeddings()

    return Chroma(
        collection_name   = collection_name,
        embedding_function = embeddings,
        persist_directory  = persist_dir,
    )


def get_retriever(
    vector_store:  Chroma,
    k:             int         = 5,
    section_filter: str | None = None,
) -> object:
    """
    Return a LangChain retriever.

    Parameters
    ----------
    k              : number of chunks to retrieve per query
    section_filter : optionally restrict to a section
                     e.g. "experience", "skills", "education"
    """
    search_kwargs: dict = {"k": k}
    if section_filter:
        search_kwargs["filter"] = {"section": section_filter}

    return vector_store.as_retriever(
        search_type   = "similarity",
        search_kwargs = search_kwargs,
    )