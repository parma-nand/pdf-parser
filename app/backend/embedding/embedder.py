"""
embedding/embedder.py
======================
Step 5 – Generate embeddings and save/upsert to a vector DB.

Supported backends (install what you need):
    Local  : sentence-transformers  →  pip install sentence-transformers
    Cloud  : OpenAI                 →  pip install openai
    Vector DB: ChromaDB             →  pip install chromadb
    Vector DB: Pinecone             →  pip install pinecone-client

Usage
-----
    from embedding.embedder import Embedder

    emb = Embedder(model="local")          # uses all-MiniLM-L6-v2
    emb = Embedder(model="openai",
                   openai_api_key="sk-...")

    embed_schema  = emb.encode(embed_schema)        # adds .embedding to each chunk
    emb.save_json(embed_schema, "embedding/resume_embeddings.json")
    emb.upsert_chroma(embed_schema)                 # load into ChromaDB
    emb.upsert_pinecone(embed_schema, index)        # load into Pinecone
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from schema.resume_schema import ResumeEmbedSchema, EmbedChunk


# ══════════════════════════════════════════════════════════════════════════════
# Embedder class
# ══════════════════════════════════════════════════════════════════════════════

class Embedder:
    """
    Encodes text chunks into dense vector embeddings and
    provides helpers to persist / upsert them.

    Parameters
    ----------
    model          : "local" (sentence-transformers) | "openai"
    model_name     : specific model string  (default auto-selected)
    openai_api_key : required when model="openai"
    batch_size     : how many chunks to encode at once
    """

    LOCAL_MODEL_DEFAULT  = "all-MiniLM-L6-v2"        # 80MB, 384-dim, great quality/speed
    LOCAL_MODEL_LARGE    = "all-mpnet-base-v2"         # 420MB, 768-dim, best quality
    OPENAI_MODEL_DEFAULT = "text-embedding-3-small"   # 1536-dim, cheap

    def __init__(
        self,
        model:          Literal["local", "openai"] = "local",
        model_name:     Optional[str] = None,
        openai_api_key: Optional[str] = None,
        batch_size:     int = 32,
    ):
        self.backend    = model
        self.batch_size = batch_size
        self._model     = None   # lazy-loaded

        if model == "local":
            self.model_name = model_name or self.LOCAL_MODEL_DEFAULT
        else:
            self.model_name    = model_name or self.OPENAI_MODEL_DEFAULT
            self.openai_api_key = (
                openai_api_key
                or os.getenv("OPENAI_API_KEY")
                or ""
            )

    def _load_local(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Run:  pip install sentence-transformers"
                )
            print(f"  Loading model '{self.model_name}' ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode_local(self, texts: list[str]) -> list[list[float]]:
        model = self._load_local()
        vecs  = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def _encode_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run:  pip install openai")
        client = OpenAI(api_key=self.openai_api_key)
        all_vecs = []
        # OpenAI recommends batches of ≤ 2048
        for i in range(0, len(texts), 100):
            batch    = texts[i:i+100]
            response = client.embeddings.create(model=self.model_name, input=batch)
            all_vecs.extend([item.embedding for item in response.data])
        return all_vecs

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, schema: ResumeEmbedSchema) -> ResumeEmbedSchema:
        """
        Generate embeddings for every chunk in schema.
        Adds .embedding to each EmbedChunk in-place and returns the schema.
        """
        texts = [c.text for c in schema.chunks]
        print(f"  Encoding {len(texts)} chunks [{self.backend}:{self.model_name}] ...")

        if self.backend == "local":
            vecs = self._encode_local(texts)
        else:
            vecs = self._encode_openai(texts)

        for chunk, vec in zip(schema.chunks, vecs):
            chunk.embedding = vec

        print(f"  Done. Embedding dim = {len(vecs[0]) if vecs else 0}")
        return schema

    def save_json(
        self,
        schema:      ResumeEmbedSchema,
        output_path: str | Path,
    ) -> Path:
        """
        Save the full schema (chunks + embeddings) as JSON.
        This file can be used as a portable embedding store or
        loaded later for Pinecone/FAISS bulk upsert.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(schema.model_dump_json(indent=2))
        print(f"  Saved embeddings JSON -> {out}")
        return out

    def upsert_chroma(
        self,
        schema:          ResumeEmbedSchema,
        collection_name: str  = "resumes",
        persist_dir:     str  = "./chroma_db",
    ):
        """
        Upsert all chunks into a local ChromaDB collection.
        install: pip install chromadb
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError("Run:  pip install chromadb")

        client     = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name     = collection_name,
            metadata = {"hnsw:space": "cosine"},
        )

        ids        = [c.chunk_id for c in schema.chunks]
        texts      = [c.text     for c in schema.chunks]
        embeddings = [c.embedding for c in schema.chunks]
        metadatas  = [c.metadata.model_dump() for c in schema.chunks]

        # ChromaDB requires metadata values to be str/int/float/bool
        clean_metas = []
        for m in metadatas:
            clean_metas.append({
                k: (str(v) if v is not None else "")
                for k, v in m.items()
            })

        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = clean_metas,
        )
        print(f"  Upserted {len(ids)} chunks -> ChromaDB '{collection_name}' at {persist_dir}")
        return collection

    def upsert_pinecone(
        self,
        schema:    ResumeEmbedSchema,
        index,                          # pinecone.Index object
        namespace: str = "resumes",
    ):
        """
        Upsert all chunks into a Pinecone index.
        install: pip install pinecone-client
        Usage:
            import pinecone
            pc    = pinecone.Pinecone(api_key="YOUR_KEY")
            index = pc.Index("resume-index")
            emb.upsert_pinecone(schema, index)
        """
        vectors = [
            {
                "id":       c.chunk_id,
                "values":   c.embedding,
                "metadata": {
                    k: (str(v) if v is not None else "")
                    for k, v in c.metadata.model_dump().items()
                },
            }
            for c in schema.chunks
            if c.embedding
        ]
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"  Upserted {len(vectors)} vectors -> Pinecone namespace '{namespace}'")

    def search_chroma(
        self,
        query:           str,
        collection,
        n_results:       int = 5,
        filter_section:  Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search in ChromaDB.
        Returns list of {text, score, metadata} dicts.
        """
        query_vec = self._encode_local([query])[0] \
            if self.backend == "local" \
            else self._encode_openai([query])[0]

        where = {"section": filter_section} if filter_section else None

        results = collection.query(
            query_embeddings = [query_vec],
            n_results        = n_results,
            include          = ["documents", "distances", "metadatas"],
            where            = where,
        )

        out = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            out.append({
                "text":     doc,
                "score":    round(1 - dist, 4),   # cosine similarity
                "metadata": meta,
            })
        return out