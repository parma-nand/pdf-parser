# PDF Parser — Backend

A modular Python backend for parsing, processing, and querying PDF documents using a RAG (Retrieval-Augmented Generation) pipeline.

---

## 📁 Folder Structure

```
app/backend/
│
├── api/                    # FastAPI route definitions and request/response handlers
│
├── config/                 # Configuration files (env settings, constants, model configs)
│
├── embedding/              # Text embedding logic (e.g., sentence-transformers, OpenAI embeddings)
│
├── information_extraction/ # Extracts structured data from parsed PDF text (NER, sections, fields)
│
├── normalize/              # Text normalization utilities (cleaning, deduplication, formatting)
│
├── rag/                    # RAG pipeline: retrieval, reranking, and LLM-augmented generation
│
├── schema/                 # Pydantic models and data schemas for request/response validation
│
├── section_divide/         # Logic to detect and split PDF content into named sections
│
├── tokenizer_chunking/     # Chunking strategies for splitting text before embedding
│
├── main.py                 # FastAPI app entry point — registers routers and starts the server
├── run_pipeline.py         # Standalone script to run the full parsing pipeline end-to-end
└── run_rag.py              # Standalone script to run the RAG query pipeline
```

---

## 🔄 Pipeline Flow

```
PDF Input
   │
   ▼
section_divide      →  Splits document into logical sections (Introduction, Experience, etc.)
   │
   ▼
normalize           →  Cleans raw text (whitespace, encoding artifacts, bullet glyphs)
   │
   ▼
information_extraction  →  Pulls structured fields (name, skills, companies, dates)
   │
   ▼
tokenizer_chunking  →  Chunks text into embedding-friendly segments
   │
   ▼
embedding           →  Generates vector embeddings for each chunk
   │
   ▼
rag                 →  Retrieves relevant chunks + generates LLM responses
   │
   ▼
api                 →  Exposes endpoints for upload, query, and document management
```

---

## 🚀 Entry Points

| File | Purpose |
|------|---------|
| `main.py` | Start the FastAPI server (`uvicorn app.backend.main:app`) |
| `run_pipeline.py` | Run the full PDF → embed pipeline on a file |
| `run_rag.py` | Run a RAG query against already-processed documents |

---

## ⚙️ Key Modules

### `api/`
FastAPI routers. Likely includes endpoints for:
- `POST /upload` — ingest a PDF
- `POST /query` — ask a question against the document
- `GET /documents` — list processed documents

### `config/`
Centralised settings — model names, vector DB connection, chunk sizes, API keys (via `.env`).

### `embedding/`
Wraps embedding model calls (e.g., `all-MiniLM-L6-v2`). Produces float vectors per chunk.

### `rag/`
Core retrieval logic — vector search (FAISS/Qdrant/ChromaDB), optional BM25 hybrid retrieval, CrossEncoder reranking, and LLM call for final answer generation.

### `schema/`
Pydantic models for type-safe data handling across the pipeline.

### `tokenizer_chunking/`
Implements chunking strategies — rule-based (paragraph/line-break splits) or token-length-based sliding windows.

---

## 🛠️ Tech Stack

- **FastAPI** — REST API framework
- **PyMuPDF / pdfplumber** — PDF text extraction
- **sentence-transformers** — Embedding generation
- **Qdrant / FAISS / ChromaDB** — Vector storage (depending on config)
- **OpenAI / LLaMA** — LLM for RAG generation
- **Pydantic** — Schema validation
- **Docker Compose** — Service orchestration

---

## 📝 Notes

- Run `run_pipeline.py` first to process and index a PDF before querying.
- All configuration (model names, DB URIs, API keys) should be set in `config/` or a `.env` file at the project root.
- The `schema/` module is the single source of truth for data shapes — update it before changing pipeline inputs/outputs.
