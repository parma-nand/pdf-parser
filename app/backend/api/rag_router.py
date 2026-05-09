"""
api/rag_router.py
==================
API 2 — RAG (Retrieval-Augmented Generation)

POST /api/rag/ask          — ask any question about the resume
POST /api/rag/match        — match resume against a job description
POST /api/rag/summary      — generate AI professional summary
POST /api/rag/interview    — generate interview questions for a role
POST /api/rag/search       — raw semantic search (no LLM, just retrieval)
GET  /api/rag/status       — check if RAG is loaded

Flow
----
1. Call POST /api/resume/parse  first  (stores embed_schema.json to disk)
2. Call any /api/rag/* endpoint (auto-loads ChromaDB from embed_schema.json)
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.models import (
    AskRequest, MatchRequest, InterviewRequest,
    SearchRequest, RAGResponse, SearchResponse, SearchResult,
)

router   = APIRouter(prefix="/api/rag", tags=["RAG"])
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Singleton RAG instance ────────────────────────────────────────────────────
# RAG is expensive to initialise (builds ChromaDB, loads embedding model).
# We keep one instance alive for the lifetime of the server process.

_rag_instance = None


def _get_rag():
    """
    Return the singleton ResumeRAG instance.
    Rebuilds ChromaDB every time a new resume is parsed
    (embed_schema.json gets overwritten by the parser API).
    """
    global _rag_instance
    import sys
    sys.path.insert(0, str(BASE_DIR))

    embed_schema_path = BASE_DIR / "schema" / "embed_schema.json"
    if not embed_schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No resume found. "
                "Call POST /api/resume/parse first to upload a resume."
            )
        )

    from rag.resume_rag import ResumeRAG
    # Always rebuild so the RAG reflects the latest parsed resume
    _rag_instance = ResumeRAG(k=5, rebuild_store=True)
    _rag_instance.load(embed_schema_path=embed_schema_path)
    return _rag_instance


def _reset_rag():
    """Called after a new resume is parsed to force RAG rebuild on next query."""
    global _rag_instance
    _rag_instance = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def rag_status():
    """Check if a resume has been parsed and RAG is ready to query."""
    embed_schema = BASE_DIR / "schema" / "embed_schema.json"
    ui_schema    = BASE_DIR / "schema" / "ui_schema.json"

    if not embed_schema.exists():
        return {
            "status":  "not_ready",
            "message": "No resume uploaded yet. Call POST /api/resume/parse first."
        }

    import json
    ui = json.loads(ui_schema.read_text()) if ui_schema.exists() else {}
    pi = ui.get("personal_info", {})
    return {
        "status":        "ready",
        "candidate":     pi.get("name", "unknown"),
        "resume_id":     ui.get("resume_id"),
        "total_chunks":  json.loads(embed_schema.read_text()).get("total_chunks", 0),
        "message":       "RAG is ready. You can now query the resume."
    }


@router.post("/ask", response_model=RAGResponse)
async def ask(req: AskRequest):
    """
    Ask any natural language question about the uploaded resume.

    Example request body:
        { "question": "What is the candidate's current company?" }
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        rag    = _get_rag()
        answer = rag._qa_chain.invoke(req.question)
        return RAGResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match", response_model=RAGResponse)
async def match_skills(req: MatchRequest):
    """
    Compare the resume against a job description.
    Returns matched skills, missing skills, match score, and recommendation.

    Example request body:
        { "job_description": "We need Python, AWS, Kafka, 5+ years backend" }
    """
    if not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    try:
        rag    = _get_rag()
        result = rag._skill_chain.invoke(req.job_description)
        return RAGResponse(answer=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary", response_model=RAGResponse)
async def generate_summary():
    """
    Generate an AI-written professional summary from the resume content.
    This is LLM-generated — not copied from the written summary section.
    """
    try:
        rag     = _get_rag()
        summary = rag._summary_chain.invoke("Generate a professional summary")
        return RAGResponse(answer=summary)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interview", response_model=RAGResponse)
async def interview_questions(req: InterviewRequest):
    """
    Generate 5 tailored interview questions for a specific role.

    Example request body:
        { "role": "Senior Full Stack Engineer" }
    """
    if not req.role.strip():
        raise HTTPException(status_code=400, detail="Role cannot be empty.")
    try:
        rag       = _get_rag()
        questions = rag._interview_chain.invoke(req.role)
        return RAGResponse(answer=questions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def semantic_search(req: SearchRequest):
    """
    Raw semantic search — returns top-k matching chunks without LLM generation.
    Useful for debugging or building custom UIs.

    Example request body:
        { "query": "machine learning experience", "k": 5, "section": "experience" }
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        rag     = _get_rag()
        # Override k if specified
        results = rag._vector_store.similarity_search_with_score(
            req.query, k=req.k,
            filter={"section": req.section} if req.section else None,
        )
        return SearchResponse(
            query   = req.query,
            results = [
                SearchResult(
                    section = doc.metadata.get("section", ""),
                    score   = round(1 - float(dist), 4),
                    text    = doc.page_content,
                )
                for doc, dist in results
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))