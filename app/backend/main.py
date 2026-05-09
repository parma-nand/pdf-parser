"""
main.py  -  FastAPI entry point

Start:
    python main.py
    uvicorn main:app --reload --port 8000

Swagger docs:  http://localhost:8000/docs

Flow:
    1. POST /api/resume/parse   upload PDF, get structured JSON
    2. POST /api/rag/ask        ask a question about the resume
    2. POST /api/rag/match      match resume vs job description
    2. POST /api/rag/summary    generate professional summary
    2. POST /api/rag/interview  generate interview questions
    2. POST /api/rag/search     raw semantic search
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.resume_router import router as resume_router
from api.rag_router    import router as rag_router

app = FastAPI(
    title   = "Resume Parser + RAG API",
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(resume_router)
app.include_router(rag_router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Resume Parser + RAG API is running.",
        "docs":    "http://localhost:8000/docs",
        "endpoints": {
            "parse_resume":        "POST /api/resume/parse",
            "ask_question":        "POST /api/rag/ask",
            "match_job":           "POST /api/rag/match",
            "generate_summary":    "POST /api/rag/summary",
            "interview_questions": "POST /api/rag/interview",
            "semantic_search":     "POST /api/rag/search",
            "rag_status":          "GET  /api/rag/status",
            "parser_status":       "GET  /api/resume/status",
        }
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)