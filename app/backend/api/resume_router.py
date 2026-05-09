"""
api/resume_router.py
=====================
API 1 — Resume Parser

POST /api/resume/parse
    Upload a PDF resume → returns full structured JSON

GET  /api/resume/status
    Health check — confirms parser is ready
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from api.models import ParseResumeResponse

router = APIRouter(prefix="/api/resume", tags=["Resume Parser"])

BASE_DIR = Path(__file__).resolve().parent.parent


def _run_pipeline_on_bytes(pdf_bytes: bytes, filename: str) -> dict:
    """
    Write PDF to a temp file, run the full pipeline, return ui_schema dict.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    from section_divide.extractor         import extract_and_divide
    from normalize.normalizer             import normalize_sections
    from information_extraction.extractor import extract_information
    from tokenizer_chunking.chunker       import tokenize_and_chunk
    from schema.resume_schema             import build_ui_schema, build_embed_schema

    OUTPUT_DIRS = {
        "section_divide":         BASE_DIR / "section_divide",
        "normalize":              BASE_DIR / "normalize",
        "information_extraction": BASE_DIR / "information_extraction",
        "tokenizer_chunking":     BASE_DIR / "tokenizer_chunking",
        "schema":                 BASE_DIR / "schema",
    }
    for d in OUTPUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()

    # Write PDF to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        stem = Path(filename).stem

        sections   = extract_and_divide(tmp_path)
        normalized = normalize_sections(sections)
        extracted  = extract_information(normalized)
        chunks     = tokenize_and_chunk(normalized, extracted, stem)
        ui_schema  = build_ui_schema(extracted)
        embed_schema = build_embed_schema(chunks, resume_id=ui_schema.resume_id)

        # Persist outputs
        (OUTPUT_DIRS["section_divide"]         / "sections.json").write_text(
            json.dumps(sections,   indent=2, ensure_ascii=False))
        (OUTPUT_DIRS["normalize"]              / "normalized.json").write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False))
        (OUTPUT_DIRS["information_extraction"] / "extracted.json").write_text(
            json.dumps(extracted,  indent=2, ensure_ascii=False))
        (OUTPUT_DIRS["tokenizer_chunking"]     / "chunks.json").write_text(
            json.dumps(chunks,     indent=2, ensure_ascii=False))
        (OUTPUT_DIRS["schema"] / "ui_schema.json").write_text(
            ui_schema.model_dump_json(indent=2))
        (OUTPUT_DIRS["schema"] / "embed_schema.json").write_text(
            embed_schema.model_dump_json(indent=2))

        return {
            "ui":    ui_schema,
            "meta": {
                "sections_found": list(sections["sections"].keys()),
                "total_words":    normalized["total_words"],
                "total_chunks":   embed_schema.total_chunks,
            }
        }
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status")
async def parser_status():
    """Check if the resume parser is ready."""
    return {"status": "ready", "message": "Resume parser is running."}


@router.post("/parse", response_model=ParseResumeResponse)
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload a PDF resume and get back structured JSON.

    - Accepts: multipart/form-data with field name 'file'
    - Returns: full structured resume with personal info, skills,
               experience, education, certifications, projects, summary

    The parsed resume is also saved locally so the RAG API can
    use it immediately after this call.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result   = _run_pipeline_on_bytes(pdf_bytes, file.filename)
        ui       = result["ui"]
        meta     = result["meta"]
        ui_dict  = ui.model_dump()

        return ParseResumeResponse(
            resume_id      = ui.resume_id,
            source_file    = file.filename,
            processed_at   = ui.processed_at,
            personal_info  = ui_dict["personal_info"],
            summary        = ui.summary,
            skills         = {
                **ui_dict["skills"],
                "all_skills": ui.skills.all_skills,
            },
            experience     = ui_dict["experience"],
            education      = ui_dict["education"],
            certifications = ui_dict["certifications"],
            projects       = ui_dict["projects"],
            total_skills   = ui.total_skills,
            top_skills     = ui.top_skills,
            sections_found = meta["sections_found"],
            total_words    = meta["total_words"],
            total_chunks   = meta["total_chunks"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")