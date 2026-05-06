"""
schema/resume_schema.py
========================
Pydantic v2 schemas for the resume pipeline.

Two output shapes:
  1. ResumeUISchema   — clean, flat structure for rendering in a frontend UI
  2. ResumeEmbedSchema — chunk list with metadata ready for vector DB upsert

Usage
-----
    from schema.resume_schema import build_ui_schema, build_embed_schema

    ui_data    = build_ui_schema(extracted, chunks)
    embed_data = build_embed_schema(chunks)
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# Sub-models
# ══════════════════════════════════════════════════════════════════════════════

class PersonalInfo(BaseModel):
    name:     Optional[str] = None
    email:    Optional[str] = None
    phone:    Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github:   Optional[str] = None
    website:  Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if v and not re.match(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", v, re.I):
            return None
        return v


class SkillSet(BaseModel):
    programming_languages: list[str] = Field(default_factory=list)
    web_frameworks:        list[str] = Field(default_factory=list)
    databases:             list[str] = Field(default_factory=list)
    cloud_devops:          list[str] = Field(default_factory=list)
    ml_ai:                 list[str] = Field(default_factory=list)
    data_engineering:      list[str] = Field(default_factory=list)
    cs_fundamentals:       list[str] = Field(default_factory=list)
    tools_other:           list[str] = Field(default_factory=list)
    other:                 list[str] = Field(default_factory=list)

    @property
    def all_skills(self) -> list[str]:
        out = []
        for field in self.model_fields:
            out.extend(getattr(self, field))
        return out


class ExperienceEntry(BaseModel):
    title:    Optional[str] = None
    company:  Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    bullets:  list[str]     = Field(default_factory=list)


class EducationEntry(BaseModel):
    degree:      Optional[str] = None
    field:       Optional[str] = None
    institution: Optional[str] = None
    duration:    Optional[str] = None
    year:        Optional[str] = None
    gpa:         Optional[str] = None


class CertificationEntry(BaseModel):
    name:   Optional[str] = None
    issuer: Optional[str] = None
    year:   Optional[str] = None


class ProjectEntry(BaseModel):
    name:               Optional[str] = None
    technologies:       list[str]     = Field(default_factory=list)
    description_points: list[str]     = Field(default_factory=list)
    github_url:         Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# UI Schema  — one clean object per resume for frontend rendering
# ══════════════════════════════════════════════════════════════════════════════

class ResumeUISchema(BaseModel):
    """
    Clean structured resume for UI rendering.
    Serialize with:  resume.model_dump()  or  resume.model_dump_json(indent=2)
    """
    resume_id:       str                    = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file:     str                    = ""
    processed_at:    str                    = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    personal_info:   PersonalInfo           = Field(default_factory=PersonalInfo)
    summary:         str                    = ""
    skills:          SkillSet               = Field(default_factory=SkillSet)
    experience:      list[ExperienceEntry]  = Field(default_factory=list)
    education:       list[EducationEntry]   = Field(default_factory=list)
    certifications:  list[CertificationEntry] = Field(default_factory=list)
    projects:        list[ProjectEntry]     = Field(default_factory=list)

    # Computed convenience fields for UI display
    total_experience_entries: int = 0
    total_skills:             int = 0
    top_skills:               list[str] = Field(default_factory=list)  # first 10

    def model_post_init(self, __context):
        self.total_experience_entries = len(self.experience)
        self.total_skills = len(self.skills.all_skills)
        self.top_skills   = self.skills.all_skills[:10]


# ══════════════════════════════════════════════════════════════════════════════
# Embedding chunk schema — one object per chunk for vector DB
# ══════════════════════════════════════════════════════════════════════════════

class ChunkMetadata(BaseModel):
    """Metadata stored alongside each embedding in the vector DB."""
    chunk_id:          str
    resume_id:         str
    source_file:       str
    section:           str
    chunk_index:       int
    total_chunks:      int
    token_count:       int
    char_count:        int
    # Candidate fields — used as DB filters
    candidate_name:    Optional[str] = None
    candidate_email:   Optional[str] = None
    candidate_phone:   Optional[str] = None
    candidate_location: Optional[str] = None
    linkedin:          Optional[str] = None
    github:            Optional[str] = None
    skills_summary:    Optional[str] = None
    processed_at:      str           = ""


class EmbedChunk(BaseModel):
    """
    One chunk ready for embedding + vector DB upsert.

    Workflow:
        chunk.text  -> embeddings model -> chunk.embedding
        then upsert: (chunk.chunk_id, chunk.embedding, chunk.metadata.model_dump())
    """
    chunk_id:   str
    text:       str                  # feed this to the embeddings model
    embedding:  Optional[list[float]] = None  # filled after encode()
    token_count: int
    char_count:  int
    metadata:   ChunkMetadata


class ResumeEmbedSchema(BaseModel):
    """
    Full embedding payload for one resume.
    Serialize: schema.model_dump_json(indent=2)
    """
    resume_id:    str
    source_file:  str
    processed_at: str
    total_chunks: int
    sections:     list[str]
    chunks:       list[EmbedChunk]


# ══════════════════════════════════════════════════════════════════════════════
# Builder functions
# ══════════════════════════════════════════════════════════════════════════════

def _clean_skill_list(raw: list[str]) -> list[str]:
    """
    Remove non-skill entries from a raw skill list.
    Filters out: dates, sentences, long lines, lines with bullet chars.
    """
    out = []
    date_re   = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})\b", re.I)
    long_re   = re.compile(r".{40,}")        # anything over 40 chars is probably a sentence
    bullet_re = re.compile(r"^[-•*]")
    for s in raw:
        s = s.strip()
        if not s or len(s) < 2:
            continue
        if bullet_re.match(s):
            continue
        if date_re.search(s) and len(s.split()) <= 5:
            continue
        if long_re.match(s) and "," not in s:
            continue
        # Must not look like a full sentence (verb + many words)
        if len(s.split()) > 8:
            continue
        out.append(s)
    return out


def _clean_cert(raw: dict) -> CertificationEntry:
    """Parse a certification dict — split pipe-separated names properly."""
    name_raw = raw.get("name", "") or ""

    # Pattern: "Name | Issuer | Year"
    parts = [p.strip() for p in name_raw.split("|")]

    name   = parts[0] if parts else name_raw
    issuer = parts[1] if len(parts) > 1 else raw.get("issuer")
    year   = parts[2] if len(parts) > 2 else raw.get("year")

    # Year must be 4 digits
    if year:
        m = re.search(r"\b(19|20)\d{2}\b", year)
        year = m.group(0) if m else None

    return CertificationEntry(name=name, issuer=issuer, year=year)


def _clean_project(raw: dict) -> ProjectEntry:
    """
    Build a ProjectEntry, extracting github URL and tech stack from description.
    """
    name   = raw.get("name", "")
    techs  = list(raw.get("technologies", []))
    points = list(raw.get("description_points", []))

    github_url = None
    clean_points = []

    github_re = re.compile(r"github\.com/[\w\-/]+", re.I)
    tech_line_re = re.compile(
        r"^(python|java|react|node|django|flask|fastapi|spring|go|typescript|"
        r"javascript|html|css|sql|postgresql|mongodb|redis|docker|kubernetes|"
        r"aws|gcp|azure|scikit|pandas|numpy|tensorflow|pytorch)",
        re.I,
    )

    for pt in points:
        pt = pt.strip()
        if not pt:
            continue
        # Extract github URL
        gm = github_re.search(pt)
        if gm and not github_url:
            github_url = "https://" + gm.group(0)
        # Tech-stack lines (no verb, short, comma-separated) -> add to techs
        if tech_line_re.match(pt) and len(pt.split()) <= 12:
            new_techs = [t.strip() for t in pt.split(",") if t.strip()]
            for t in new_techs:
                if t not in techs:
                    techs.append(t)
            continue
        # Skip duplicate project name lines and github lines
        if pt.lower().startswith("github.com") or (github_re.match(pt) and len(pt) < 60):
            continue
        clean_points.append(pt)

    return ProjectEntry(
        name=name,
        technologies=techs,
        description_points=clean_points,
        github_url=github_url,
    )


def build_ui_schema(extracted: dict, resume_id: str | None = None) -> ResumeUISchema:
    """
    Convert the raw extracted dict (Step 3 output) into a clean ResumeUISchema.

    Parameters
    ----------
    extracted   : dict from information_extraction/extractor.py
    resume_id   : optional stable ID; auto-generated UUID if not provided

    Returns
    -------
    ResumeUISchema  — call .model_dump() or .model_dump_json() to serialise
    """
    rid = resume_id or str(uuid.uuid4())

    # ── Personal info ────────────────────────────────────────────────────────
    pi_raw = extracted.get("personal_info", {})
    personal = PersonalInfo(
        name     = pi_raw.get("name"),
        email    = pi_raw.get("email"),
        phone    = pi_raw.get("phone"),
        location = pi_raw.get("location"),
        linkedin = pi_raw.get("linkedin"),
        github   = pi_raw.get("github"),
        website  = pi_raw.get("website"),
    )

    # ── Skills — clean raw list before categorising ──────────────────────────
    skills_raw = extracted.get("skills", {})
    raw_list   = _clean_skill_list(skills_raw.get("raw_list", []))
    cat_raw    = skills_raw.get("categorized", {})

    def _clean_cat(lst):
        return _clean_skill_list(lst)

    skills = SkillSet(
        programming_languages = _clean_cat(cat_raw.get("programming_languages", [])),
        web_frameworks        = _clean_cat(cat_raw.get("web_frameworks", [])),
        databases             = _clean_cat(cat_raw.get("databases", [])),
        cloud_devops          = _clean_cat(cat_raw.get("cloud_devops", [])),
        ml_ai                 = _clean_cat(cat_raw.get("ml_ai", [])),
        data_engineering      = _clean_cat(cat_raw.get("data_engineering", [])),
        cs_fundamentals       = _clean_cat(cat_raw.get("cs_fundamentals", [])),
        tools_other           = _clean_cat(cat_raw.get("tools_other", [])),
        other                 = _clean_cat(cat_raw.get("other", [])),
    )

    # ── Experience ───────────────────────────────────────────────────────────
    experience = []
    for e in extracted.get("experience", []):
        # Extract location from duration line  e.g. "March 2021 - Present | San Francisco, CA"
        dur   = e.get("duration", "")
        loc   = None
        if "|" in (dur or ""):
            loc_part = dur.split("|", 1)[1].strip()
            dur      = dur.split("|", 1)[0].strip()
            if loc_part and not re.search(r"\d{4}", loc_part):
                loc = loc_part

        experience.append(ExperienceEntry(
            title    = e.get("title"),
            company  = e.get("company"),
            location = loc or e.get("location"),
            duration = dur,
            bullets  = [b for b in e.get("bullets", []) if b and len(b) > 5],
        ))

    # ── Education ────────────────────────────────────────────────────────────
    education = [
        EducationEntry(
            degree      = e.get("degree"),
            field       = e.get("field"),
            institution = e.get("institution"),
            duration    = e.get("duration"),
            year        = e.get("year"),
            gpa         = e.get("gpa"),
        )
        for e in extracted.get("education", [])
    ]

    # ── Certifications ───────────────────────────────────────────────────────
    certifications = [_clean_cert(c) for c in extracted.get("certifications", [])]

    # ── Projects ─────────────────────────────────────────────────────────────
    projects = [_clean_project(p) for p in extracted.get("projects", [])]

    return ResumeUISchema(
        resume_id      = rid,
        source_file    = extracted.get("source_file", ""),
        personal_info  = personal,
        summary        = extracted.get("summary", ""),
        skills         = skills,
        experience     = experience,
        education      = education,
        certifications = certifications,
        projects       = projects,
    )


def build_embed_schema(chunks_result: dict, resume_id: str | None = None) -> ResumeEmbedSchema:
    """
    Convert the Step 4 chunks JSON into a typed ResumeEmbedSchema.

    Parameters
    ----------
    chunks_result : dict from tokenizer_chunking/chunker.py
    resume_id     : must match the ID used in build_ui_schema for same resume

    Returns
    -------
    ResumeEmbedSchema  — .chunks[i].text feeds the embedding model
                       — .chunks[i].metadata goes into the vector DB
    """
    rid  = resume_id or str(uuid.uuid4())
    meta = chunks_result.get("metadata", {})
    now  = datetime.now(timezone.utc).isoformat()

    embed_chunks = []
    for c in chunks_result.get("chunks", []):
        cm = c.get("metadata", {})
        embed_chunks.append(EmbedChunk(
            chunk_id    = c["chunk_id"],
            text        = c["text"],
            token_count = c.get("token_count", 0),
            char_count  = c.get("char_count", len(c.get("text",""))),
            metadata    = ChunkMetadata(
                chunk_id           = c["chunk_id"],
                resume_id          = rid,
                source_file        = cm.get("source_file", ""),
                section            = cm.get("section", ""),
                chunk_index        = cm.get("chunk_index", 0),
                total_chunks       = cm.get("total_chunks_in_section", 1),
                token_count        = c.get("token_count", 0),
                char_count         = c.get("char_count", 0),
                candidate_name     = cm.get("candidate_name"),
                candidate_email    = cm.get("candidate_email"),
                candidate_phone    = cm.get("candidate_phone"),
                candidate_location = cm.get("candidate_location"),
                linkedin           = cm.get("linkedin"),
                github             = cm.get("github"),
                skills_summary     = cm.get("skills_summary"),
                processed_at       = cm.get("processed_at", now),
            )
        ))

    return ResumeEmbedSchema(
        resume_id    = rid,
        source_file  = meta.get("source_file", ""),
        processed_at = meta.get("processed_at", now),
        total_chunks = len(embed_chunks),
        sections     = chunks_result.get("sections_processed", []),
        chunks       = embed_chunks,
    )