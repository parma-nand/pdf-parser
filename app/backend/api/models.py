"""
api/models.py
==============
Pydantic request / response models for all API endpoints.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


# ── Resume Parser API ─────────────────────────────────────────────────────────

class PersonalInfoResponse(BaseModel):
    name:     Optional[str] = None
    email:    Optional[str] = None
    phone:    Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github:   Optional[str] = None
    website:  Optional[str] = None


class SkillsResponse(BaseModel):
    programming_languages: list[str] = []
    web_frameworks:        list[str] = []
    databases:             list[str] = []
    cloud_devops:          list[str] = []
    ml_ai:                 list[str] = []
    data_engineering:      list[str] = []
    cs_fundamentals:       list[str] = []
    tools_other:           list[str] = []
    other:                 list[str] = []
    all_skills:            list[str] = []


class ExperienceResponse(BaseModel):
    title:    Optional[str] = None
    company:  Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    bullets:  list[str]     = []


class EducationResponse(BaseModel):
    degree:      Optional[str] = None
    field:       Optional[str] = None
    institution: Optional[str] = None
    duration:    Optional[str] = None
    year:        Optional[str] = None
    gpa:         Optional[str] = None


class CertificationResponse(BaseModel):
    name:   Optional[str] = None
    issuer: Optional[str] = None
    year:   Optional[str] = None


class ProjectResponse(BaseModel):
    name:               Optional[str] = None
    technologies:       list[str]     = []
    description_points: list[str]     = []
    github_url:         Optional[str] = None


class ParseResumeResponse(BaseModel):
    """Full response from POST /api/resume/parse"""
    resume_id:       str
    source_file:     str
    processed_at:    str
    personal_info:   PersonalInfoResponse
    summary:         str
    skills:          SkillsResponse
    experience:      list[ExperienceResponse]
    education:       list[EducationResponse]
    certifications:  list[CertificationResponse]
    projects:        list[ProjectResponse]
    total_skills:    int
    top_skills:      list[str]
    # Pipeline metadata
    sections_found:  list[str]
    total_words:     int
    total_chunks:    int


# ── RAG API ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    resume_id: Optional[str] = None   # for future multi-resume support

    model_config = {"json_schema_extra": {
        "example": {"question": "What is the candidate's current role?"}
    }}


class MatchRequest(BaseModel):
    job_description: str
    resume_id: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "job_description": "We need a Python backend engineer with AWS and Kafka experience."
        }
    }}


class InterviewRequest(BaseModel):
    role: str
    resume_id: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {"role": "Senior Backend Engineer"}
    }}


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    section: Optional[str] = None   # filter by section e.g. "experience"

    model_config = {"json_schema_extra": {
        "example": {"query": "machine learning experience", "k": 5}
    }}


class RAGResponse(BaseModel):
    """Generic RAG text response"""
    answer:    str
    resume_id: Optional[str] = None


class SearchResult(BaseModel):
    section: str
    score:   float
    text:    str


class SearchResponse(BaseModel):
    query:   str
    results: list[SearchResult]


class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None