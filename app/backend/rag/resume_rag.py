"""
rag/resume_rag.py
==================
Main ResumeRAG class — single entry point for all RAG operations.

Usage
-----
    from rag.resume_rag import ResumeRAG

    rag = ResumeRAG()
    rag.load("schema/embed_schema.json")   # builds ChromaDB + chains

    # General question
    print(rag.ask("What is the candidate's most recent job?"))

    # Skill match against a JD
    print(rag.match_skills("We need a Python + AWS + Kafka engineer..."))

    # Auto-generate professional summary
    print(rag.generate_summary())

    # Tailored interview questions
    print(rag.interview_questions("Senior Backend Engineer"))
"""

from __future__ import annotations

from pathlib import Path

from rag.vector_store import build_vector_store, load_vector_store, get_retriever
from rag.chains import (
    build_qa_chain,
    build_skill_match_chain,
    build_summary_chain,
    build_interview_chain,
)
from config.settings import settings


class ResumeRAG:
    """
    One-stop class for resume RAG operations.

    Parameters
    ----------
    k              : number of chunks retrieved per query (default 5)
    rebuild_store  : if True, always rebuild ChromaDB from embed_schema.json
                     if False, load existing ChromaDB if it exists
    """

    def __init__(self, k: int = 5, rebuild_store: bool = True):
        self.k             = k
        self.rebuild_store = rebuild_store
        self._vector_store = None
        self._retriever    = None

        # Lazy-built chains
        self._qa_chain       = None
        self._skill_chain    = None
        self._summary_chain  = None
        self._interview_chain = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(
        self,
        embed_schema_path: str | Path | None = None,
        persist_dir:       str | None        = None,
        collection_name:   str | None        = None,
    ) -> "ResumeRAG":
        """
        Build or load the vector store and initialise all chains.
        Call this once before using ask() / match_skills() etc.
        """
        settings.validate()

        print("\nLoading ResumeRAG ...")

        if self.rebuild_store:
            self._vector_store = build_vector_store(
                embed_schema_path = embed_schema_path,
                persist_dir       = persist_dir,
                collection_name   = collection_name,
            )
        else:
            self._vector_store = load_vector_store(
                persist_dir     = persist_dir,
                collection_name = collection_name,
            )

        self._retriever = get_retriever(self._vector_store, k=self.k)

        # Build all chains once
        self._qa_chain        = build_qa_chain(self._retriever)
        self._skill_chain     = build_skill_match_chain(self._retriever)
        self._summary_chain   = build_summary_chain(self._retriever)
        self._interview_chain = build_interview_chain(self._retriever)

        print("  ResumeRAG ready.\n")
        return self

    # ── Public methods ────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """
        Ask any question about the resume.

        Examples
        --------
        rag.ask("What programming languages does the candidate know?")
        rag.ask("How many years of experience does the candidate have?")
        rag.ask("What is the candidate's highest degree?")
        """
        self._check_loaded()
        print(f"\nQ: {question}")
        answer = self._qa_chain.invoke(question)
        print(f"A: {answer}")
        return answer

    def match_skills(self, job_description: str) -> str:
        """
        Compare the candidate's skills against a job description.

        Example
        -------
        rag.match_skills('''
            We are looking for a Senior Backend Engineer with:
            - 5+ years Python experience
            - AWS (EC2, Lambda, S3)
            - Kafka / Redis for streaming
            - PostgreSQL or MongoDB
            - Docker + Kubernetes
        ''')
        """
        self._check_loaded()
        print("\nRunning skill match ...")
        result = self._skill_chain.invoke(job_description)
        print(result)
        return result

    def generate_summary(self) -> str:
        """
        Generate a professional summary from the resume content.
        This is LLM-generated — not copied from the resume.
        """
        self._check_loaded()
        print("\nGenerating professional summary ...")
        summary = self._summary_chain.invoke("Generate a professional summary")
        print(summary)
        return summary

    def interview_questions(self, role: str) -> str:
        """
        Generate 5 tailored interview questions for a given role.

        Example
        -------
        rag.interview_questions("Senior Full Stack Engineer")
        """
        self._check_loaded()
        print(f"\nGenerating interview questions for: {role}")
        questions = self._interview_chain.invoke(role)
        print(questions)
        return questions

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """
        Raw semantic search — returns top-k matching chunks with scores.
        Useful for debugging retrieval quality.
        """
        self._check_loaded()
        results = self._vector_store.similarity_search_with_score(
            query, k=k or self.k
        )
        output = []
        for doc, score in results:
            output.append({
                "section": doc.metadata.get("section", ""),
                "score":   round(float(score), 4),
                "text":    doc.page_content[:200],
            })
        return output

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_loaded(self):
        if self._vector_store is None:
            raise RuntimeError(
                "Call rag.load() before using RAG methods."
            )