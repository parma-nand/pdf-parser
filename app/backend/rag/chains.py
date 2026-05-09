"""
rag/chains.py
==============
LangChain RAG chains for resume question-answering.

Three chains are provided:
  1. ResumeQAChain       — general Q&A over the whole resume
  2. SkillMatchChain     — compare resume skills against a job description
  3. SummaryChain        — generate a professional summary from the resume

All chains use ChatOpenAI (gpt-4o-mini) configured via config/settings.py.
"""

from __future__ import annotations


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from config.settings import settings


# ══════════════════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════════════════

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert HR assistant and resume analyst.
Use ONLY the resume context below to answer the question.
If the answer is not in the context, say "Not found in the resume."

Resume Context:
{context}

Question: {question}

Answer (be concise and factual):"""
)

_SKILL_MATCH_PROMPT = PromptTemplate(
    input_variables=["context", "job_description"],
    template="""You are an expert technical recruiter.

Given the candidate's resume context and a job description, provide:
1. MATCHED SKILLS       — skills the candidate has that match the JD
2. MISSING SKILLS       — skills in the JD the candidate lacks
3. MATCH SCORE          — percentage fit (0-100%)
4. RECOMMENDATION       — short hiring recommendation (2-3 sentences)

Resume Context:
{context}

Job Description:
{job_description}

Analysis:"""
)

_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are a professional resume writer.
Using ONLY the resume information below, write a compelling professional summary 
in 3-4 sentences. Focus on years of experience, key technical skills, notable 
achievements, and career highlights. Do not invent anything not in the context.

Resume Information:
{context}

Professional Summary:"""
)

_INTERVIEW_PROMPT = PromptTemplate(
    input_variables=["context", "role"],
    template="""You are an experienced technical interviewer.
Based on the candidate's resume below, generate 5 tailored interview questions
for the role of {role}. 
For each question, explain why it is relevant to this candidate's background.

Resume:
{context}

Interview Questions:"""
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: format retrieved docs into a single string
# ══════════════════════════════════════════════════════════════════════════════

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('section', 'general').upper()}]\n{doc.page_content}"
        for doc in docs
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chain builders
# ══════════════════════════════════════════════════════════════════════════════

def build_qa_chain(retriever) -> object:
    """
    General Q&A chain.
    Ask anything about the resume.

    Usage:
        chain = build_qa_chain(retriever)
        answer = chain.invoke("What is the candidate's current role?")
    """
    llm = settings.get_llm()

    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def build_skill_match_chain(retriever) -> object:
    """
    Skill matching chain — compares resume against a job description.

    Usage:
        chain = build_skill_match_chain(retriever)
        result = chain.invoke("We need a Python backend engineer with AWS...")
    """
    llm = settings.get_llm()

    chain = (
        {
            "context":         retriever | _format_docs,
            "job_description": RunnablePassthrough(),
        }
        | _SKILL_MATCH_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def build_summary_chain(retriever) -> object:
    """
    Auto-generate a professional summary from resume content.

    Usage:
        chain = build_summary_chain(retriever)
        summary = chain.invoke("Generate a professional summary")
    """
    llm = settings.get_llm()

    chain = (
        {
            "context": retriever | _format_docs,
        }
        | _SUMMARY_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def build_interview_chain(retriever) -> object:
    """
    Generate tailored interview questions for a specific role.

    Usage:
        chain = build_interview_chain(retriever)
        questions = chain.invoke({"role": "Senior Backend Engineer"})
    """
    llm = settings.get_llm()

    chain = (
        {
            "context": retriever | _format_docs,
            "role":    RunnablePassthrough(),
        }
        | _INTERVIEW_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain