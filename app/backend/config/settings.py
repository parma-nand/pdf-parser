"""
config/settings.py
===================
Central config loader — reads from config/.env

Usage anywhere in the project:
    from config.settings import settings
    llm = settings.get_llm()
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings:
    openai_api_key:     str   = os.getenv("OPENAI_API_KEY",     "")
    llm_model:          str   = os.getenv("LLM_MODEL",          "gpt-4o-mini")
    llm_temperature:    float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_max_tokens:     int   = int(os.getenv("LLM_MAX_TOKENS",    "1024"))
    embedding_model:    str   = os.getenv("EMBEDDING_MODEL",    "all-MiniLM-L6-v2")
    chroma_persist_dir: str   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection:  str   = os.getenv("CHROMA_COLLECTION",  "resumes")
    pinecone_api_key:   str   = os.getenv("PINECONE_API_KEY",   "")
    pinecone_index:     str   = os.getenv("PINECONE_INDEX",     "")
    pinecone_namespace: str   = os.getenv("PINECONE_NAMESPACE", "resumes")

    def get_llm(self):
        from langchain_openai import ChatOpenAI
        if not self.openai_api_key or self.openai_api_key == "sk-xxxx":
            raise ValueError("Set OPENAI_API_KEY in config/.env")
        return ChatOpenAI(
            model=self.llm_model,
            api_key=self.openai_api_key,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

    def validate(self):
        if not self.openai_api_key or self.openai_api_key == "sk-xxxx":
            raise ValueError("Set your real OPENAI_API_KEY in config/.env")


settings = Settings()