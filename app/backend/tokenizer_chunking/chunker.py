"""
tokenizer_chunking/chunker.py
==============================
Step 4 – Tokenize, chunk, and build metadata-rich JSON ready for embedding
         into a vector database (Pinecone, Chroma, Weaviate, FAISS, etc.).

Strategy
--------
• Word-based tokenizer (zero external downloads required).  1 word ≈ 1 token.
• Each section becomes one or more overlapping chunks of `chunk_size` tokens
  with `overlap` tokens of context carried over from the previous chunk.
• Every chunk carries a metadata dict with:
    - chunk_id          unique deterministic ID
    - source_file       original PDF path
    - section           which resume section this chunk came from
    - chunk_index       position within that section's chunks
    - total_chunks      how many chunks that section produced
    - token_count       word count of this chunk
    - char_count        character count
    - candidate_name    extracted name  (for vector DB filter)
    - candidate_email   extracted email (for vector DB filter)
    - skills_summary    comma-joined raw skills (for vector DB filter)
    - processed_at      ISO-8601 timestamp
• The `text` field is the string you feed to your embeddings model.
"""

import uuid
from datetime import datetime, timezone

# ── Tunable defaults ────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 200   # words per chunk
DEFAULT_OVERLAP    = 40    # words of overlap between consecutive chunks


# ══════════════════════════════════════════════════════════════════════════════
# Core tokenizer (word-level, no downloads)
# ══════════════════════════════════════════════════════════════════════════════

def word_tokenize(text: str) -> list[str]:
    """
    Split text into word tokens using simple whitespace splitting.
    Preserves punctuation attached to words (e.g. "Python," stays as one token)
    which is fine for chunking purposes.
    """
    return [t for t in text.split() if t]


def tokens_to_text(tokens: list[str]) -> str:
    """Rejoin tokens into a readable string."""
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# Sliding-window chunker
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window_chunks(
    tokens: list[str],
    size: int,
    overlap: int,
) -> list[list[str]]:
    """
    Produce overlapping token windows.

    Example (size=5, overlap=2, tokens=[1..10]):
      [1,2,3,4,5]  [4,5,6,7,8]  [7,8,9,10]
    """
    if not tokens:
        return []
    chunks: list[list[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start += size - overlap
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Section-level chunk builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_section_chunks(
    section_name: str,
    cleaned_text: str,
    global_meta: dict,
    stem: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Tokenize one section and return a list of chunk dicts.

    Each chunk dict:
    {
        "chunk_id":    str,
        "text":        str,   ← feed this to your embeddings model
        "token_count": int,
        "char_count":  int,
        "metadata":    { ... }
    }
    """
    tokens      = word_tokenize(cleaned_text)
    token_windows = sliding_window_chunks(tokens, chunk_size, overlap)
    total       = len(token_windows)
    chunks: list[dict] = []

    for idx, window in enumerate(token_windows):
        text      = tokens_to_text(window)
        chunk_id  = f"{stem}__{section_name}__{idx}__{uuid.uuid4().hex[:8]}"

        chunks.append({
            "chunk_id":    chunk_id,
            "text":        text,
            "token_count": len(window),
            "char_count":  len(text),
            "metadata": {
                # ── identification ──────────────────────────────────────
                "chunk_id":          chunk_id,
                "source_file":       global_meta["source_file"],
                # ── position ────────────────────────────────────────────
                "section":           section_name,
                "chunk_index":       idx,
                "total_chunks_in_section": total,
                "is_first_chunk":    idx == 0,
                "is_last_chunk":     idx == total - 1,
                # ── size ────────────────────────────────────────────────
                "token_count":       len(window),
                "char_count":        len(text),
                # ── candidate fields (for DB filtering) ─────────────────
                "candidate_name":    global_meta.get("candidate_name"),
                "candidate_email":   global_meta.get("candidate_email"),
                "candidate_phone":   global_meta.get("candidate_phone"),
                "candidate_location": global_meta.get("candidate_location"),
                "linkedin":          global_meta.get("linkedin"),
                "github":            global_meta.get("github"),
                "skills_summary":    global_meta.get("skills_summary"),
                # ── provenance ──────────────────────────────────────────
                "processed_at":      global_meta["processed_at"],
                "chunk_size_config": chunk_size,
                "overlap_config":    overlap,
            },
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_and_chunk(
    normalized_result: dict,
    extracted_info:    dict,
    stem:              str,
    chunk_size:        int = DEFAULT_CHUNK_SIZE,
    overlap:           int = DEFAULT_OVERLAP,
) -> dict:
    """
    Step-4 entry point.

    Parameters
    ----------
    normalized_result : output of normalize.normalizer.normalize_sections()
    extracted_info    : output of information_extraction.extractor.extract_information()
    stem              : filename stem used to namespace chunk IDs
    chunk_size        : max words per chunk  (default 200)
    overlap           : overlapping words between chunks  (default 40)

    Returns
    -------
    {
      "metadata":     { … global candidate metadata … },
      "total_chunks": int,
      "sections_processed": [str],
      "chunks": [
          {
            "chunk_id":    str,
            "text":        str,
            "token_count": int,
            "char_count":  int,
            "metadata":    { … }
          },
          …
      ]
    }
    """
    sections   = normalized_result.get("sections", {})
    personal   = extracted_info.get("personal_info", {})
    skills_obj = extracted_info.get("skills", {})
    raw_skills = skills_obj.get("raw_list", [])

    # ── Global metadata (shared across all chunks from this resume) ──────────
    global_meta: dict = {
        "source_file":        normalized_result.get("source_file", ""),
        "candidate_name":     personal.get("name"),
        "candidate_email":    personal.get("email"),
        "candidate_phone":    personal.get("phone"),
        "candidate_location": personal.get("location"),
        "linkedin":           personal.get("linkedin"),
        "github":             personal.get("github"),
        "skills_summary":     ", ".join(raw_skills[:30]),
        "processed_at":       datetime.now(timezone.utc).isoformat(),
        "chunk_size":         chunk_size,
        "overlap":            overlap,
    }

    # ── Build chunks per section ─────────────────────────────────────────────
    all_chunks: list[dict]       = []
    sections_processed: list[str] = []

    for section_name, section_data in sections.items():
        cleaned_text = section_data.get("cleaned_text", "").strip()
        if not cleaned_text:
            continue

        section_chunks = _build_section_chunks(
            section_name=section_name,
            cleaned_text=cleaned_text,
            global_meta=global_meta,
            stem=stem,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(section_chunks)
        sections_processed.append(section_name)

    return {
        "metadata":            global_meta,
        "total_chunks":        len(all_chunks),
        "sections_processed":  sections_processed,
        "chunks":              all_chunks,
    }