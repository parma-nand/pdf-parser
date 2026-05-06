"""
section_divide/extractor.py
============================
Step 1 - Extract text from a PDF resume and divide into named sections.

Uses PyMuPDF (fitz) as the primary extractor — correctly handles all font
types including subset-embedded fonts that cause pdfplumber to produce
run-together words. Falls back to pdfplumber if PyMuPDF is not installed.

Heading detection rules (strict)
---------------------------------
A line is a section heading ONLY if:
  1. Its full lowercased text (stripped of colons/whitespace) exactly matches
     a known keyword — OR —
  2. It is short (<=50 chars), looks like a heading (ALL CAPS, Title Case,
     or single capitalised phrase), AND contains a known keyword as the
     dominant content (keyword covers >60% of the line length).

This prevents lines like "Software Engineer - Enterprise Solutions" from
being misidentified as the "experience" section.
"""

import re
import unicodedata
from pathlib import Path


# ── Section keyword map ──────────────────────────────────────────────────────

SECTION_KEYWORDS = {
    "personal_info": {
        "contact", "personal", "profile", "about me",
        "personal information", "contact information", "contact details",
    },
    "summary": {
        "summary", "professional summary", "career summary", "objective",
        "career objective", "professional profile", "overview",
        "executive summary", "about",
    },
    "skills": {
        "skills", "technical skills", "core competencies", "competencies",
        "key skills", "skill set", "technologies", "tools", "expertise",
        "areas of expertise", "proficiencies", "technical expertise",
        "technical skills & tools", "skills & technologies",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment", "employment history", "career history", "work history",
        "positions held", "internships",
    },
    "education": {
        "education", "academic background", "qualifications",
        "academic qualifications", "educational background",
        "certifications and education",
    },
    "certifications": {
        "certifications", "certificates", "accreditations", "licenses",
        "professional certifications", "courses", "training",
        "courses & certifications",
    },
    "projects": {
        "projects", "key projects", "notable projects",
        "personal projects", "academic projects", "side projects",
    },
    "achievements": {
        "achievements", "awards", "honors", "accomplishments", "recognition",
    },
    "languages": {
        "languages", "language skills",
    },
    "interests": {
        "interests", "hobbies", "extracurricular",
    },
    "references": {
        "references", "referees",
    },
}

# flat lookup: keyword -> canonical name
_KW_LOOKUP: dict = {}
for _canon, _kws in SECTION_KEYWORDS.items():
    for _kw in _kws:
        _KW_LOOKUP[_kw] = _canon


# ── Text cleaning ─────────────────────────────────────────────────────────────

_CID_BULLET_RE  = re.compile(r"\(cid:127\)")
_CID_OTHER_RE   = re.compile(r"\(cid:\d+\)")
_RAW_ESCAPE_RE  = re.compile(r"\\u[0-9a-fA-F]{4}")
_DASH_RE        = re.compile(r"[\u2013\u2014\u2015\u2212]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_BULLET_RE      = re.compile(
    "["
    "\u2022\u2023\u2024\u2025\u2043\u204c\u204d"
    "\u2219\u25aa\u25ab\u25b4\u25b8\u25cf\u25e6"
    "\u2713\u2714\u2717\u2718\u2794\u27a2\u29bf"
    "\u00b7\u00bb\u203b"
    "]",
    re.UNICODE,
)

_ENC: dict = {}
_ENC[chr(0xe2)+chr(0x80)+chr(0x99)] = "'"
_ENC[chr(0xe2)+chr(0x80)+chr(0x9c)] = '"'
_ENC[chr(0xe2)+chr(0x80)+chr(0x9d)] = '"'
_ENC[chr(0xe2)+chr(0x80)+chr(0x98)] = "'"
_ENC[chr(0xe2)+chr(0x80)+chr(0x93)] = "-"
_ENC[chr(0xe2)+chr(0x80)+chr(0x94)] = "-"
_ENC[chr(0xe2)+chr(0x80)+chr(0xa6)] = "..."
_ENC[chr(0xc2)+chr(0xa0)]           = " "
_ENC[chr(0x00)]                     = ""
_ENC[chr(0xfeff)]                   = ""
_ENC["\ufffd"]                      = ""


def _clean_text(text: str) -> str:
    for bad, good in _ENC.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = _CID_BULLET_RE.sub("- ", text)
    text = _CID_OTHER_RE.sub("", text)
    text = _BULLET_RE.sub("- ", text)
    text = _DASH_RE.sub("-", text)
    text = _RAW_ESCAPE_RE.sub("", text)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )
    return text


def _clean_line(line: str) -> str:
    line = line.rstrip()
    line = _MULTI_SPACE_RE.sub(" ", line)
    return line.strip()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_with_pymupdf(pdf_path: Path) -> str:
    import fitz
    doc   = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=3, y_tolerance=3,
                keep_blank_chars=False, use_text_flow=True,
            )
            if not words:
                continue
            lines, c_top, c_line = [], words[0]["top"], []
            for w in sorted(words, key=lambda x: (round(x["top"]/3), x["x0"])):
                if abs(w["top"] - c_top) <= 3:
                    c_line.append(w["text"])
                else:
                    if c_line:
                        lines.append(" ".join(c_line))
                    c_line, c_top = [w["text"]], w["top"]
            if c_line:
                lines.append(" ".join(c_line))
            pages.append("\n".join(lines))
    return "\n".join(pages)


def extract_raw_text(pdf_path: Path) -> str:
    try:
        text = _extract_with_pymupdf(pdf_path)
    except ImportError:
        text = _extract_with_pdfplumber(pdf_path)
    return _clean_text(text)


# ── Strict section heading detector ──────────────────────────────────────────

def _is_heading(line: str) -> tuple:
    """
    Return (True, canonical_name) only when the line is clearly a section
    heading — not a job title, company name, or content line.

    Rules (in order):
    1. EXACT match  — stripped lowercase == known keyword
    2. EXACT match  — after stripping decorator chars (=, -, |)
    3. DOMINANT match — line is short + looks like a heading + the matched
       keyword covers at least 60% of the cleaned line length.
       This blocks "Software Engineer - Enterprise Solutions" from matching
       "experience" because "experience" (10) / line_len (38) = 26% < 60%.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False, ""

    lower = stripped.lower().rstrip(":").strip()

    # Strip decorator chars used as visual separators
    lower_nodec = re.sub(
        r"^[\-=\|\s\u2500\u2550]+|[\-=\|\s\u2500\u2550]+$", "", lower
    ).strip()

    # Rule 1 & 2: exact match
    for candidate in (lower, lower_nodec):
        if candidate in _KW_LOOKUP:
            return True, _KW_LOOKUP[candidate]

    # Rule 3: dominant keyword match — only for heading-looking lines
    looks_like_heading = (
        stripped.isupper()                                      # ALL CAPS
        or re.match(r"^[A-Z][A-Z\s&/()\-]+$", stripped)        # ALL CAPS with symbols
        or (stripped.istitle() and "|" not in stripped)         # Title Case, no pipe
    )

    if looks_like_heading and len(lower_nodec) <= 50:
        for kw, canon in _KW_LOOKUP.items():
            if kw in lower_nodec:
                # keyword must cover at least 60% of the line
                if len(kw) / max(len(lower_nodec), 1) >= 0.60:
                    return True, canon

    return False, ""


# ── Section divider ───────────────────────────────────────────────────────────

def divide_into_sections(raw_text: str) -> dict:
    sections: dict       = {}
    current_section: str = "unclassified"

    for raw_line in raw_text.split("\n"):
        line = _clean_line(raw_line)
        if not line:
            continue

        is_heading, canon = _is_heading(line)
        if is_heading:
            current_section = canon
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, [])
            sections[current_section].append(line)

    return {k: v for k, v in sections.items() if v}


# ── Entry point ───────────────────────────────────────────────────────────────

def extract_and_divide(pdf_path: Path) -> dict:
    raw_text = extract_raw_text(pdf_path)
    sections = divide_into_sections(raw_text)
    return {
        "raw_text":      raw_text,
        "sections":      sections,
        "source_file":   str(pdf_path),
        "total_chars":   len(raw_text),
        "section_count": len(sections),
    }