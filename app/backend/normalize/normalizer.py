"""
normalize/normalizer.py  (fixed)
=================================
Step 2 - Clean and normalize the text within each section.

Fixes applied
-------------
1.  Strip leading/trailing whitespace per line.
2.  Collapse multiple consecutive blank lines into one.
3.  Remove non-printable / control characters.
4.  Normalize unicode NFKC.
5.  Collapse inner spaces/tabs per line.
6.  Remove PDF artefacts: page numbers, dividers, (cid:N) glyphs.
7.  Fix broken multi-byte encoding sequences.
8.  Replace ALL unicode bullet variants with a plain "- " prefix.
9.  Fix run-together words caused by bullet removal without space insertion
    e.g. "Usercanregister/login,view/editpersonal" -> split at camelCase.
10. Remove stray raw unicode escapes like \u2022 appearing as literal text.
"""

import re
import unicodedata


# ── Compiled patterns ────────────────────────────────────────────────────────

_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")
_DIVIDER_RE  = re.compile(r"^[\-_=\*\|~]{3,}\s*$")
_CID_RE      = re.compile(r"\(cid:\d+\)")

# Literal \uXXXX sequences appearing as raw text in some PDFs
_RAW_UNICODE_ESC_RE = re.compile(r"\\u[0-9a-fA-F]{4}")

# All unicode bullet / list-marker variants -> replace with "- "
_BULLET_RE = re.compile(
    "["
    "\u2022\u2023\u2024\u2025\u2043\u204c\u204d"
    "\u2219\u25aa\u25ab\u25b4\u25b8\u25cf\u25e6"
    "\u2713\u2714\u2717\u2718\u2794\u27a2\u29bf"
    "\u00b7\u00bb\u203b\u2015\u2012"
    "]",
    re.UNICODE,
)

# camelCase boundary splitter
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")

# Encoding fix table
_ENCODING_FIXES: dict = {}
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x99)] = "'"
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x9c)] = '"'
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x9d)] = '"'
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x98)] = "'"
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x93)] = "-"
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0x94)] = "-"
_ENCODING_FIXES[chr(0xe2)+chr(0x80)+chr(0xa6)] = "..."
_ENCODING_FIXES[chr(0xc2)+chr(0xa0)]           = " "
_ENCODING_FIXES[chr(0x00)]                     = ""
_ENCODING_FIXES[chr(0xfeff)]                   = ""


def _fix_encoding(text: str) -> str:
    for bad, good in _ENCODING_FIXES.items():
        text = text.replace(bad, good)
    return text


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_control_chars(text: str) -> str:
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )


def _fix_run_together(token: str) -> str:
    """Split run-together CamelCase tokens that have no spaces and are long."""
    if " " not in token and len(token) > 12:
        return _CAMEL_RE.sub(r"\1 \2", token)
    return token


def _normalize_line(line: str) -> str:
    line = line.strip()
    line = _RAW_UNICODE_ESC_RE.sub(" ", line)   # remove raw \uXXXX text
    line = _CID_RE.sub("", line)                # remove (cid:N)
    line = _BULLET_RE.sub("- ", line)           # unicode bullets -> "- "
    line = re.sub(r"[ \t]+", " ", line)         # collapse whitespace
    # Fix run-together words token by token
    tokens = [_fix_run_together(t) for t in line.split(" ")]
    line = " ".join(tokens)
    line = re.sub(r"[ \t]+", " ", line).strip()
    # Fix run-together words caused by PDF font encoding
    return line


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PAGE_NUM_RE.match(stripped):
        return True
    if _DIVIDER_RE.match(stripped):
        return True
    return False


def _collapse_blank_lines(lines: list) -> list:
    result = []
    prev_blank = False
    for line in lines:
        if line.strip() == "":
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return result


def normalize_text(lines: list) -> dict:
    lines = [_fix_encoding(l) for l in lines]
    lines = [_normalize_unicode(l) for l in lines]
    lines = [_remove_control_chars(l) for l in lines]
    lines = [_normalize_line(l) for l in lines]
    lines = [l for l in lines if not _is_noise_line(l)]
    lines = _collapse_blank_lines(lines)

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    cleaned_text = "\n".join(lines)
    word_count   = len(cleaned_text.split())

    return {
        "lines":        lines,
        "cleaned_text": cleaned_text,
        "word_count":   word_count,
    }


def normalize_sections(raw_result: dict) -> dict:
    sections   = raw_result.get("sections", {})
    normalized = {}

    for section_name, lines in sections.items():
        normalized[section_name] = normalize_text(lines)

    total_words = sum(v["word_count"] for v in normalized.values())

    return {
        "source_file": raw_result.get("source_file", ""),
        "sections":    normalized,
        "total_words": total_words,
    }