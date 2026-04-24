import fitz  # PyMuPDF

from otherformat import OtherFormats
from sectiondevide import section
from parsedsection import parse_sections_data


import re


# ─────────────────────────────────────────────
# TEXT EXTRACTION (SINGLE COLUMN)
# ─────────────────────────────────────────────
def extract_text_single_column(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        blocks = page.get_text("blocks")
        blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for b in blocks:
            text += b[4].strip() + "\n"

        text += "\n"

    return clean_text(text)


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# PERSONAL INFO
# ─────────────────────────────────────────────
def extract_email(text):
    cleaned = re.sub(r'\s*@\s*', '@', text)
    cleaned = re.sub(r'\s*\.\s*', '.', cleaned)

    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cleaned)
    return match.group(0) if match else None


def extract_personal_info(pdf_path, text):
    data = {}

    # EMAIL
    data['email'] = extract_email(text)

    # PHONE
    phone = re.search(r'(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}', text)
    data['phone'] = re.sub(r'\D', '', phone.group(0)) if phone else None

    # NAME (top lines heuristic)
    lines = text.split("\n")
    for line in lines[:5]:
        if len(line.split()) <= 5 and not re.search(r'\d|@', line):
            data['name'] = line.strip()
            break
    else:
        data['name'] = None

    # LOCATION
    location = re.search(
        r'\b(Pune|Mumbai|San Francisco, CA|Delhi|Bangalore|Hyderabad|Chennai|India)\b.*',
        text
    )
    data['location'] = location.group(0) if location else None

    # LINKS (PyMuPDF)
    data["linkedin"] = None
    data["github"] = None

    doc = fitz.open(pdf_path)
    for page in doc:
        for link in page.get_links():
            uri = link.get("uri", "")
            if "linkedin.com" in uri:
                data["linkedin"] = uri
            elif "github.com" in uri:
                data["github"] = uri

    # fallback (text-based)
    if not data["linkedin"] and re.search(r'\bLinkedIn\b', text, re.I):
        data["linkedin"] = "linkedin"

    if not data["github"] and re.search(r'\bGit(hub)?\b', text, re.I):
        data["github"] = "github"

    return data


def extract_text_auto(pdf_path):
    doc = fitz.open(pdf_path)

    for page in doc:
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        page_width = page.rect.width

        left = [b for b in text_blocks if b[0] < page_width * 0.45]
        right = [b for b in text_blocks if b[0] > page_width * 0.55]

        if len(left) > 2 and len(right) > 2:
            return OtherFormats.extract_text(pdf_path)  # your 2-column logic
        else:
            return extract_text_single_column(pdf_path)
def parse_resume(pdf_path):
    text = extract_text_auto(pdf_path)
    personal_info = extract_personal_info(pdf_path, text)
    sections = section.extract_sections(text)
    # parsed_sections = parse_sections_data(sections)

    return {
        "personal_info": personal_info,
        "sections": sections
    }
        
# Resume Test File 4
if __name__ == "__main__":
  path = r"C:\Users\realm\Desktop\AI_ML_Project\pdf-parser\TestFile\Test FIles\Resume Test File 1.pdf"
  text_extraction=parse_resume(path)
  import json
  print(json.dumps(text_extraction, indent=4))

