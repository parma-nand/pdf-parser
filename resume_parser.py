import fitz  # PyMuPDF
import re


# ─────────────────────────────────────────────
# TEXT EXTRACTION (LAYOUT AWARE)
# ─────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        blocks = page.get_text("blocks")

        # Filter only text blocks
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
        if not text_blocks:
            continue

        page_width = page.rect.width

        # Detect columns
        left_blocks  = [b for b in text_blocks if b[0] <= page_width * 0.45]
        right_blocks = [b for b in text_blocks if b[0] >  page_width * 0.45]
        is_two_col   = len(left_blocks) >= 3 and len(right_blocks) >= 3

        if is_two_col:
            # Find column split dynamically
            left_x1  = max(b[2] for b in left_blocks)
            right_x0 = min(b[0] for b in right_blocks)
            split_x  = (left_x1 + right_x0) / 2

            left_col  = sorted([b for b in text_blocks if b[2] <= split_x + 5], key=lambda b: b[1])
            right_col = sorted([b for b in text_blocks if b[0] >= split_x - 5], key=lambda b: b[1])

            for b in left_col:
                full_text += b[4].strip() + "\n"

            full_text += "\n"

            for b in right_col:
                full_text += b[4].strip() + "\n"

        else:
            # Single column
            text_blocks.sort(key=lambda b: (round(b[1] / 15) * 15, b[0]))
            for b in text_blocks:
                full_text += b[4].strip() + "\n"

        full_text += "\n"

    return clean_text(full_text)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# LINK EXTRACTION (ROBUST)
# ─────────────────────────────────────────────
def extract_links(pdf_path: str, text: str):
    data = {"linkedin": None, "github": None}

    doc = fitz.open(pdf_path)

    # 1. Extract clickable links (best)
    for page in doc:
        links = page.get_links()
        for link in links:
            uri = link.get("uri", "")
            if "linkedin.com" in uri:
                data["linkedin"] = uri
            elif "github.com" in uri:
                data["github"] = uri

    # 2. Fallback: regex URLs
    if not data["linkedin"]:
        match = re.search(r'https?://[^\s]*linkedin\.com/[^\s]+', text, re.I)
        if match:
            data["linkedin"] = match.group(0)

    if not data["github"]:
        match = re.search(r'https?://[^\s]*github\.com/[^\s]+', text, re.I)
        if match:
            data["github"] = match.group(0)

    # 3. Fallback: keyword detection (icon-based resumes)
    if not data["linkedin"] and re.search(r'\bLinkedIn\b', text, re.I):
        data["linkedin"] = "FOUND (no URL)"

    if not data["github"] and re.search(r'\bGit(hub)?\b', text, re.I):
        data["github"] = "FOUND (no URL)"

    return data


# ─────────────────────────────────────────────
# PERSONAL INFO EXTRACTION
# ─────────────────────────────────────────────
# -------- EMAIL (ROBUST) --------
def extract_email(text):
    # Step 1: Normalize broken emails
    cleaned = re.sub(r'\s*@\s*', '@', text)
    cleaned = re.sub(r'\s*\.\s*', '.', cleaned)

    # Step 2: Extract email
    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cleaned)

    return email.group(0) if email else None
def extract_personal_info(pdf_path: str):
    text = extract_text(pdf_path)

    data = {}

    # EMAIL
    data['email'] = extract_email(text)

    # PHONE
    phone = re.search(r'(\+?\d{1,3}[\s-]?)?[6-9]\d{9}', text)
    data['phone'] = re.sub(r'\D', '', phone.group(0)) if phone else None

    # NAME (top heuristic)
    lines = text.split("\n")
    for line in lines[:5]:
        if len(line.split()) <= 5 and not re.search(r'\d|@', line):
            data['name'] = line.strip()
            break
    else:
        data['name'] = None

    # LOCATION
    location = re.search(
        r'\b(Pune|Mumbai|Delhi|Bangalore|Hyderabad|Chennai|India)\b.*',
        text
    )
    data['location'] = location.group(0) if location else None

    # LINKS
    links = extract_links(pdf_path, text)
    data.update(links)

    return data


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    path = r"C:/Users/realm/Desktop/AI_ML_Project/pdf-parser/TestFile/Test FIles/Resume Test File 4.pdf"

    result = extract_personal_info(path)

    print("\nFINAL OUTPUT:\n")
    for k, v in result.items():
        print(f"{k}: {v}")
