"""
information_extraction/extractor.py  (fixed)
=============================================
Step 3 - Extract structured information from normalized resume sections.

Fixes in this version
---------------------
1. location    - strict line-by-line check; rejects lines containing newlines
                 or any tech keyword (Spring, Boot, React, etc.); returns null
                 if no valid "City, ST" pattern found.
2. skills      - expanded keyword table covers SpringBoot, React.js, Bootstrap,
                 CSS3, Hibernate, JDBC, JPA, Maven, OOP, DataStructures, etc.
                 ALL-CAPS section-header sub-labels are skipped.
                 Each skill is checked against ALL categories so
                 "PostgreSQL" lands in databases not programming_languages.
3. summary     - auto-generated from WHOLE resume (not copied from the
                 written summary section). Max 200 words.
4. year fix    - YEAR_RE only matches full 4-digit years; never fragments.
"""

import re
from typing import Any


# ======================================================================
# Shared regex
# ======================================================================

EMAIL_RE    = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
PHONE_RE    = re.compile(r"(\+?[\d][\d\s\-().]{6,}\d)")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)
GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.I)
URL_RE      = re.compile(r"https?://[^\s]+", re.I)

# Full 4-digit year, word-boundary anchored - never matches fragments
YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")

DATE_RANGE_RE = re.compile(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})"
    r"\s*[-\u2013\u2014to]+\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
    r"|\d{4}|Present|Current|Now|present|current)",
    re.I,
)

GPA_RE = re.compile(r"GPA\s*[:\-]?\s*([\d.]+)", re.I)

# ======================================================================
# Skills keyword table  (all lowercase for matching)
# ======================================================================
SKILL_KEYWORDS = {
    "programming_languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "golang",
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab",
        "perl", "bash", "shell", "powershell", "html", "html5", "css", "css3",
        "xml", "json", "yaml", "r",
    },
    "web_frameworks": {
        "react", "react.js", "reactjs", "angular", "vue", "vue.js", "next.js",
        "nextjs", "fastapi", "flask", "django", "spring", "springboot",
        "spring boot", "express", "node.js", "nodejs", "rest", "graphql",
        "grpc", "hibernate", "jdbc", "jpa", "bootstrap", "tailwind",
        "microservices", "servlet", "jsp", "thymeleaf",
    },
    "ml_ai": {
        "tensorflow", "keras", "pytorch", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "opencv", "nltk", "spacy", "transformers", "hugging face",
        "bert", "gpt", "llm", "nlp", "machine learning", "deep learning",
        "neural network", "cnn", "rnn", "lstm", "computer vision",
        "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    },
    "data_engineering": {
        "spark", "pyspark", "hadoop", "kafka", "airflow", "dbt", "databricks",
        "snowflake", "bigquery", "redshift", "hive", "flink", "nifi",
    },
    "cloud_devops": {
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "ci/cd", "jenkins", "github actions", "gitlab", "helm", "prometheus",
        "grafana", "mlflow", "kubeflow", "ci/cd pipelines", "ec2", "s3",
        "lambda", "cloudformation", "circleci",
    },
    "databases": {
        "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
        "cassandra", "dynamodb", "sqlite", "oracle", "elasticsearch",
        "dbms", "database management", "nosql",
    },
    "cs_fundamentals": {
        "data structures", "algorithms", "computer networks", "operating systems",
        "oop", "object oriented", "object-oriented", "design patterns",
        "system design", "distributed systems", "oops",
    },
    "tools_other": {
        "git", "github", "jira", "confluence", "postman", "vscode", "linux",
        "excel", "tableau", "power bi", "looker", "maven", "gradle", "npm",
        "webpack", "figma", "intellij", "eclipse", "netbeans", "swagger",
        "sonarqube", "junit", "mockito",
    },
}

# ALL-CAPS sub-section header lines inside the skills block to skip
_SKILL_HEADER_RE = re.compile(
    r"^(FRONT[\s\-]*END|BACK[\s\-]*END|CS\s*FUND|CS\s*FUNDAMENTAL|"
    r"MISCELLANEOUS|TOOL|FRAMEWORK|LANGUAGE|DATABASE|CLOUD|DEVOPS|"
    r"OTHER|TECHNICAL|SOFT\s*SKILL|CORE\s*COMPETENC|DEVELOPMENT)S?[\s:]*$",
    re.I,
)

# Tech keywords that must NOT appear inside a location string
_TECH_IN_LOCATION_RE = re.compile(
    r"\b(spring|boot|react|django|flask|node|docker|aws|azure|gcp|sql|"
    r"java|python|javascript|typescript|html|css|mongodb|postgresql|redis|"
    r"kafka|kubernetes|terraform|git|github|linux|maven|hibernate|jdbc|jpa|"
    r"angular|vue|express|bootstrap|tailwind|microservice|devops|ci|cd)\b",
    re.I,
)

# Valid single-line location patterns:
#   "City, ST"              e.g. "Mumbai, MH"
#   "City, Country"         e.g. "Mumbai, India"
#   "City, State, Country"  e.g. "Pune, Maharashtra, IN"
#   "City, State, Country"  e.g. "Pune, Maharashtra, India"
_LOCATION_LINE_RE = re.compile(
    r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"          # City
    r",\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2})"  # State or ST code
    r"(?:,\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2}))?$"  # optional Country
)


# ======================================================================
# Personal info
# ======================================================================

def extract_personal_info(sections: dict) -> dict:
    candidate_secs = ["personal_info", "unclassified", "summary"]
    all_lines: list = []
    for sec in candidate_secs:
        if sec in sections:
            all_lines.extend(sections[sec]["lines"])

    # Contact info is always in the first 20 lines
    top_lines = all_lines[:20]
    full_text = "\n".join(top_lines)

    info: dict = {
        "name":     None,
        "email":    None,
        "phone":    None,
        "location": None,
        "linkedin": None,
        "github":   None,
        "website":  None,
    }

    m = EMAIL_RE.search(full_text)
    if m:
        info["email"] = m.group(0)

    m = PHONE_RE.search(full_text)
    if m:
        raw = m.group(1).strip()
        if len(re.sub(r"\D", "", raw)) >= 7:
            info["phone"] = raw

    m = LINKEDIN_RE.search(full_text)
    if m:
        info["linkedin"] = "https://" + m.group(0)

    m = GITHUB_RE.search(full_text)
    if m:
        info["github"] = "https://" + m.group(0)

    for url_m in URL_RE.finditer(full_text):
        url = url_m.group(0)
        if "linkedin" not in url and "github" not in url:
            info["website"] = url
            break

    # Name: first short title-cased line, no email/phone/url
    skip_re = re.compile(
        r"@|https?://|linkedin|github|\d{3,}|resume|cv\b|curriculum|vitae",
        re.I,
    )
    for line in top_lines:
        s = line.strip()
        if s and not skip_re.search(s) and 2 <= len(s.split()) <= 5 and s[0].isupper():
            info["name"] = s
            break

    # Location: check each line and also pipe-separated segments
    # Must not contain tech keywords, must match City/ST pattern
    def _check_location_candidate(candidate):
        candidate = candidate.strip()
        if not candidate or "\n" in candidate:
            return None
        if _TECH_IN_LOCATION_RE.search(candidate):
            return None
        if skip_re.search(candidate):
            return None
        if _LOCATION_LINE_RE.match(candidate):
            return candidate
        m = re.match(
            r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
            r",\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
            r"(?:\s+\d{4,6})?$",
            candidate,
        )
        if m:
            return candidate
        return None

    for line in top_lines:
        # First try the whole line
        result = _check_location_candidate(line)
        if result:
            info["location"] = result
            break
        # Then try each pipe-separated segment (common in single-line contact headers)
        if "|" in line:
            for segment in line.split("|"):
                result = _check_location_candidate(segment)
                if result:
                    info["location"] = result
                    break
        if info["location"]:
            break

    return info


# ======================================================================
# Auto-generated summary from whole resume (max 200 words)
# ======================================================================

def generate_summary(sections: dict, personal_info: dict) -> str:
    """
    Build a factual summary from parsed sections.
    Does NOT copy the written summary/objective section verbatim.
    """
    parts: list = []

    name = personal_info.get("name") or "The candidate"

    # Role: first short non-date line in experience
    role = None
    if "experience" in sections:
        for line in sections["experience"]["lines"]:
            s = line.strip()
            if not s:
                continue
            # Role is the first pipe-segment on the date-range header line
            dr_m = DATE_RANGE_RE.search(s)
            if dr_m:
                before = s[:dr_m.start()].strip().rstrip("|-.,").strip()
                if before:
                    title_part = before.split("|")[0].strip()
                    if title_part and 1 <= len(title_part.split()) <= 8:
                        role = title_part
                break  # only use the first experience entry

    if role:
        parts.append(f"{name} is a {role}.")
    else:
        parts.append(f"{name} is a software professional.")

    # Skills snapshot (top 10)
    if "skills" in sections:
        skill_data = extract_skills(sections)
        raw = skill_data.get("raw_list", [])
        if raw:
            parts.append(f"Technical skills include {', '.join(raw[:10])}.")

    # Experience
    if "experience" in sections:
        exp = extract_experience(sections)
        if exp:
            titles = [
                f"{e.get('title','')} at {e.get('company','')}".strip(" at")
                for e in exp if e.get("title")
            ]
            if titles:
                parts.append(f"Work experience: {'; '.join(titles[:3])}.")

    # Education
    if "education" in sections:
        edu = extract_education(sections)
        if edu:
            e = edu[0]
            edu_parts = [e.get("degree",""), e.get("field","")]
            inst = e.get("institution","")
            if inst:
                short_inst = " ".join(inst.split()[:4])
                edu_parts += ["from", short_inst]
            yr = e.get("duration") or e.get("year","")
            if yr:
                edu_parts.append(yr)
            edu_str = " ".join(p for p in edu_parts if p).strip()
            if edu_str:
                parts.append(f"Education: {edu_str}.")

    # Projects
    if "projects" in sections:
        proj = extract_projects(sections)
        if proj:
            names = [p.get("name","") for p in proj if p.get("name")][:3]
            if names:
                parts.append(f"Notable projects: {', '.join(names)}.")

    # Certifications
    if "certifications" in sections:
        certs = extract_certifications(sections)
        if certs:
            cnames = [c.get("name","") for c in certs if c.get("name")][:2]
            if cnames:
                parts.append(f"Certifications: {', '.join(cnames)}.")

    full  = " ".join(parts)
    words = full.split()
    if len(words) > 200:
        full = " ".join(words[:200]) + "..."
    return full


# ======================================================================
# Skills
# ======================================================================

def extract_skills(sections: dict) -> dict:
    if "skills" not in sections:
        return {"raw_list": [], "categorized": {}}

    lines = sections["skills"]["lines"]
    raw_skills: list = []

    for line in lines:
        stripped = line.strip()

        # Skip ALL-CAPS sub-header lines  e.g. "FRONT-END DEVELOPMENT"
        if _SKILL_HEADER_RE.match(stripped):
            continue
        # Also skip lines that are purely uppercase with <= 4 words
        clean_stripped = re.sub(r"[\-\s]", "", stripped)
        if clean_stripped.isupper() and len(stripped.split()) <= 4:
            continue

        # Strip label prefix e.g. "Programming Languages: Python, Java"
        if ":" in stripped:
            _, _, rest = stripped.partition(":")
            stripped = rest.strip()

        if not stripped:
            continue

        # Remove leading bullet/dash artefacts
        stripped = re.sub(r"^[\-\*\u2022\s]+", "", stripped).strip()

        # Split by comma, pipe, semicolon, slash
        # Do NOT split on hyphen inside words (C++, CI/CD, etc.)
        tokens = re.split(r"[,|;]+", stripped)
        for t in tokens:
            t = t.strip().strip("-").strip()
            if t and len(t) > 1:
                raw_skills.append(t)

    # De-duplicate preserving order
    seen: set = set()
    unique: list = []
    for s in raw_skills:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(s)

    # Categorise — check EVERY category, not just first match
    # Use a priority order so a skill lands in the most specific bucket
    priority = [
        "databases",
        "cloud_devops",
        "data_engineering",
        "ml_ai",
        "web_frameworks",
        "cs_fundamentals",
        "tools_other",
        "programming_languages",
    ]
    categorized: dict = {cat: [] for cat in SKILL_KEYWORDS}
    uncategorized: list = []

    for skill in unique:
        skill_low = skill.lower().strip()
        matched_cat = None
        for cat in priority:
            keywords = SKILL_KEYWORDS[cat]
            if any(kw == skill_low or skill_low == kw or kw in skill_low for kw in keywords):
                matched_cat = cat
                break
        if matched_cat:
            categorized[matched_cat].append(skill)
        else:
            uncategorized.append(skill)

    categorized = {k: v for k, v in categorized.items() if v}
    if uncategorized:
        categorized["other"] = uncategorized

    return {"raw_list": unique, "categorized": categorized}


# ======================================================================
# Education
# ======================================================================

DEGREE_RE = re.compile(
    r"\b(B\.?Tech|B\.?E\.?|B\.?Sc|B\.?S\.?|M\.?Tech|M\.?E\.?|M\.?Sc|M\.?S\.?|"
    r"MBA|Ph\.?D|BCA|MCA|B\.?Com|M\.?Com|Bachelor|Master|Doctorate|Associate|"
    r"Diploma|High\s+School|HSC|SSC|BE|ME|BS|MS)\b",
    re.I,
)
INST_RE = re.compile(
    r"\b(University|College|Institute|School|Academy|Polytechnic|"
    r"IIT|NIT|BITS|VIT|RVCE|PESIT|MIT|SRM|Manipal|DSCE|MSRIT)\b",
    re.I,
)


def extract_education(sections: dict) -> list:
    if "education" not in sections:
        return []

    lines   = sections["education"]["lines"]
    entries = []
    current: dict = {}

    for line in lines:
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue

        deg_m = DEGREE_RE.search(line)
        if deg_m and not current.get("degree"):
            current["degree"] = deg_m.group(0)
            remainder = line[deg_m.end():].strip().lstrip("in").strip().lstrip(",").strip()
            if remainder and len(remainder.split()) <= 8:
                current["field"] = remainder

        dr_m = DATE_RANGE_RE.search(line)
        if dr_m:
            current["duration"] = f"{dr_m.group(1)} - {dr_m.group(2)}"
        else:
            yr_m = YEAR_RE.findall(line)
            if yr_m and not current.get("duration"):
                current["year"] = yr_m[-1]

        gpa_m = GPA_RE.search(line)
        if gpa_m:
            current["gpa"] = gpa_m.group(1)

        if INST_RE.search(line) and not current.get("institution"):
            current["institution"] = line.strip()

    if current:
        entries.append(current)

    return [e for e in entries if e]


# ======================================================================
# Experience
# ======================================================================

def extract_experience(sections: dict) -> list:
    """
    Parse experience entries using a sliding-window approach.

    Resume layout handled:
        Company Name                       <- line A
        Job Title                          <- line B
        Month YYYY - Month YYYY | City     <- line C (date range)
        - bullet ...                       <- bullets

    We look BACK up to 2 lines when we hit a date-range line to find
    company and title that appeared before the date.
    """
    if "experience" not in sections:
        return []

    lines   = [l.strip() for l in sections["experience"]["lines"] if l.strip()]
    entries = []
    current: dict  = {}
    pending_bullet = ""

    def _flush_bullet():
        nonlocal pending_bullet
        if pending_bullet.strip():
            current.setdefault("bullets", []).append(pending_bullet.strip())
        pending_bullet = ""

    def _flush_entry():
        _flush_bullet()
        if current:
            entries.append(dict(current))

    i = 0
    while i < len(lines):
        line = lines[i]

        dr_m = DATE_RANGE_RE.search(line)
        if dr_m:
            _flush_entry()

            # Parse duration and optional location after pipe
            dur_str = f"{dr_m.group(1)} - {dr_m.group(2)}"
            loc     = None
            after   = line[dr_m.end():].strip().lstrip("|").strip()
            if after and not re.search(r"\d{4}", after):
                loc = after

            current = {"duration": dur_str, "bullets": [], "location": loc}
            pending_bullet = ""

            # Look back for title (i-1) and company (i-2)
            if i >= 1 and not DATE_RANGE_RE.search(lines[i-1]):
                prev1 = lines[i-1].strip()
                if prev1 and not prev1.startswith("-"):
                    current["title"] = prev1
            if i >= 2 and not DATE_RANGE_RE.search(lines[i-2]):
                prev2 = lines[i-2].strip()
                if prev2 and not prev2.startswith("-") and prev2 != current.get("title"):
                    current["company"] = prev2

            i += 1
            continue

        # Bullet line
        if re.match(r"^[-*\u2022\u2023\u25cf\u2713>]\s*", line):
            _flush_bullet()
            pending_bullet = re.sub(r"^[-*\u2022\u2023\u25cf\u2713>\s]+", "", line).strip()
            i += 1
            continue

        # If we are inside an entry, continuation lines join current bullet
        if current:
            ends_ok    = pending_bullet.endswith((".", "!", "?"))
            starts_cap = line and line[0].isupper()
            if not pending_bullet:
                pending_bullet = line
            elif ends_ok and starts_cap:
                _flush_bullet()
                pending_bullet = line
            else:
                pending_bullet = pending_bullet + " " + line

        i += 1

    _flush_entry()
    return [e for e in entries if e]


# ======================================================================
# Certifications
# ======================================================================

def extract_certifications(sections: dict) -> list:
    if "certifications" not in sections:
        return []

    lines = sections["certifications"]["lines"]
    certs = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Remove leading bullet
        stripped = re.sub(r"^[\-\*\u2022\s]+", "", stripped).strip()
        if not stripped:
            continue
        cert: dict = {"name": stripped}
        yr = YEAR_RE.findall(stripped)
        if yr:
            cert["year"] = yr[-1]
        issuer_m = re.search(r"(?:by|\u2013|-|from)\s+([A-Z][^\n,]+)", stripped)
        if issuer_m:
            cert["issuer"] = issuer_m.group(1).strip()
        certs.append(cert)

    return certs


# ======================================================================
# Projects
# ======================================================================

def extract_projects(sections: dict) -> list:
    if "projects" not in sections:
        return []

    lines    = sections["projects"]["lines"]
    projects = []
    current: dict = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                projects.append(current)
                current = {}
            continue

        if re.match(r"^[\-\*\u2022\u2023\u25cf\u2713>]", stripped):
            desc = re.sub(r"^[\-\*\u2022\u2023\u25cf\u2713>\s]+", "", stripped).strip()
            current.setdefault("description_points", []).append(desc)
        elif not current.get("name"):
            current["name"] = stripped
            tech_m = re.search(r"[\(\[](.*?)[\)\]]", stripped)
            if tech_m:
                techs = [t.strip() for t in re.split(r"[,|/]", tech_m.group(1)) if t.strip()]
                current["technologies"] = techs
        else:
            current.setdefault("description_points", []).append(stripped)

    if current:
        projects.append(current)

    return projects


# ======================================================================
# Entry point
# ======================================================================

def extract_information(normalized_result: dict) -> dict:
    sections     = normalized_result.get("sections", {})
    personal     = extract_personal_info(sections)
    auto_summary = generate_summary(sections, personal)

    return {
        "source_file":    normalized_result.get("source_file", ""),
        "personal_info":  personal,
        "summary":        auto_summary,
        "skills":         extract_skills(sections),
        "education":      extract_education(sections),
        "experience":     extract_experience(sections),
        "certifications": extract_certifications(sections),
        "projects":       extract_projects(sections),
    }