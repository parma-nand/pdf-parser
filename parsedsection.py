import re
def parse_skills(text):
    text = text.split("\n", 1)[-1]

    # Remove category labels
    text = re.sub(r'\b(frontend|backend|databases|cloud & devops|data & ml|tools):', '', text, flags=re.I)

    skills = re.split(r',|\n', text)

    skills = [s.strip() for s in skills if s.strip()]

    return list(set(skills))

def parse_experience(text):
    experiences = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    i = 0
    while i < len(lines):
        # Detect company (line without bullets, short)
        if not lines[i].startswith("•") and len(lines[i]) < 50:
            company = lines[i]

            # Next line → role
            if i + 1 < len(lines):
                role = lines[i + 1]
            else:
                i += 1
                continue

            # Next line → duration
            if i + 2 < len(lines):
                duration = lines[i + 2]
            else:
                i += 1
                continue

            # Collect bullets
            desc = []
            j = i + 3
            while j < len(lines) and lines[j].startswith("•"):
                desc.append(lines[j])
                j += 1

            experiences.append({
                "company": company,
                "role": role,
                "duration": duration,
                "description": desc
            })

            i = j
        else:
            i += 1

    return experiences
def parse_projects(text):
    projects = []

    lines = text.split("\n")
    current = {}

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Title detection
        if "—" in line:
            if current:
                projects.append(current)
                current = {}
            current["title"] = line

        elif "github.com" in line:
            current["github"] = line

        elif line.startswith("•"):
            current.setdefault("description", []).append(line)

    if current:
        projects.append(current)

    return projects
def parse_education(text):
    education = []

    blocks = text.split("\n\n")

    for block in blocks:
        if "university" in block.lower() or "berkeley" in block.lower():
            education.append({
                "details": block.strip()
            })

    return education
def parse_sections_data(sections):
    parsed = {}

    if "skills" in sections:
        parsed["skills"] = parse_skills(sections["skills"])

    if "experience" in sections:
        parsed["experience"] = parse_experience(sections["experience"])

    if "projects" in sections:
        parsed["projects"] = parse_projects(sections["projects"])

    if "education" in sections:
        parsed["education"] = parse_education(sections["education"])

    return parsed