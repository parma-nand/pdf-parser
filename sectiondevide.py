import re
class section : 
 @staticmethod
 def extract_sections(text):
    sections = {}
    lines = text.strip().split("\n")
    content = " ".join(lines[1:])   # remove header
    content = re.sub(r'\s+', ' ', content)
    text= content.strip()

    section_patterns = {
        "summary": r"\b(professional summary|summary|profile)\b",
        "experience": r"\b(professional experience)\b",
        "education": r"\b(education)\b",
        "skills": r"\b(skills|technical skills|skill set)\b",
        "projects": r"\b(projects|personal projects)\b",
        "certifications": r"\b(certifications|training|courses)\b",
    }

    lower_text = text.lower()
    matches = []

    for key, pattern in section_patterns.items():
        for match in re.finditer(pattern, lower_text):
            matches.append((match.start(), key))

    matches.sort()

    for i in range(len(matches)):
        start = matches[i][0]
        section_name = matches[i][1]
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)

        sections[section_name] = text[start:end].strip()

    return sections