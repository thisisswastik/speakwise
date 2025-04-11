def suggest_resume_bullets(resume_lines, missing_keywords):
    suggestions = []

    for keyword in missing_keywords:
        for line in resume_lines:
            if any(term in line.lower() for term in ["project", "system", "model", "built", "designed"]):
                new_line = f"{line} Incorporated keyword: '{keyword}' for alignment with JD."
                suggestions.append((keyword, line, new_line))
                break

    return suggestions
