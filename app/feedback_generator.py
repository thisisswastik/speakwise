def generate_feedback(metrics, matched_keywords, total_keywords):
    feedback = []

    if metrics['filler_count'] > 3:
        feedback.append("Try to reduce the number of filler words.")

    if metrics['sentiment'] == 'NEGATIVE':
        feedback.append("Your answer sounds negative. Try to sound more confident.")

    if metrics['grammar_issues'] > 2:
        feedback.append("Consider improving grammar and sentence structure.")

    if metrics['vocabulary_score'] < 4:
        feedback.append("Try using more diverse or descriptive vocabulary.")

    if len(matched_keywords) < total_keywords:
        missing = total_keywords - len(matched_keywords)
        feedback.append(f"Consider mentioning {missing} more relevant keyword(s) from the job description.")

    return feedback
