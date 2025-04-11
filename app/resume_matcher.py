# app/resume_matcher.py

import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small and fast, good for similarity tasks

def semantic_keyword_match(jd_keywords, resume_sentences, threshold=0.7):
    matched, missing = [], []

    for keyword in jd_keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        found = False

        for sentence in resume_sentences:
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(keyword_embedding, sentence_embedding)
            if similarity.item() >= threshold:
                matched.append(keyword)
                found = True
                break

        if not found:
            missing.append(keyword)

    return matched, missing

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def compute_similarity(resume_text, jd_text):
    resume_clean = preprocess_text(resume_text)
    jd_clean = preprocess_text(jd_text)

    resume_embedding = model.encode(resume_clean, convert_to_tensor=True)
    jd_embedding = model.encode(jd_clean, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(resume_embedding, jd_embedding)[0][0].item()
    return round(similarity_score * 100, 2)

def extract_missing_keywords(resume_text, jd_text, top_k=10):
    resume_words = set(preprocess_text(resume_text).split())
    jd_words = list(set(preprocess_text(jd_text).split()))

    # Score each JD word based on its relevance with resume
    missing_scores = []
    for word in jd_words:
        if word not in resume_words:
            word_embedding = model.encode(word, convert_to_tensor=True)
            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(word_embedding, resume_embedding)[0][0].item()
            missing_scores.append((word, score))

    # Sort by similarity (relevance) descending
    missing_scores.sort(key=lambda x: x[1], reverse=True)
    top_missing = [word for word, _ in missing_scores[:top_k]]
    return top_missing
