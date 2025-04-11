import replicate
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk import sent_tokenize

# Ensure NLTK punkt tokenizer is available
nltk.download('punkt')

# Load BERT model once
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# -----------------------------
# 1. Generate Ideal Answer using LLaMA-2
# -----------------------------
def generate_ideal_answers(jd_text):
    prompt = f"""
    You are skilled in generating ideal answers for job interviews based on job descriptions.
    Your task is to provide a comprehensive and relevant answer to the given below job description.

    Job Description:
    {jd_text}

    Ideal Answer:
    """

    try:
        output = replicate.run(
            "meta/llama-2-13b-chat",
            input={
                "prompt": prompt,
                "temperature": 0.6,
                "top_p": 0.9,
                "max_new_tokens": 300,
                "top_k":10
            }
        )
        return "".join(output)
    except Exception as e:
        return f"[LLaMA-2 Error] {e}"


# -----------------------------
# 2. Compute Semantic Similarity
# -----------------------------
def compute_similarity(user_answer, ideal_answer, threshold=0.4):
    user_sentences = sent_tokenize(user_answer)
    ideal_sentences = sent_tokenize(ideal_answer)

    user_embeddings = bert_model.encode(user_sentences, convert_to_tensor=True)
    ideal_embeddings = bert_model.encode(ideal_sentences, convert_to_tensor=True)

    matched = []
    missing = []
    sentence_scores = []

    for i, ideal_emb in enumerate(ideal_embeddings):
        scores = util.pytorch_cos_sim(ideal_emb, user_embeddings)[0]
        max_score = scores.max().item()
        sentence_scores.append(max_score)

        if max_score >= threshold:
            matched.append((ideal_sentences[i], max_score))
        else:
            missing.append((ideal_sentences[i], max_score))

    return matched, missing, sentence_scores


# -----------------------------
# 3. Visualize Alignment
# -----------------------------
def visualize_alignment(ideal_answer, sentence_scores, threshold=0.4):
    ideal_sentences = sent_tokenize(ideal_answer)
    output_html = ""

    for sentence, score in zip(ideal_sentences, sentence_scores):
        color = "green" if score >= threshold else "red"
        opacity = str(min(1, max(0.4, score)))  # fade weak matches
        output_html += f'<span style="color:{color}; opacity:{opacity}; font-weight:bold;">{sentence}</span><br><br>'

    st.markdown("### 🔍 Semantic Match Visualization (Ideal Answer)")
    st.markdown(output_html, unsafe_allow_html=True)
