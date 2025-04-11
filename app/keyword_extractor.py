from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import re

# Load models once
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Extract Keywords from Text
# -----------------------------
def extract_keywords(text, top_n=20):
    """
    Extract keywords from raw text using KeyBERT
    Args:
        text (str): Input text to analyze
        top_n (int): Number of keywords to extract
    Returns:
        list: Extracted keywords
    """
    try:
        # Clean text: remove punctuation except hyphens
        cleaned_text = re.sub(r'[^\w\s-]', ' ', text.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Extract keywords
        keywords = kw_model.extract_keywords(
            cleaned_text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=top_n
        )

        # Filter and format keywords
        filtered_keywords = [kw[0].strip() for kw in keywords 
                           if len(kw[0]) > 2 and kw[1] > 0.2]

        print("✅ Extracted Keywords:", filtered_keywords)
        return filtered_keywords

    except Exception as e:
        print(f"❌ Error extracting keywords: {str(e)}")
        return []

# -----------------------------
# Semantic Matching Logic
# -----------------------------
def keyword_match(keywords, answer, threshold=0.35):
    """
    Match keywords against answer using semantic similarity
    Args:
        keywords (list): Keywords to match against
        answer (str): User's transcript text
        threshold (float): Similarity threshold (0-1)
    Returns:
        list: Matched keywords
    """
    matched = []
    
    try:
        if not answer.strip() or not keywords:
            return matched

        # Clean input lightly (retain structure)
        answer_clean = re.sub(r'[^\w\s.,;!?]', '', answer.lower())

        # Break answer into chunks
        answer_chunks = sent_tokenize(answer_clean)
        
        if not answer_chunks:
            return matched

        # Generate embeddings
        chunk_embeddings = semantic_model.encode(answer_chunks, convert_to_tensor=True)
        keyword_embeddings = semantic_model.encode(keywords, convert_to_tensor=True)

        # Find matches
        for i, kw_emb in enumerate(keyword_embeddings):
            scores = util.pytorch_cos_sim(kw_emb, chunk_embeddings)[0]
            if scores.max().item() >= threshold:
                matched.append(keywords[i])

        print("✅ Matched Keywords:", matched)
        return matched

    except Exception as e:
        print(f"❌ Error in keyword matching: {str(e)}")
        return []