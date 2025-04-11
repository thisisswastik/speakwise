import re
import nltk
import language_tool_python
from nltk.corpus import stopwords
from transformers import pipeline as hf_pipeline
import textstat

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize NLP tools
try:
    tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    print(f"Grammar checker initialization error: {str(e)}")
    tool = None

try:
    sentiment_analyzer = hf_pipeline("sentiment-analysis")
except Exception as e:
    print(f"Sentiment analyzer initialization error: {str(e)}")
    sentiment_analyzer = None

FILLERS = {"um", "uh", "like", "you know", "actually", "basically", "so", "literally"}

def analyze_transcript(text, duration_seconds=60):
    """Analyze transcript with robust error handling"""
    metrics = {
        "filler_count": 0,
        "sentiment": "NEUTRAL",
        "sentiment_score": 0.0,
        "grammar_issues": 0,
        "vocabulary_score": 0.0,
        "speech_pace_wpm": 0.0,
        "fluency_score": 0.0
    }
    
    if not text.strip():
        return metrics

    try:
        # Clean text and basic processing
        clean_text = re.sub(r'[^\w\s.,;!?]', '', text)
        words = nltk.word_tokenize(clean_text.lower())
        num_words = len(words)
        
        # Filler Word Analysis
        metrics["filler_count"] = sum(1 for word in words if word in FILLERS)

        # Sentiment Analysis with fallback
        if sentiment_analyzer:
            try:
                sentiment_result = sentiment_analyzer(clean_text[:512])[0]  # Truncate to model limit
                metrics["sentiment"] = sentiment_result['label']
                metrics["sentiment_score"] = round(sentiment_result['score'], 2)
            except Exception as e:
                print(f"Sentiment analysis error: {str(e)}")

        # Grammar Check with fallback
        if tool:
            try:
                matches = tool.check(clean_text)
                metrics["grammar_issues"] = len(matches)
            except Exception as e:
                print(f"Grammar check error: {str(e)}")

        # Vocabulary Analysis
        try:
            stop_words = set(stopwords.words("english"))
            content_words = [w for w in words if w not in stop_words and w.isalpha()]
            unique_words = set(content_words)
            if unique_words:
                avg_word_len = sum(len(w) for w in unique_words) / len(unique_words)
                metrics["vocabulary_score"] = round(avg_word_len, 2)
        except Exception as e:
            print(f"Vocabulary analysis error: {str(e)}")

        # Speech Pace Calculation
        try:
            if duration_seconds > 0:
                metrics["speech_pace_wpm"] = round(num_words / (duration_seconds / 60), 2)
        except ZeroDivisionError:
            metrics["speech_pace_wpm"] = 0.0

        # Fluency Score Calculation (Composite Metric)
        try:
            # Base score components
            filler_penalty = min(metrics["filler_count"] * 2, 20)  # Max 20% penalty
            grammar_penalty = min(metrics["grammar_issues"], 15)    # Max 15% penalty
            pace_penalty = 0
            
            # Ideal WPM range: 110-150
            if metrics["speech_pace_wpm"] < 100:
                pace_penalty = (100 - metrics["speech_pace_wpm"]) * 0.2
            elif metrics["speech_pace_wpm"] > 160:
                pace_penalty = (metrics["speech_pace_wpm"] - 160) * 0.2
                
            total_penalty = filler_penalty + grammar_penalty + pace_penalty
            metrics["fluency_score"] = max(0, 100 - total_penalty)
            
        except Exception as e:
            print(f"Fluency calculation error: {str(e)}")

    except Exception as e:
        print(f"General analysis error: {str(e)}")

    return metrics