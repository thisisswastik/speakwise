# SpeakWise - AI-Powered Career Success Toolkit

**SpeakWise** is an AI-driven platform designed to help job seekers improve their interview skills, analyze resumes, and get personalized feedback. It uses state-of-the-art models for natural language processing (NLP) and contextual evaluation to assess both the candidate's resume and their interview responses.

---

## Features

### 1. **Resume & Job Description Analysis**
   - Upload your **resume** and **job description (JD)**.
   - **Resume match score**: Get an assessment of how well your resume aligns with the job description.
   - **Missing keywords**: See which critical keywords are missing in your resume.

### 2. **Interview Feedback**
   - Upload your **audio interview responses**.
   - The platform generates feedback based on fluency, keyword coverage, speech pace, grammar, and filler words.
   - You can also **edit the transcript** if needed and get real-time suggestions to improve your responses.

### 3. **Contextual Evaluation**
   - Paste your **job description** and **interview answer**.
   - Get an **ideal answer** generated using an AI model.
   - Evaluate your answer's **alignment with the ideal answer** and identify **missing sentences** or areas for improvement.

---

## Technologies Used

- **Streamlit**: Frontend web framework for fast and interactive app development.
- **Sentence-Transformers**: Used for semantic similarity and matching answers with job descriptions.
- **Whisper**: Open-source speech recognition system for transcribing interview audio.
- **Transformers (Hugging Face)**: For utilizing pre-trained models in NLP tasks.
- **scikit-learn**: For machine learning tasks like clustering, feature extraction, etc.
- **OpenAI (Alternative)**: To generate ideal answers for interview feedback, using contextual models.
  
---

## Setup & Installation

Follow the steps below to run the project locally.

### Prerequisites
Make sure you have Python 3.7+ installed. You also need a **Replicate API Token** to use the LLaMA-based contextual evaluation.

### 1. Clone the repository:

```bash
git clone https://github.com/thisisswastik/speakwise.git
cd speakwise
```
### 2. Install the requirements.txt

```bash
pip install -r requirements.txt
```
### 3. Set up your Replicate API Token:
Visit Replicate API Tokens to create a new API token.

Paste the token in the API Token input section in the app.

### 4. Run the app:
``` bash
streamlit run main.py
```



