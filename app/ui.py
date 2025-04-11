import streamlit as st
from app.nlp_pipeline import analyze_transcript
from app.keyword_extractor import extract_keywords, keyword_match
from app.feedback_generator import generate_feedback
from app.audio_input import transcribe_audio
from app.record_audio import record_audio
from app.resume_matcher import compute_similarity as resume_similarity, extract_missing_keywords
from app.contextual_eval import generate_ideal_answers, compute_similarity as contextual_similarity, visualize_alignment

def launch_app():
    st.set_page_config(page_title="SpeakWise", layout="centered")
    st.title("🎙️ SpeakWise - Career Success Toolkit")

    # Initialize session state
    for key, val in {
        'jd_text': None,
        'resume_text': None,
        'recording_status': 'idle',
        'audio_duration': 60
    }.items():
        st.session_state.setdefault(key, val)

    # --- Tabs
    tab_resume, tab_interview, tab_context = st.tabs(
        ["📄 Resume Review", "🎙️ Interview Feedback", "🦙 Contextual Evaluation"]
    )

    # --- Resume Tab ---
    with tab_resume:
        st.header("📄 Resume & JD Analysis")

        uploaded_jd = st.file_uploader("Upload Job Description (TXT)", type="txt", key="resume_jd_upload")
        if uploaded_jd:
            st.session_state.jd_text = uploaded_jd.read().decode("utf-8")
            st.success("✅ Job Description uploaded!")

        uploaded_resume = st.file_uploader("Upload Resume (TXT/PDF)", type=["txt", "pdf"], key="resume_upload")
        if uploaded_resume:
            try:
                if uploaded_resume.type == "application/pdf":
                    import fitz
                    with fitz.open(stream=uploaded_resume.read(), filetype="pdf") as doc:
                        st.session_state.resume_text = "\n".join([page.get_text() for page in doc])
                else:
                    st.session_state.resume_text = uploaded_resume.read().decode("utf-8")
                st.success("✅ Resume uploaded!")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

        if st.session_state.jd_text and st.session_state.resume_text:
            st.subheader("🔍 Analysis Report")
            with st.spinner("Analyzing resume..."):
                try:
                    similarity = resume_similarity(st.session_state.resume_text, st.session_state.jd_text)
                    missing_keywords = extract_missing_keywords(st.session_state.resume_text, st.session_state.jd_text)

                    col1, col2 = st.columns(2)
                    col1.metric("JD Match Score", f"{similarity:.1f}%")
                    col2.metric("Missing Keywords", len(missing_keywords))

                    with st.expander("🔍 View Missing Keywords"):
                        st.write("These keywords from the JD are missing/mismatched in your resume:")
                        st.write(missing_keywords)

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

    # --- Interview Feedback Tab ---
    with tab_interview:
        st.header("🎙️ Interview Practice & Analysis")

        if not st.session_state.jd_text:
            uploaded_jd = st.file_uploader("📄 Upload Job Description (TXT)", type="txt", key="interview_jd_upload")
            if uploaded_jd:
                st.session_state.jd_text = uploaded_jd.read().decode("utf-8")
                st.success("✅ Job Description uploaded!")

        st.subheader("🎤 Input Options")
        transcript = ""

        col_upload, col_record = st.columns(2)

        with col_upload:
            uploaded_audio = st.file_uploader("Upload Audio (.wav)", type=["wav"], key="audio_upload")
            if uploaded_audio:
                try:
                    with open("temp_audio.wav", "wb") as f:
                        f.write(uploaded_audio.read())
                    audio_data = transcribe_audio("temp_audio.wav")
                    transcript = audio_data['text']
                    st.session_state.audio_duration = audio_data['duration']
                    st.success("✅ Audio processed")
                except Exception as e:
                    st.error(f"❌ Audio processing failed: {e}")

        with col_record:
            duration = st.slider("Recording duration (seconds)", 10, 120, 30, key="rec_duration")
            if st.button("🎤 Start Recording", key="rec_button"):
                try:
                    st.session_state.recording_status = "recording"
                    with st.spinner(f"Recording for {duration} seconds..."):
                        audio_path = record_audio(duration)
                        audio_data = transcribe_audio(audio_path)
                        transcript = audio_data['text']
                        st.session_state.audio_duration = audio_data['duration']
                        st.session_state.recording_status = "complete"
                        st.success("✅ Recording complete")
                except Exception as e:
                    st.error(f"❌ Recording failed: {e}")
                    st.session_state.recording_status = "error"

        if not (uploaded_audio or transcript):
            st.session_state.audio_duration = st.number_input(
                "Answer duration (seconds)", min_value=10, max_value=300, value=60, key="manual_duration"
            )

        st.subheader("📝 Transcript Editor")
        user_input = st.text_area(
            "Edit your transcript (autofilled from audio if present):",
            value=transcript,
            height=200,
            key="transcript_editor"
        )

        if st.session_state.jd_text and user_input.strip():
            st.subheader("📊 Analysis Results")
            with st.spinner("Analyzing your response..."):
                try:
                    metrics = analyze_transcript(user_input, st.session_state.audio_duration)
                    keywords = extract_keywords(st.session_state.jd_text)
                    matched = keyword_match(keywords, user_input)
                    feedback = generate_feedback(metrics, matched, len(keywords))

                    metric_data = [
                        ("Fluency Score", f"{metrics['fluency_score']:.1f}%", metrics['fluency_score'] > 0),
                        ("Keyword Coverage", f"{len(matched)}/{len(keywords)}", True),
                        ("Speech Pace", f"{metrics['speech_pace_wpm']} WPM", True),
                        ("Filler Words", metrics['filler_count'], True),
                        ("Grammar Issues", metrics['grammar_issues'], True)
                    ]

                    valid_metrics = [md for md in metric_data if md[2]]
                    metric_cols = st.columns(len(valid_metrics))
                    for i, (label, value, _) in enumerate(valid_metrics):
                        metric_cols[i].metric(label, value)

                    tab_fb, tab_kw = st.tabs(["💡 Feedback", "🔑 Keywords"])
                    with tab_fb:
                        st.subheader("Improvement Suggestions")
                        for s in feedback:
                            st.write(f"- {s}")

                    with tab_kw:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Matched Keywords")
                            st.write(matched or "No keywords matched")
                        with col2:
                            missing = list(set(keywords) - set(matched))
                            st.subheader("Missing Keywords")
                            st.write(missing or "All keywords covered!")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

    # --- Contextual Evaluation Tab ---
    with tab_context:
        st.header("🦙 LLM-Based Answer Evaluation")

        jd_text = st.text_area("📄 Paste Job Description", placeholder="Paste the JD here...", height=150)
        user_answer = st.text_area("🧑‍💼 Paste Your Interview Answer", placeholder="Paste your answer here...", height=150)

        if st.button("🧠 Evaluate with LLaMA"):
            if jd_text and user_answer:
                with st.spinner("Generating ideal answer using LLaMA-2..."):
                    ideal_answer = generate_ideal_answers(jd_text)
                    st.success("✅ Ideal answer generated.")

                    st.markdown("### ✅ Ideal Answer")
                    st.write(ideal_answer)

                    matched, missing, scores = contextual_similarity(user_answer, ideal_answer)

                    st.markdown(f"🔍 **Matched Sentences:** {len(matched)}")
                    st.markdown(f"❌ **Missing Sentences:** {len(missing)}")

                    if matched:
                        st.markdown("#### ✅ Matched")
                        for s, score in matched:
                            st.markdown(f"<span style='color:green'>✔ {s} ({score:.2f})</span>", unsafe_allow_html=True)

                    if missing:
                        st.markdown("#### ❌ Missing")
                        for s, score in missing:
                            st.markdown(f"<span style='color:red'>✘ {s} ({score:.2f})</span>", unsafe_allow_html=True)

                    visualize_alignment(ideal_answer, scores)
            else:
                st.warning("Please provide both JD and your answer.")


if __name__ == "__main__":
    launch_app()
