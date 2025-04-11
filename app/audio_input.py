# app/audio_input.py

import speech_recognition as sr
import wave
import contextlib
import os

def transcribe_audio(file_path):
    """Transcribe audio file and return text + metadata"""
    result = {
        "text": "",
        "duration": 0,
        "error": None
    }
    
    try:
        # Get audio duration
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            result["duration"] = frames / float(rate)
            
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Process audio file
        with sr.AudioFile(file_path) as source:
            audio = r.record(source)  # Read entire file
            
        # Try Google Speech Recognition first
        try:
            result["text"] = r.recognize_google(audio)
        except sr.UnknownValueError:
            # Fallback to whisper (requires ffmpeg)
            result["text"] = r.recognize_whisper(
                audio,
                model="small",
                load_options=dict(device="cpu")
            )
            
    except sr.RequestError as e:
        result["error"] = f"API unavailable: {str(e)}"
    except Exception as e:
        result["error"] = f"Transcription error: {str(e)}"
        
    return result