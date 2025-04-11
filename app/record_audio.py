# app/record_audio.py

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from datetime import datetime
import os

def record_audio(duration=10, sample_rate=16000, channels=1):
    """Record audio from microphone and save as WAV file"""
    try:
        # Create recordings directory if not exists
        os.makedirs("recordings", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.wav"
        
        # Show recording indicator
        print(f"🎤 Recording for {duration} seconds...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=np.int16
        )
        sd.wait()  # Wait until recording is finished
        
        # Save as WAV file
        sf.write(filename, audio_data, sample_rate)
        print(f"✅ Saved recording to: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Recording failed: {str(e)}")
        return None