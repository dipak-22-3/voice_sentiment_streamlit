import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from textblob import TextBlob
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
from datetime import datetime

st.set_page_config(page_title="Voice Emotion Analyzer", layout="centered")

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- AUDIO PROCESSOR ----------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_data = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.audio_data.extend(audio.flatten())
        return frame

# ---------------- FUNCTIONS ----------------
def extract_features(y, sr=22050):
    energy = np.mean(librosa.feature.rms(y=y))
    pitch = np.mean(librosa.feature.zero_crossing_rate(y))
    return energy, pitch

def text_sentiment(text):
    return TextBlob(text).sentiment.polarity

def classify_emotion(text_score, energy):
    if text_score > 0.3 and energy > 0.03:
        return "Happy 😊"
    elif text_score < -0.3 and energy > 0.03:
        return "Angry 😠"
    elif text_score < -0.2 and energy < 0.02:
        return "Sad 😢"
    else:
        return "Calm 😐"

# ---------------- UI ----------------
st.title("🎤 Voice Sentiment & Emotion Analyzer")
st.caption("Real-time emotion detection using voice + NLP")

ctx = webrtc_streamer(
    key="voice",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor and len(ctx.audio_processor.audio_data) > 22050:
    y = np.array(ctx.audio_processor.audio_data).astype(np.float32)
    ctx.audio_processor.audio_data = []

    energy, pitch = extract_features(y)
    text_score = 0  # Real-time mic text not reliable → emotion mainly voice-based

    emotion = classify_emotion(text_score, energy)

    score = (text_score * 0.6) + (energy * 40)

    st.session_state.history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "emotion": emotion,
        "score": score
    })

    # -------- DASHBOARD --------
    st.subheader("📊 Live Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Emotion", emotion)
    col2.metric("Energy", f"{energy:.4f}")
    col3.metric("Confidence", "High" if energy > 0.03 else "Low")

    # -------- HISTORY CHART --------
    st.subheader("📈 Emotion Trend")
    scores = [h["score"] for h in st.session_state.history]

    fig, ax = plt.subplots()
    ax.plot(scores, marker="o")
    ax.axhline(0)
    ax.set_ylabel("Emotion Score")
    ax.set_xlabel("Time")
    st.pyplot(fig)
    
