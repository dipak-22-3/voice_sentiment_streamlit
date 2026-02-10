import streamlit as st
import librosa
import numpy as np
import tempfile
import time
from textblob import TextBlob
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Voice Emotion Analyzer",
    layout="centered"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# -------------------------------------------------
# THEME
# -------------------------------------------------
if st.session_state.theme == "dark":
    st.markdown(
        "<style>body { background-color:#0e1117; color:white; }</style>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<style>body { background-color:white; color:black; }</style>",
        unsafe_allow_html=True
    )

# -------------------------------------------------
# TOP BAR
# -------------------------------------------------
col1, col2 = st.columns([5,1])
with col1:
    st.markdown("## 🎤 Voice Emotion Analyzer")
with col2:
    if st.button("🌗"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# -------------------------------------------------
# VOICE FEATURE EXTRACTION
# -------------------------------------------------
def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path)

    energy = float(np.mean(librosa.feature.rms(y=y)))

    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

    duration = librosa.get_duration(y=y, sr=sr)
    return energy, pitch_mean, duration

# -------------------------------------------------
# EMOTION CLASSIFICATION (HUMAN-LIKE)
# -------------------------------------------------
def classify_emotion(energy, pitch, speech_rate, text_sentiment):
    score = (
        0.45 * energy +
        0.25 * (pitch / 300 if pitch > 0 else 0) +
        0.20 * speech_rate +
        0.10 * (text_sentiment + 1) / 2
    )

    if energy > 0.06 and pitch > 180:
        emotion = "Angry 😠"
        color = "#ff4d4d"
    elif speech_rate < 1.5 and energy < 0.04:
        emotion = "Calm 😌"
        color = "#4da6ff"
    else:
        emotion = "Neutral 😐"
        color = "#cccccc"

    return emotion, score, color

# -------------------------------------------------
# CIRCULAR EMOTION METER
# -------------------------------------------------
def emotion_meter(score, emotion, color):
    percent = min(max(score, 0), 1) * 100

    svg = f"""
    <div style="display:flex;justify-content:center;">
    <svg width="220" height="220" viewBox="0 0 36 36">
      <path d="M18 2.0845
               a 15.9155 15.9155 0 0 1 0 31.831
               a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none" stroke="#333" stroke-width="3"/>
      <path d="M18 2.0845
               a 15.9155 15.9155 0 0 1 0 31.831"
            fill="none" stroke="{color}" stroke-width="3"
            stroke-dasharray="{percent},100"/>
      <text x="18" y="18" text-anchor="middle" fill="white" font-size="4">{emotion}</text>
      <text x="18" y="23" text-anchor="middle" fill="white" font-size="3">{score:.2f}</text>
    </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)

# -------------------------------------------------
# UI: AUDIO UPLOAD
# -------------------------------------------------
st.markdown("### 🎧 Upload your voice (.wav or .mp3)")
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    energy, pitch, duration = extract_voice_features(audio_path)

    # Text sentiment optional (no mic, so empty)
    text = ""
    text_sentiment = TextBlob(text).sentiment.polarity

    words = 10  # neutral baseline
    speech_rate = words / duration if duration > 0 else 0

    emotion, score, color = classify_emotion(
        energy, pitch, speech_rate, text_sentiment
    )

    st.session_state.history.append({
        "time": time.strftime("%H:%M:%S"),
        "score": score,
        "emotion": emotion
    })

    # OUTPUT
    st.markdown("## 🎯 Emotion Analysis")
    emotion_meter(score, emotion, color)

    st.markdown("### 🔍 Voice Details")
    st.write(f"Energy: {energy:.3f}")
    st.write(f"Pitch: {pitch:.1f} Hz")
    st.write(f"Speech Rate: {speech_rate:.2f}")

# -------------------------------------------------
# HISTORY
# -------------------------------------------------
if st.session_state.history:
    st.markdown("## 📈 Emotion History")

    scores = [h["score"] for h in st.session_state.history]

    fig, ax = plt.subplots()
    ax.plot(scores, marker="o")
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Emotion Score")
    st.pyplot(fig)
    
