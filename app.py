import streamlit as st
from textblob import TextBlob
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment
import plotly.graph_objects as go
import io

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="Voice Sentiment Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Mode
st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #41444e;
            padding: 15px;
            border-radius: 10px;
            color: white;
        }
        h1 { color: #4da6ff; font-weight: 700; }
        .stAlert { background-color: #262730; color: #fafafa; border: 1px solid #41444e; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. Helper Functions ---

def convert_bytes_to_wav(audio_bytes):
    """
    Converts audio bytes (WebM/OGG) to WAV format using Pydub.
    """
    try:
        # Load the audio bytes (Pydub handles format detection)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Export to WAV
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        st.error(f"Error converting audio format: {e}")
        return None

def transcribe_audio(audio_bytes):
    """
    Transcribes audio using Google Speech Recognition.
    """
    r = sr.Recognizer()
    
    # CRITICAL FIX: Convert WebM bytes to WAV first
    wav_io = convert_bytes_to_wav(audio_bytes)
    
    if wav_io is None:
        return "FORMAT_ERROR"
    
    try:
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "NO_SPEECH"
    except sr.RequestError:
        return "API_ERROR"
    except Exception as e:
        return f"Error: {str(e)}"

def create_gauge_chart(score):
    if score > 0.1:
        bar_color = "#22c55e" # Green
    elif score < -0.1:
        bar_color = "#ef4444" # Red
    else:
        bar_color = "#94a3b8" # Gray

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Polarity Score", 'font': {'size': 24, 'color': "white"}},
        number = {'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': bar_color},
            'bgcolor': "#1f2937",
            'borderwidth': 2,
            'bordercolor': "#374151",
            'steps': [
                {'range': [-1, -0.1], 'color': '#7f1d1d'},
                {'range': [-0.1, 0.1], 'color': '#334155'},
                {'range': [0.1, 1], 'color': '#14532d'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#0e1117", 
        font={'color': "white", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- 3. Main Application Layout ---

st.title("🎙️ Voice Sentiment Dashboard")
st.markdown("Record your voice to analyze sentiment in real-time.")

col1, col2 = st.columns([1, 2])

with col1:
    st.info("Step 1: Record Audio")
    audio_data = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=False,
        use_container_width=True
    )

with col2:
    st.info("Step 2: Analysis")
    transcript_placeholder = st.empty()

# --- 4. Processing Logic ---

if audio_data is not None:
    audio_bytes = audio_data['bytes']
    
    with st.spinner("Processing audio..."):
        text = transcribe_audio(audio_bytes)
        
        if text and text not in ["API_ERROR", "NO_SPEECH", "FORMAT_ERROR"] and not text.startswith("Error"):
            transcript_placeholder.success(f'"{text}"')
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                label = "POSITIVE"
            elif polarity < -0.1:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            st.divider()
            
            # Dashboard
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.metric(label="Sentiment", value=label)
            with m_col2: st.metric(label="Subjectivity", value=f"{subjectivity:.2f}")
            with m_col3: st.metric(label="Word Count", value=len(text.split()))

            chart_col, details_col = st.columns([2, 1])
            with chart_col: st.plotly_chart(create_gauge_chart(polarity), use_container_width=True)
            with details_col:
                st.write("### Details")
                st.markdown(f"- **Polarity:** `{polarity:.4f}`")

        elif text == "NO_SPEECH":
            transcript_placeholder.warning("No speech detected. Please speak clearly.")
        elif text == "API_ERROR":
            transcript_placeholder.error("Could not connect to Google Speech API.")
        elif text == "FORMAT_ERROR":
            transcript_placeholder.error("Error converting audio format.")
        else:
            transcript_placeholder.error(text)


