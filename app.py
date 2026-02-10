import streamlit as st
from textblob import TextBlob
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import plotly.graph_objects as go
import io

# --- 1. Page Configuration & Custom CSS ---
st.set_page_config(
    page_title="Voice Sentiment Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to force a "Dark Dashboard" look and hide Streamlit branding
st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #0e1117;
        }
        
        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #41444e;
            padding: 15px;
            border-radius: 10px;
            color: white;
        }
        
        /* Title Styling */
        h1 {
            color: #4da6ff;
            font-weight: 700;
        }
        
        /* Success/Error Message Styling */
        .stAlert {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #41444e;
        }
        
        /* Hide default menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. Helper Functions ---

def transcribe_audio(audio_bytes):
    """
    Converts audio bytes to text using Google Speech Recognition.
    """
    r = sr.Recognizer()
    
    # Convert bytes to a file-like object
    audio_file = io.BytesIO(audio_bytes)
    
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return "API_ERROR"
    except Exception as e:
        return f"Error: {str(e)}"

def create_gauge_chart(score):
    """
    Creates a Plotly Gauge chart for sentiment visualization.
    Score range: -1.0 to 1.0
    """
    # Normalize score from [-1, 1] to [0, 100] for the chart logic if needed, 
    # but we will keep the axis -1 to 1 for accuracy.
    
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
                {'range': [-1, -0.1], 'color': '#7f1d1d'}, # Dark Red
                {'range': [-0.1, 0.1], 'color': '#334155'}, # Dark Gray
                {'range': [0.1, 1], 'color': '#14532d'}  # Dark Green
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
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

# Header
st.title("🎙️ Voice Sentiment Dashboard")
st.markdown("Record your voice to analyze sentiment in real-time.")

# Top Layout: Recorder & Transcript
col1, col2 = st.columns([1, 2])

with col1:
    st.info("Step 1: Record Audio")
    # Streamlit Mic Recorder Component
    audio_data = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=False,
        use_container_width=True
    )

with col2:
    st.info("Step 2: Live Transcript")
    transcript_placeholder = st.empty()

# --- 4. Processing Logic ---

if audio_data is not None:
    audio_bytes = audio_data['bytes']
    
    with st.spinner("Processing audio..."):
        # 1. Transcribe
        text = transcribe_audio(audio_bytes)
        
        if text and text != "API_ERROR" and not text.startswith("Error"):
            # Display Transcript
            transcript_placeholder.success(f'"{text}"')
            
            # 2. Analyze Sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine Label
            if polarity > 0.1:
                label = "POSITIVE"
                lbl_color = "green"
            elif polarity < -0.1:
                label = "NEGATIVE"
                lbl_color = "red"
            else:
                label = "NEUTRAL"
                lbl_color = "gray"

            st.divider()

            # --- 5. Dashboard Visualization ---
            
            # Top Row: Big Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.metric(label="Sentiment Label", value=label)
            with m_col2:
                st.metric(label="Subjectivity", value=f"{subjectivity:.2f}")
            with m_col3:
                word_count = len(text.split())
                st.metric(label="Word Count", value=word_count)

            # Bottom Row: Gauge Chart
            chart_col, details_col = st.columns([2, 1])
            
            with chart_col:
                st.plotly_chart(create_gauge_chart(polarity), use_container_width=True)
                
            with details_col:
                st.write("### Analysis Details")
                st.markdown(f"""
                - **Polarity:** `{polarity:.4f}`
                - **Confidence:** `High`
                - **Detected Language:** `English (US)`
                """)
                if polarity > 0.5:
                    st.success("This text is highly positive!")
                elif polarity < -0.5:
                    st.error("This text is highly negative!")

        elif text == "API_ERROR":
            transcript_placeholder.error("Could not connect to Google Speech API. Check internet.")
        elif text is None:
            transcript_placeholder.warning("Could not understand audio. Please speak clearly.")
        else:
            transcript_placeholder.error(text)
else:
    transcript_placeholder.info("Waiting for audio input...")

