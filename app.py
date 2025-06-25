import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from xgboost import XGBClassifier
import tempfile
import warnings

warnings.filterwarnings("ignore")

# 🎯 Load XGBoost Model
@st.cache_resource
def load_model(path="best_model.json"):
    model = XGBClassifier()
    model.load_model(path)
    return model

xgb_model = load_model()
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

# 🎼 Feature Extractor
def extract_feature(file_path, n_mfcc=40, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# 🔍 Prediction
def predict_emotion(audio_path):
    features = extract_feature(audio_path)
    features = features.reshape(1, -1)
    pred_idx = xgb_model.predict(features)[0]
    return emotion_labels[pred_idx]

# 🌐 App Config
st.set_page_config(page_title="🎵 Emotion Classifier", layout="wide")

# Custom CSS styles for color changes
st.markdown("""
    <style>
    body {
        background-color: #f9f9fc;
    }
    .stSidebar {
        background-color: #14213d;
        color: white;
    }
    .st-eb {
        color: #fca311 !important;
    }
    .stButton>button {
        background-color: #fca311;
        color: black;
    }
    .st-bx {
        background-color: #e5e5e5;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎧 Emotion Classifier")
    st.markdown(
        """
        Upload a `.wav` file to detect the emotion expressed in the audio.

        **🎚 Features Used**:
        - MFCC
        - Chroma
        - Mel Spectrogram

        **🧠 Model**: XGBoost (`xgb_model.json`)
        """, unsafe_allow_html=True
    )

# Main Heading
st.markdown("<h2 style='color: #14213d;'>🔊 Audio Emotion Detection Dashboard</h2>", unsafe_allow_html=True)
st.markdown("---")

# Upload UI
uploaded_file = st.file_uploader("📁 Choose a `.wav` file", type=["wav"])

# If file uploaded
if uploaded_file:
    st.markdown("### 🎧 Audio Preview")
    st.audio(uploaded_file, format='audio/wav')

    # Load and resample audio
    y, sr = librosa.load(uploaded_file, sr=16000)

    # Waveform
    st.markdown("#### 📉 Waveform")
    fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#1f77b4")
    ax_wave.set_title("Waveform", fontsize=12)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    st.pyplot(fig_wave)

    # Mel Spectrogram
    st.markdown("#### 📊 Mel Spectrogram")
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spect, ref=np.max)

    fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax_mel, cmap='magma')
    ax_mel.set_title("Mel Spectrogram (dB)", fontsize=12)
    fig_mel.colorbar(img, ax=ax_mel, format='%+2.0f dB')
    st.pyplot(fig_mel)

    # Save to temp file for prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Prediction
    with st.spinner("⏳ Analyzing..."):
        try:
            emotion = predict_emotion(tmp_path)
            st.markdown(
                f"""
                <div style="border: 2px solid #fca311; border-radius: 10px; padding: 20px; background-color: #fff8e7;">
                    <h3 style="color:#14213d;">🎯 Predicted Emotion:</h3>
                    <h1 style="color:#e63946;">{emotion.upper()}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("⬆️ Upload a `.wav` file to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>💡 Built with ❤️ using Streamlit & XGBoost</p>",
    unsafe_allow_html=True
)

