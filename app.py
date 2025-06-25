import streamlit as st
import librosa
import numpy as np
from xgboost import XGBClassifier
import tempfile
import warnings

warnings.filterwarnings("ignore")


# 🧠 Load the XGBoost Model
@st.cache_resource
def load_model(path="xgb_model.json"):
    model = XGBClassifier()
    model.load_model(path)
    return model


# 🎯 Load model once
xgb_model = load_model()

# 🎭 Emotion Labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

# 🎼 Audio Feature Extraction
def extract_feature(file_path, n_mfcc=40, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels).T, axis=0)

    return np.hstack([mfcc, chroma, mel])


# 🎙️ Predict Emotion
def predict_emotion(audio_path):
    features = extract_feature(audio_path)
    features = features.reshape(1, -1)
    pred_idx = xgb_model.predict(features)[0]
    return emotion_labels[pred_idx]


# 🌐 App Config
st.set_page_config(page_title="Emotion Classifier", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🎧 Emotion Classifier")
    st.markdown(
        """
        Upload a `.wav` file to detect the emotion expressed in the audio.

        **Features used**:
        - MFCC
        - Chroma
        - Mel Spectrogram

        **Model**: XGBoost (`xgb_model.json`)
        """
    )

# Main Layout
st.markdown("<h2 style='color: #5A5A5A;'>🔊 Audio Emotion Detection Dashboard</h2>", unsafe_allow_html=True)
st.markdown("---")

# Upload Zone
uploaded_file = st.file_uploader("📁 Choose a `.wav` file", type=["wav"])

# If audio is uploaded
if uploaded_file:
    st.markdown("### 🎧 Audio Preview")
    st.audio(uploaded_file, format='audio/wav')

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("⏳ Processing..."):
        try:
            emotion = predict_emotion(tmp_path)
            # Result Card
            st.markdown(
                f"""
                <div style="border:2px solid #4CAF50; border-radius:10px; padding:20px; background-color:#F0FFF0;">
                    <h3 style="color:#333;">🎯 Predicted Emotion:</h3>
                    <h1 style="color:#FF5733;">{emotion.upper()}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

else:
    st.info("⬆️ Please upload a `.wav` file to begin.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit & XGBoost</p>",
    unsafe_allow_html=True
)
