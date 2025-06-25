import streamlit as st
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import tempfile
import warnings
import os

warnings.filterwarnings("ignore")

# Load Model
@st.cache_resource
def load_model(path="best_model.json"):
    model = XGBClassifier()
    model.load_model(path)
    return model

xgb_model = load_model()

# Labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

# Feature Extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    if y.size == 0:
        raise ValueError("Empty audio file.")
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# Emotion Prediction
def predict_emotion(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    pred = xgb_model.predict(features)[0]
    return emotion_labels[pred]

# Streamlit UI
st.title("🎧 Emotion Detection from Audio")
uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)

        # Waveform
        st.subheader("Waveform")
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Waveform")
        st.pyplot(fig1)

        # Mel Spectrogram
        st.subheader("Mel Spectrogram")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mel_db, x_axis="time", y_axis="mel", sr=sr, ax=ax2, cmap="magma")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set_title("Mel Spectrogram")
        st.pyplot(fig2)

        # Prediction
        emotion = predict_emotion(file_path)
        st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a `.wav` file to start.")

