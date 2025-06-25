# 🎧 Audio Emotion Classification using XGBoost

This project implements a complete pipeline for classifying emotions from speech using machine learning. It includes a fully functional *Streamlit web app* that allows users to upload ⁠ .wav ⁠ audio files and get real-time emotion predictions using a pre-trained *XGBoost* model.

---

## 🚀 Features

•⁠  ⁠Upload a ⁠ .wav ⁠ file through the web interface
•⁠  ⁠Extracts MFCC, Chroma, and Mel Spectrogram features
•⁠  ⁠Predicts emotion using a trained ⁠ XGBoost ⁠ model
•⁠  ⁠Displays the predicted emotion in a clean UI
•⁠  ⁠Lightweight and fast — suitable for real-time demos

---

## 🎯 Emotions Recognized

The model classifies speech into the following 8 emotions:

•⁠  ⁠😠 Angry  
•⁠  ⁠😌 Calm  
•⁠  ⁠🤢 Disgust  
•⁠  ⁠😨 Fearful  
•⁠  ⁠😄 Happy  
•⁠  ⁠😐 Neutral  
•⁠  ⁠😢 Sad  
•⁠  ⁠😲 Surprise  

---

## 🧠 Model Details

•⁠  ⁠Model: ⁠ XGBClassifier ⁠ from ⁠ xgboost ⁠
•⁠  ⁠Trained on MFCC, Chroma, and Mel-spectrogram features
•⁠  ⁠Input: ⁠ .wav ⁠ audio sampled at 16 kHz
•⁠  ⁠Saved as: ⁠ xgb_model.json ⁠

---

## 📁 Project Structure
| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.86      | 0.89   | 0.87     | 38.18   |
| 1                | 0.95      | 0.97   | 0.96     | 38.18   |
| 2                | 0.75      | 0.94   | 0.83     | 38.18   |
| 3                | 0.88      | 0.90   | 0.89     | 39.18   |
| 4                | 0.83      | 0.74   | 0.78     | 39.18   |
| 5                | 0.97      | 0.76   | 0.85     | 19.18   |
| 6                | 0.74      | 0.68   | 0.71     | 38.18   |
| 7                | 0.76      | 0.72   | 0.74     | 39.18   |
| **Accuracy**     |           |        | 0.83     | 0.83    |
| **Macro Avg**    | 0.84      | 0.83   | 0.83     | 288.18  |
| **Weighted Avg** | 0.83      | 0.83   | 0.83     | 288.18  |

