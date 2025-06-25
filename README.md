
🎧 Speech Emotion Recognition with XGBoost
This project delivers an end-to-end machine learning solution for detecting emotions in human speech. At its core is a fast and lightweight Streamlit application where users can upload .wav audio files and instantly receive emotion predictions using a trained XGBoost model.



WEB APP LINK - [https://marsproject23123017-zwjxcfup6bfxvdbdl29gfq.streamlit.app/](https://marsproject23123017-8hp45dqqyqsx4t5gvfzkub.streamlit.app/)
🚀 Key Highlights
🎙️ Upload .wav audio files directly via an intuitive web interface

🎼 Automatically extracts essential audio features — MFCCs, Chroma, and Mel Spectrogram

🤖 Uses a robust XGBoost model for emotion classification

📊 Clean and user-friendly output visualization

⚡ Optimized for real-time responsiveness and demos

🎯 Emotions Detected
The system can identify the following eight emotional states from speech:

 Angry
 Calm
 Disgust
 Fearful
 Happy
 Neutral
 Sad
 Surprise

## Under the Hood
Classifier: XGBClassifier from the xgboost library

Training Data: Features include MFCC, Chroma vectors, and Mel Spectrograms

Input Format: .wav audio files at 16 kHz sampling rate

Model Artifact: Saved in JSON format as best_model.json
## Project Structure
README.md 23123017_HARSHDEEPSINGH_emotion_classification.ipynb app.py requirements.txt best_model.json




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

