# MARSproject_23123017
This project focuses on emotion classification from audio recordings using robust tree-based ensemble models. It begins with thorough pre-processing of audio signals, including mono conversion, resampling, and the extraction of meaningful features such as MFCCs, chroma, mel spectrograms, zero-crossing rate, spectral contrast, and tonnetz using the Librosa library. To address class imbalance in the dataset, SMOTE (Synthetic Minority Oversampling Technique) is applied to the training data. Three powerful gradient boosting models—CatBoost, LightGBM, and XGBoost—are trained using the balanced features, and their performance is evaluated on a held-out test set. Finally, a soft-voting ensemble model combines their predictions to improve overall accuracy and F1-score, and the best-performing model is saved based on macro F1 performance and compact file size.



