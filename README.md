# MARSproject_23123017
This project focuses on emotion classification from audio recordings using robust tree-based ensemble models. It begins with thorough pre-processing of audio signals, including mono conversion, resampling, and the extraction of meaningful features such as MFCCs, chroma, mel spectrograms, zero-crossing rate, spectral contrast, and tonnetz using the Librosa library. To address class imbalance in the dataset, SMOTE (Synthetic Minority Oversampling Technique) is applied to the training data. Three powerful gradient boosting models—CatBoost, LightGBM, and XGBoost—are trained using the balanced features, and their performance is evaluated on a held-out test set. Finally, a soft-voting ensemble model combines their predictions to improve overall accuracy and F1-score, and the best-performing model is saved based on macro F1 performance and compact file size.


              precision    recall  f1-score   support

       angry       0.85      0.89      0.87        38
        calm       0.88      0.95      0.91        38
     disgust       0.92      0.92      0.92        38
     fearful       0.71      0.90      0.80        39
       happy       0.61      0.56      0.59        39
     neutral       0.73      0.58      0.65        19
         sad       0.76      0.74      0.75        38
    surprise       0.94      0.77      0.85        39
    accuracy                           0.80       288
    macro avg       0.80      0.79      0.79       288
    weighted avg       0.80      0.80      0.80       288
  
