import pandas as pd
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Enable TensorFlow JIT optimization
tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================
# Data Loading
# =========================
CSV_PATH = "C:/Users/prana/OneDrive/Desktop/thackur/thackur/backend/KAGGLE/AUDIO/shuffled_file.csv"
df = pd.read_csv(CSV_PATH)

# =========================
# Data Preprocessing
# =========================
scaler = StandardScaler()
X = df.drop('LABEL', axis=1)  
X = scaler.fit_transform(X)
y = df['LABEL'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle data imbalance using class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y), y=y)))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Model Architecture
# =========================
model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_deepfake_model.h5", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Model training
model.fit(X_train, y_train, 
          epochs=25, 
          batch_size=32, 
          validation_split=0.2, 
          class_weight=class_weights,
          callbacks=[early_stopping, model_checkpoint, reduce_lr])

# =========================
# Enhanced Audio Feature Extraction
# =========================
def extract_features_from_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = {
            'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'rms': np.mean(librosa.feature.rms(y=y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y))
        }

        # MFCCs (corrected naming)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}'] = np.mean(mfccs[i])  # Correctly matches your dataset

        # Ensure feature consistency
        required_features = [
            'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
            'rolloff', 'zero_crossing_rate'
        ] + [f'mfcc{i+1}' for i in range(20)]

        features_df = pd.DataFrame([features])[required_features]  
        features_scaled = scaler.transform(features_df)
        return features_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# =========================
# Predict Audio Deepfake
# =========================
def predict_audio_deepfake(file_path):
    features = extract_features_from_audio(file_path)
    if features is not None:
        prediction = model.predict(features)[0][0]
        result = "REAL" if prediction < 0.5 else "FAKE"
        confidence = prediction if result == "FAKE" else 1 - prediction
        print(f"Prediction for {os.path.basename(file_path)}: {result} with {confidence:.2f}% confidence.")
    else:
        print("Failed to extract features. Please check the audio file.")

# Example usage
predict_audio_deepfake("C:/Users/prana/OneDrive/Desktop/thackur/thackur/backend/fake_fr.mp3")
