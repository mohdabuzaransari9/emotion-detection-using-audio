import streamlit as st
import numpy as np
import librosa
import joblib
import pickle
import warnings
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Paths to models and encoders (adjust paths as necessary)

MODEL_PATH = "emotion-recognition.keras"
ENCODER_PATH = 'encoder.pickle'
SCALER_PATH = 'scaler.pkl'

# Load the model, encoder, and scaler
try:
    
    model = load_model(MODEL_PATH)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model or encoder: {e}")
    st.stop()

# Feature Extraction Functions
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate):
    result = np.array([])
    # Extract features
    try:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sample_rate).T, axis=0)
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate).T, axis=0)
        centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=sample_rate).T, axis=0)
        # Concatenate all features
        result = np.hstack((result, zcr, chroma_stft, mfcc, rms, mel, rolloff, centroid, contrast, bandwidth, tonnetz))
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
    return result

def get_features_recorded(data, sr):
    original_features = extract_features(data, sr)
    noise_features = extract_features(noise(data), sr)
    pitch_stretch_features = extract_features(pitch(stretch(data), sr), sr)
    return np.vstack([original_features, noise_features, pitch_stretch_features])

def preprocess_audio(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        features = get_features_recorded(audio, sr)
        return scaler.transform(features)
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return None

# Streamlit App Interface
st.title("Emotion Detector")
st.write("""
This application uses deep learning to predict emotions based on audio recordings. 
Users can upload a WAV file, and the app processes it to extract features like MFCCs, pitch, and spectral contrast. 
The model then analyzes these features to identify the emotion conveyed in the audio, displaying the predicted emotion label. 
With built-in data augmentation and preprocessing, the app is designed for accurate emotion recognition in varied audio inputs.
""")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", ])

if uploaded_file is not None:
    features = preprocess_audio(uploaded_file)

    if features is not None and st.button("Predict"):
        try:
            prediction = model.predict(features)
            predicted_label = encoder.inverse_transform(prediction)[0]
            st.write(f"Predicted Emotion: {predicted_label}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
