import streamlit as st
import numpy as np
import librosa
import joblib
import pickle
import warnings
import tensorflow
import tempfile
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from audio_recorder_streamlit import audio_recorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set page config for better mobile experience
st.set_page_config(
    page_title="Emotion Detector 🎭",
    page_icon="🎭",
    layout="wide"
)

# Add custom CSS for better mobile responsiveness
st.markdown("""
    <style>
        .stAudio {
            width: 100%;
        }
        .uploadedFile {
            width: 100%;
        }
        @media (max-width: 768px) {
            .stButton>button {
                width: 100%;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Paths to models and encoders
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

# Emotion emojis mapping
emotion_emojis = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😮"
}

# Streamlit App Interface
st.title("Voice Emotion Detector 🎭")
st.markdown("""
    ### Welcome to the Voice Emotion Analysis Tool! 🎤
    
    This app can detect emotions in your voice through:
    - 📤 Uploading audio files
    - 🎙️ Recording your voice directly
    
    **Supported emotions:** Happy 😊 | Sad 😢 | Angry 😠 | Neutral 😐 | Fear 😨 | Disgust 🤢 | Surprise 😮
""")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Record Audio"])

with tab1:
    st.markdown("""
        ### Upload your audio file 📤
        **Instructions:**
        1. Click 'Browse files' to upload a WAV file
        2. Listen to your audio to confirm it's correct
        3. Click 'Analyze' to detect the emotion
    """)
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Analyze 🔍", key="analyze_upload"):
            with st.spinner("Analyzing your audio... 🎵"):
                features = preprocess_audio(uploaded_file)
                
                if features is not None:
                    try:
                        prediction = model.predict(features)
                        predicted_label = encoder.inverse_transform(prediction)[0]
                        # Convert the numpy array to string before using lower()
                        emotion_str = str(predicted_label).lower()
                        emoji = emotion_emojis.get(emotion_str, "🎭")
                        st.success(f"Detected Emotion: {predicted_label} {emoji}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

with tab2:
    st.markdown("""
        ### Record your voice 🎙️
        **Instructions:**
        1. Click the microphone icon to start recording
        2. Click it again to stop recording
        3. Listen to your recording to confirm it's clear
        4. Click 'Analyze' to detect the emotion
    """)
    
    # Add audio recorder with custom styling and stop button
    audio_bytes = audio_recorder(
        pause_threshold=60.0,  # Set a long threshold so recording doesn't auto-stop
        icon_size="2x",  # Make the icon bigger
        text="",  # Remove default text
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        if st.button("Analyze 🔍", key="analyze_recording"):
            with st.spinner("Analyzing your recording... 🎵"):
                # Save the recorded audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
                    fp.write(audio_bytes)
                    temp_file_path = fp.name
                
                try:
                    features = preprocess_audio(temp_file_path)
                    
                    if features is not None:
                        prediction = model.predict(features)
                        predicted_label = encoder.inverse_transform(prediction)[0]
                        # Convert the numpy array to string before using lower()
                        emotion_str = str(predicted_label).lower()
                        emoji = emotion_emojis.get(emotion_str, "🎭")
                        st.success(f"Detected Emotion: {predicted_label} {emoji}")
                    
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

# Add footer with instructions
st.markdown("""
    ---
    ### 📝 Tips for best results:
    1. Ensure you're in a quiet environment
    2. Speak clearly and naturally
    3. Keep recordings between 3-10 seconds
    4. For uploads, use WAV format audio files
    
    ### 🔧 Troubleshooting:
    - If recording doesn't work, check your browser's microphone permissions
    - Make sure your microphone is properly connected and selected
    - Try refreshing the page if you encounter any issues
""")