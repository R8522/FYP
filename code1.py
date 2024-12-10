import streamlit as st
import os
from sklearn.externals import joblib
import librosa
from model import predict_chicken_health

# Load the trained model and encoder (adjust paths as needed)
model = joblib.load('chicken_health_model.pkl')  # Your trained model
encoder = joblib.load('label_encoder.pkl')  # Your encoder if needed

def main():
    st.title("Chicken Health Prediction from Audio")

    st.markdown("Upload an audio file of chicken sound to predict if it's healthy or sick.")
    
    # Upload an audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Save the uploaded file temporarily
        temp_file_path = "/tmp/" + uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features and predict health
        prediction = predict_chicken_health(temp_file_path)
        
        # Display prediction result
        st.write(f"The chicken is {prediction}.")

if __name__ == "__main__":
    main()


import librosa
import joblib
import numpy as np

# You should load the classifier and encoder here (if you're not passing them directly)
clf = joblib.load('chicken_health_model.pkl')
encoder = joblib.load('label_encoder.pkl')

def extract_features(file_path):
    # Extract audio features as done previously
    y, sr = librosa.load(file_path)
    # Feature extraction code, e.g., MFCCs, Chroma, etc.
    features = librosa.feature.mfcc(y=y, sr=sr)
    return features

def predict_chicken_health(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Reshape for prediction
    prediction = clf.predict(features)
    return encoder.inverse_transform(prediction)[0]

