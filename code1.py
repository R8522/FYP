from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')


!pip install librosa

import librosa
import numpy as np

def preprocess_audio(file_path, target_sr=22050):
    # Load audio file and resample
    y, sr = librosa.load(file_path, sr=target_sr)
    return y, sr
