import os
import numpy as np
import librosa
import pickle

# =========================
# CONFIG
# =========================
DATASET_DIR = "raw_data/gtzan"
OUT_PKL = "data/gtzan_features.pkl"

SR = 22050
DURATION = 30.0
N_MFCC = 20

# =========================
# FEATURE FUNCTION
# =========================
def extract_mfcc_features(path, sr=SR, duration=DURATION, n_mfcc=N_MFCC):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)

    target_len = int(sr * duration)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    return np.concatenate([mfcc_mean, mfcc_std]).astype(np.float32)

# =========================
# COLLECT FILES FIRST
# =========================
features = []
genres = []
paths = []

for genre in os.listdir(DATASET_DIR):
    genre_path = os.path.join(DATASET_DIR, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if not file.lower().endswith((".wav", ".mp3", ".au")):
            continue

        file_path = os.path.join(genre_path, file)

        try:
            feat = extract_mfcc_features(file_path)

            features.append(feat)
            genres.append(genre)
            paths.append(file_path)

        except Exception as e:
            print("Skipping:", file_path, "|", str(e))

# =========================
# CONVERT TO NUMPY
# =========================
X = np.vstack(features)                  
genres = np.array(genres)               
paths = np.array(paths, dtype=object)    

print("X shape:", X.shape)
print("Genres shape:", genres.shape)

# =========================
# SAVE PICKLE
# =========================
data = {
    "X": X,
    "genres": genres,
    "paths": paths
}

with open(OUT_PKL, "wb") as f:
    pickle.dump(data, f)

print("Saved:", OUT_PKL)