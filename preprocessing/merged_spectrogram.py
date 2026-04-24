import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

# =========================
# CONFIG
# =========================
PICKLE1 = "data/gtzan_features.pkl"
PICKLE2 = "data/bangla_features.pkl"

SPECTROGRAM_ROOT = "data/spectrograms"
OUT_NPY = "data/Xz_merged.npy"

# =========================
# LOAD PATHS FROM PICKLES
# =========================
def load_paths(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    paths = data.get("paths", [])
    if isinstance(paths, np.ndarray):
        paths = paths.tolist()
    return paths

print("Loading paths from pickles...")
paths1 = load_paths(PICKLE1)
paths2 = load_paths(PICKLE2)

all_audio_paths = paths1 + paths2

print(f"Total files from pickles: {len(all_audio_paths)}")

# =========================
# CONVERT AUDIO PATH → SPECTROGRAM .npy PATH
# =========================
def get_spectrogram_path(audio_path):
    parts = audio_path.split(os.sep)
    if 'data_merged' in parts:
        idx = parts.index('data_merged')
        relative = parts[idx + 1:]                    # genre + filename
    else:
        relative = parts[-2:]                         # fallback

    # Change extension from .wav to .npy
    filename = os.path.splitext(relative[-1])[0] + ".npy"
    relative[-1] = filename

    return os.path.join(SPECTROGRAM_ROOT, *relative)


# =========================
# LOAD EXISTING SPECTROGRAMS
# =========================
print("Loading existing spectrogram .npy files...")

mel_list = []
missing_count = 0

for audio_path in tqdm(all_audio_paths, desc="Loading spectrograms"):
    npy_path = get_spectrogram_path(audio_path)
    
    if os.path.exists(npy_path):
        try:
            mel = np.load(npy_path)                    # shape (128, 512)
            mel_list.append(mel)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
    else:
        missing_count += 1
        if missing_count <= 5:   # show only first few
            print(f"Missing: {npy_path}")

print(f"\nLoaded {len(mel_list)} spectrograms successfully.")
if missing_count > 0:
    print(f"Warning: {missing_count} files were missing.")

# =========================
# CREATE FINAL ARRAY
# =========================
Xz = np.array(mel_list)                    # (N, 128, 512)
Xz = np.expand_dims(Xz, axis=-1)           # (N, 128, 512, 1)

print(f"Final merged shape: {Xz.shape}")

# =========================
# NORMALIZATION
# =========================
print("Normalizing...")
X_min = Xz.min()
X_max = Xz.max()
Xz_norm = (Xz - X_min) / (X_max - X_min + 1e-8)

print(f"Normalized range: [{Xz_norm.min():.4f}, {Xz_norm.max():.4f}]")

# =========================
# SAVE
# =========================
np.save(OUT_NPY, Xz_norm)
print(f"\n✅ Done! Merged dataset saved as '{OUT_NPY}'")
print(f"Shape ready for ConvVAE: {Xz_norm.shape}")