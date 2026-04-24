import os
import pickle
import numpy as np
import librosa
from tqdm import tqdm

# =========================
# CONFIG - Fixed 30s clips
# =========================
N_FRAMES = 512
IN_PKL = "data/gtzan_features.pkl"
OUT_ROOT = "data/spectrograms"

SR = 22050
DURATION = 30.0
SAMPLES = int(SR * DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# =========================
# LOAD PATHS
# =========================
with open(IN_PKL, "rb") as f:
    data = pickle.load(f)

paths = data["paths"]
print(f"Loaded {len(paths)} audio files")

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================
def get_output_path(input_path):
    filename = os.path.splitext(os.path.basename(input_path))[0] + ".npy"
    genre = os.path.basename(os.path.dirname(input_path))
    out_dir = os.path.join(OUT_ROOT, genre)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def pad_or_crop_time(mel, target_frames):
    """Force exact time dimension"""
    if mel.shape[1] == target_frames:
        return mel
    elif mel.shape[1] < target_frames:
        pad_width = target_frames - mel.shape[1]
        return np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mel[:, :target_frames]          # crop if somehow longer


def extract_log_mel(in_path, target_frames=N_FRAMES):
    try:
        y, sr = librosa.load(in_path, sr=SR, mono=True)

        # === FIXED 30-SECOND CLIP ===
        if len(y) < SAMPLES:
            y = np.pad(y, (0, SAMPLES - len(y)))
        else:
            y = y[:SAMPLES]

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Force constant dimensions
        mel_db = pad_or_crop_time(mel_db, target_frames)

        return mel_db.astype(np.float32)

    except Exception as e:
        print(f"Failed {in_path}: {e}")
        return None


# =========================
# MAIN PROCESSING
# =========================
for path in tqdm(paths, desc="Extracting 128x512 mel spectrograms"):
    mel = extract_log_mel(path)
    if mel is not None:
        out_path = get_output_path(path)
        np.save(out_path, mel)

print(f"\n✅ Done! All spectrograms saved as shape (128, {N_FRAMES})")