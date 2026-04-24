import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# =========================
# CONFIG
# =========================
input_folder = "raw_data/bangla_beats"
output_folder = "raw_data/bangla_beats_merged"

sr = 22050
chunk_size = 10

os.makedirs(output_folder, exist_ok=True)

# =========================
# PROCESS
# =========================
for genre in os.listdir(input_folder):
    genre_path = os.path.join(input_folder, genre)

    if not os.path.isdir(genre_path):
        continue

    files = sorted(os.listdir(genre_path))
    genre_out = os.path.join(output_folder, genre)
    os.makedirs(genre_out, exist_ok=True)

    chunk_id = 0

    for i in tqdm(range(0, len(files), chunk_size)):
        chunk_files = files[i:i + chunk_size]

        if len(chunk_files) < chunk_size:
            continue

        audio_list = []

        for f in chunk_files:
            file_path = os.path.join(genre_path, f)

            try:
                y, _ = librosa.load(file_path, sr=sr, mono=True)
            except:
                y = np.zeros(sr * 3)

            audio_list.append(y)

        merged_audio = np.concatenate(audio_list)

        # save
        out_path = os.path.join(genre_out, f"chunk_{chunk_id:05d}.wav")
        sf.write(out_path, merged_audio, sr)

        chunk_id += 1

print("Done merging audio dataset.")