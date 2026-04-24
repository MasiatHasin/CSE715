import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ========================= CONFIG =========================
ENGLISH_PKL = f"{BASE_DIR}/english_features_with_lyrics.pkl"
BANGLA_PKL   = f"{BASE_DIR}/bangla_features_with_lyrics.pkl"

OUT_EMB = f"{BASE_DIR}/lyrics_embeddings.npy"
# ========================= LOAD =========================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading pickle1 (English)...")
data1 = load_pickle(ENGLISH_PKL)

print("Loading pickle2 (Bangla)...")
data2 = load_pickle(BANGLA_PKL)

# ========================= MERGE (ORDER SAFE) =========================
lyrics = np.concatenate([data1["lyrics"], data2["lyrics"]])

print(f"Total samples: {len(lyrics)}")

# ========================= EMBEDDING MODEL =========================
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ========================= ENCODE LYRICS =========================
embeddings = model.encode(
    lyrics.tolist(),
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(embeddings)

# ========================= SAVE OUTPUTS =========================
np.save(OUT_EMB, embeddings)

print("\n✅ DONE")
print("Embeddings shape:", embeddings.shape)
print("Saved to:")
print(" -", OUT_EMB)