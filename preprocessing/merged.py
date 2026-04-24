import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD BOTH PICKLES
# =========================
with open(f"data/gtzan_features.pkl", "rb") as f:
    data1 = pickle.load(f)

with open(f"data/bangla_features.pkl", "rb") as f:
    data2 = pickle.load(f)

# =========================
# EXTRACT
# =========================
X1, g1, p1 = data1["X"], data1["genres"], data1["paths"]
X2, g2, p2 = data2["X"], data2["genres"], data2["paths"]

print("Dataset 1:", X1.shape)
print("Dataset 2:", X2.shape)

# =========================
# ADD LANGUAGE LABELS
# =========================
lang1 = np.array(["english"] * len(X1))
lang2 = np.array(["bangla"] * len(X2))

# =========================
# MERGE
# =========================
X = np.vstack([X1, X2])
genres = np.concatenate([g1, g2])
paths = np.concatenate([p1, p2])
language = np.concatenate([lang1, lang2])

print("Merged X shape:", X.shape)
print("Language distribution:", dict(zip(*np.unique(language, return_counts=True))))

# =========================
# STANDARDIZE
# =========================
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

print("Xz mean/std:", Xz.mean().round(4), Xz.std().round(4))

# =========================
# SAVE MERGED VERSION
# =========================
merged_data = {
    "X": Xz,
    "genres": genres,
    "paths": paths,
    "language": language
}

with open(f"data/merged_features.pkl", "wb") as f:
    pickle.dump(merged_data, f)

print("Saved merged_features.pkl")