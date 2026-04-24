import pickle
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# ========================= CONFIG =========================
FEATURES_PKL = "data/gtzan_features.pkl"
CSV_PATH = "raw_data/songs.csv"
OUT_PKL = "data/english_features_with_lyrics.pkl"

MIN_WORDS = 15
MAX_WORDS = 180

CLASSICAL_PLACEHOLDER = (
    "instrumental orchestral classical music with no vocals, no lyrics, and no sung chorus, "
    "only strings, winds, piano, brass, and expressive orchestral movement"
)

FINAL_FALLBACK = (
    "instrumental music with no vocals, no lyrics, and no sung chorus, only instruments, "
    "rhythm, melody, and expressive arrangement"
)

# ========================= GTZAN GENRE MAP =========================
GTZAN_GENRES = {
    "blues": ["blues"],
    "classical": ["classical"],
    "country": ["country"],
    "disco": ["disco", "dance"],
    "hiphop": ["hip hop", "hip-hop", "rap"],
    "jazz": ["jazz"],
    "metal": ["metal"],
    "pop": ["pop"],
    "reggae": ["reggae"],
    "rock": ["rock"]
}

def map_to_gtzan(main, niche):
    text = f"{main} {niche}".lower()
    for gtzan, keywords in GTZAN_GENRES.items():
        for k in keywords:
            if k in text:
                return gtzan
    return None

# ========================= CLEANING =========================
def clean_lyric(text):
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.lower()

    # normalize apostrophes
    text = text.replace("’", "'").replace("‘", "'")

    # remove [chorus], [verse], etc.
    text = re.sub(r"\[.*?\]", " ", text)

    # keep apostrophes
    text = re.sub(r"[^a-z\s']", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) < 40:
        return None

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:
        return None

    return text


def reduce_repetition(text):
    words = text.split()

    if len(words) > 300:
        words = words[:300]

    return " ".join(words)


def final_lyrics_window(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    if not isinstance(text, str):
        return None

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    words = text.split()

    if len(words) < min_words:
        return None

    if len(words) > max_words:
        start = max(0, (len(words) - max_words) // 2)
        words = words[start:start + max_words]

    return " ".join(words).strip()

# ========================= QUALITY =========================
def lyric_quality_score(text):
    words = text.split()
    length = len(words)

    if length == 0:
        return 0

    unique_ratio = len(set(words)) / length

    score = (
        min(length / 200, 1.0) * 0.5 +
        unique_ratio * 0.5
    )
    return score


def process_bad_lyric(text):
    words = text.split()

    if len(words) > 120:
        start = len(words) // 3
        text = " ".join(words[start:start + 80])

    return text

# ========================= LOAD FEATURES =========================
print("Loading GTZAN features...")
with open(FEATURES_PKL, "rb") as f:
    data = pickle.load(f)

paths = data.get("paths", [])
genres = data.get("genres", [])

if isinstance(paths, np.ndarray):
    paths = paths.tolist()
if isinstance(genres, np.ndarray):
    genres = genres.tolist()

print(f"Loaded {len(genres)} songs.")

# ========================= LOAD CSV =========================
print("\nLoading songs.csv...")

df = pd.read_csv(CSV_PATH, low_memory=False)

genre_lyrics = defaultdict(list)

for _, row in df.iterrows():
    lyric_raw = row.get("lyrics")

    if not isinstance(lyric_raw, str):
        continue

    cleaned = clean_lyric(lyric_raw)
    if not cleaned:
        continue

    cleaned = reduce_repetition(cleaned)
    cleaned = final_lyrics_window(cleaned)

    if not cleaned:
        continue

    main_genre = str(row.get("genre", "")).strip()
    niche_raw = str(row.get("niche_genres", "")).strip().lower()

    mapped = map_to_gtzan(main_genre, niche_raw)

    if mapped:
        genre_lyrics[mapped].append(cleaned)

print("\nGenre distribution after mapping:")
for g, lst in genre_lyrics.items():
    print(f"  {g:10} : {len(lst)}")

# ========================= REMOVE WEAK GENRES =========================
for g in list(genre_lyrics.keys()):
    if len(genre_lyrics[g]) < 50:
        del genre_lyrics[g]

# ========================= RANK =========================
print("\nRanking lyrics...")

ranked_lyrics = {}

for g, lst in genre_lyrics.items():
    scored = [(lyric, lyric_quality_score(lyric)) for lyric in lst]
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked_lyrics[g] = [x[0] for x in scored]

# ========================= SPLIT =========================
good_lyrics = {}
ok_lyrics = {}
bad_lyrics = {}

for g, lst in ranked_lyrics.items():
    n = len(lst)

    good_lyrics[g] = lst[:int(0.5 * n)]
    ok_lyrics[g] = lst[int(0.5 * n):int(0.8 * n)]
    bad_lyrics[g] = lst[int(0.8 * n):]

    np.random.shuffle(good_lyrics[g])
    np.random.shuffle(ok_lyrics[g])
    np.random.shuffle(bad_lyrics[g])

# ========================= ASSIGN =========================
print("\nAssigning lyrics (smart)...")

assigned_lyrics = []
used_count = defaultdict(int)

all_lyrics_global = []
for sublist in ranked_lyrics.values():
    for item in sublist:
        item = final_lyrics_window(item)
        if item:
            all_lyrics_global.append(item)

fallback_idx = 0

for genre in genres:
    g = str(genre).strip().lower()

    # special case: classical gets a stable placeholder
    if g == "classical":
        lyric = CLASSICAL_PLACEHOLDER
        lyric = final_lyrics_window(lyric)
        if not lyric:
            lyric = FINAL_FALLBACK

        assigned_lyrics.append(lyric)
        used_count[g] += 1
        continue

    if g not in ranked_lyrics:
        g = None

    lyric = None

    while g and good_lyrics.get(g):
        candidate = good_lyrics[g].pop(0)
        candidate = final_lyrics_window(candidate)
        if candidate:
            lyric = candidate
            break

    if lyric is None and g:
        while ok_lyrics.get(g):
            candidate = ok_lyrics[g].pop(0)
            candidate = final_lyrics_window(candidate)
            if candidate:
                lyric = candidate
                break

    if lyric is None and g:
        while bad_lyrics.get(g):
            candidate = bad_lyrics[g].pop(0)
            candidate = process_bad_lyric(candidate)
            candidate = final_lyrics_window(candidate)
            if candidate:
                lyric = candidate
                break

    if lyric is None:
        candidate = None
        if all_lyrics_global:
            for _ in range(len(all_lyrics_global)):
                raw = all_lyrics_global[fallback_idx % len(all_lyrics_global)]
                fallback_idx += 1
                raw = final_lyrics_window(raw)
                if raw:
                    candidate = raw
                    break

        lyric = candidate if candidate else FINAL_FALLBACK

    lyric = final_lyrics_window(lyric)
    if not lyric:
        lyric = FINAL_FALLBACK

    assigned_lyrics.append(lyric)
    used_count[g] += 1

# ========================= SAVE =========================
final_data = {
    "genres": np.array(genres),
    "paths": np.array(paths, dtype=object),
    "lyrics": np.array(assigned_lyrics, dtype=object)
}

with open(OUT_PKL, "wb") as f:
    pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

np.save("data/english_lyrics.npy", np.array(assigned_lyrics, dtype=object))

# ========================= SUMMARY =========================
print(f"\n✅ SUCCESS! Saved to: {OUT_PKL}")
print(f"Total songs: {len(assigned_lyrics)}")

print("\nUsage summary:")
for g, count in sorted(used_count.items()):
    print(f"  {str(g):15} : {count}")