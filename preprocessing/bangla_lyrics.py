import pickle
import numpy as np
import re
import random
from collections import defaultdict

# ========================= CONFIG =========================
FEATURES_PKL = "data/bangla_features.pkl"
BN_LYRICS_PKL = "data/bn_genre_lyrics.pkl"
OUT_PKL = "data/bangla_features_with_lyrics.pkl"

MIN_WORDS = 15
MAX_WORDS = 120

FINAL_FALLBACK = "এটি একটি যন্ত্রসঙ্গীতধর্মী বাংলা সুর, যেখানে কোনো কণ্ঠ নেই, শুধু বাদ্যযন্ত্রের মেলোডি, আবেগ, তাল, এবং পরিবেশনা রয়েছে"

# ========================= LOAD FEATURES =========================
print("Loading bangla_features.pkl...")
with open(FEATURES_PKL, "rb") as f:
    data = pickle.load(f)

paths = data.get("paths", [])
genres = data.get("genres", [])

if isinstance(paths, np.ndarray):
    paths = paths.tolist()
if isinstance(genres, np.ndarray):
    genres = genres.tolist()

print(f"Loaded {len(genres)} Bangla songs.")

# ========================= LOAD GENRE LYRICS =========================
print("\nLoading bn_genre_lyrics.pkl...")
with open(BN_LYRICS_PKL, "rb") as f:
    genre_lyrics_raw = pickle.load(f)

genre_lyrics = {}
print("\nProcessed genre keys:")

for raw_key, lyrics_list in genre_lyrics_raw.items():
    parts = raw_key.strip().lower().split()
    clean_key = parts[1] if len(parts) >= 2 else parts[0]

    if clean_key in ["hip-hop", "hiphop"]:
        clean_key = "hip-hop"

    genre_lyrics[clean_key] = lyrics_list
    print(f"  '{raw_key}' → '{clean_key}' : {len(lyrics_list)} lyrics")

if "metal" not in genre_lyrics or len(genre_lyrics.get("metal", [])) == 0:
    if "rock" in genre_lyrics:
        print("\n⚠️ No Metal lyrics. Using Rock as fallback.")
        genre_lyrics["metal"] = genre_lyrics["rock"].copy()

# ========================= CLEANING =========================
def is_pure_bangla(text):
    if not isinstance(text, str) or not text.strip():
        return False
    if not re.search(r"[\u0980-\u09FF]", text):
        return False
    cleaned = re.sub(r"[a-zA-Z0-9]", "", text)
    return len(cleaned) >= len(text) * 0.7


def clean_bangla_lyric(text):
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not is_pure_bangla(text):
        return None

    text = text.lower()
    text = re.sub(r"[^ঀ-৿\s'.,!?()-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    if len(words) < MIN_WORDS:
        return None

    return text


def dedup_lines(text):
    parts = re.split(r"[।.!?]+", text)
    seen = set()
    unique = []

    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if part and part not in seen:
            unique.append(part)
            seen.add(part)

    return " ".join(unique).strip()


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

# ========================= QUALITY SCORING =========================
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

# ========================= FILTER + CLEAN =========================
print("\nFiltering & cleaning lyrics...")

for g in list(genre_lyrics.keys()):
    cleaned = []
    original_count = len(genre_lyrics[g])

    for lyric in genre_lyrics[g]:
        cl = clean_bangla_lyric(lyric)
        if not cl:
            continue

        cl = dedup_lines(cl)
        cl = final_lyrics_window(cl)

        if cl:
            cleaned.append(cl)

    print(f"  '{g}' : kept {len(cleaned)}/{original_count}")
    genre_lyrics[g] = cleaned

# ========================= RANK LYRICS =========================
print("\nRanking lyrics by quality...")

ranked_lyrics = {}

for g, lst in genre_lyrics.items():
    scored = [(lyric, lyric_quality_score(lyric)) for lyric in lst]
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked_lyrics[g] = [x[0] for x in scored]

# ========================= SPLIT INTO TIERS =========================
good_lyrics = {}
ok_lyrics = {}
bad_lyrics = {}

for g, lst in ranked_lyrics.items():
    n = len(lst)

    good_lyrics[g] = lst[:int(0.5 * n)]
    ok_lyrics[g] = lst[int(0.5 * n):int(0.8 * n)]
    bad_lyrics[g] = lst[int(0.8 * n):]

    random.shuffle(good_lyrics[g])
    random.shuffle(ok_lyrics[g])
    random.shuffle(bad_lyrics[g])

# ========================= ASSIGN LYRICS =========================
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

    lyric = None

    while good_lyrics.get(g):
        candidate = good_lyrics[g].pop(0)
        candidate = final_lyrics_window(candidate)
        if candidate:
            lyric = candidate
            break

    if lyric is None:
        while ok_lyrics.get(g):
            candidate = ok_lyrics[g].pop(0)
            candidate = final_lyrics_window(candidate)
            if candidate:
                lyric = candidate
                break

    if lyric is None:
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

np.save("data/bangla_lyrics.npy", np.array(assigned_lyrics, dtype=object))

# ========================= SUMMARY =========================
print(f"\n✅ SUCCESS! Saved to: {OUT_PKL}")
print(f"Total songs: {len(assigned_lyrics)}")

print("\nUsage summary:")
for g, count in sorted(used_count.items()):
    print(f"  {g:15} : {count}")