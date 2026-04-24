All data files except `Xz_merged.npy` have already been preprocessed and stored in the **"data/"** folder. Please download Xz_merged from [here](https://drive.google.com/file/d/17EtkZ3oHXpWW9D2GmNKRZ5acKC9iHfXb/view?usp=drivesdk).

---

If you wish to preprocess from scratch, please refer to the steps below.

##### Environment Setup

1. `python -m venv env`
2. `pip install -r requirements.txt`

##### Dataset Download

1. Download [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classificationhttps:/) for English music. Extract **Data -> genres_original** to **raw_data/**. Rename to **gtzan**
2. Download [BanglaBeats](https://www.kaggle.com/datasets/thisisjibon/banglabeats3sechttps:/) Dataset for Bangla music. Extract **wavs3sec** to **raw_data/**. Rename to **bangla_beats**

##### Data Preparation

Run these files in the given order:

1. `merge_bangla_beats.py`
2. `gtzan.py`
3. `bangla_beats.py`
4. `merged.py`
5. `gtzan_spectrogram.py`
6. `bangla_spectrogram.py`
7. `merged_spectrogram.py`
8. `lyric_embedding.py`
