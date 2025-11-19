# 1. load track features
# 2. convert them to vectors
# 3. compute cosine similarity
# 4. recommend top-K tracks

# src/content_based_recommender.py
import json
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# ----------------------------------------
# 1. Load all track feature vectors
# ----------------------------------------
def load_track_features():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT track_uuid, track_features
            FROM tracks
            WHERE track_features IS NOT NULL
        """))
        tracks = []
        for row in result:
            features = json.loads(row.track_features)
            vector = np.array(features, dtype=float)
            tracks.append((row.track_uuid, vector))
        return tracks

# ----------------------------------------
# 2. Cosine similarity
# ----------------------------------------
def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------------------
# 3. Recommend top-K similar tracks
# ----------------------------------------
def recommend(track_uuid, top_k=5):
    tracks = load_track_features()
    target_vector = None
    for uuid, vector in tracks:
        if uuid == track_uuid:
            target_vector = vector
            break
    if target_vector is None:
        raise ValueError(f"Track {track_uuid} not found or has no features")

    similarities = []
    for uuid, vector in tracks:
        if uuid == track_uuid:
            continue
        sim = cosine_sim(target_vector, vector)
        similarities.append((uuid, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


