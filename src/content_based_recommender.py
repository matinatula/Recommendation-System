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
# 1. Load all track feature vectors from PostgreSQL
# ----------------------------------------


def load_track_features():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT track_uuid, track_features
            FROM tracks
            WHERE track_features IS NOT NULL
        """))
        # Convert features from JSON string to numpy arrays
        track_vectors = {}
        for row in result:
            vector = np.array(json.loads(row.track_features), dtype=float)
            track_vectors[row.track_uuid] = vector
        return track_vectors

# ----------------------------------------
# 2. Cosine similarity
# ----------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------------------------------
# 3. Recommend top-K similar tracks
# ----------------------------------------


def recommend_content_based(track_uuid, top_k=5):
    vectors = load_track_features()

    if track_uuid not in vectors:
        raise ValueError(f"Track {track_uuid} not found or has no features")

    target_vector = vectors[track_uuid]

    # Compute similarity with all other tracks
    similarities = []
    for other_uuid, vector in vectors.items():
        if other_uuid == track_uuid:
            continue
        sim = cosine_similarity(target_vector, vector)
        similarities.append((other_uuid, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
