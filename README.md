# Content Based Recommender

src/data_generation.py

vector = [
    -210.2, 98.4, -12.3, 45.1, -8.2,   # MFCC 1–5
    125.4, 0.82,                       # tempo,beat_strength
    0.91, -7.8,                        # energy, loudness
    2700.2, 0.66, 0.48,                # spectral centroid, rolloff, bandwidth
    1, 0, 1, 0                         # emotions: happy, sad, energetic, calm
]

# Collaborative Filtering

Collaborative filtering recommends songs based on user listening behavior, not audio features.

Here’s how it works:

### 1. User-item matrix:
    - Rows = users
    - Columns = tracks
    - Values = how much user listened to a track (play count, implicit feedback)

### 2. Implicit feedback:
    - We don’t care about explicit ratings.
    - Use play counts, skips, or completions as confidence scores.

### 3. ALS model:
    - Factorizes the user-item matrix into user vectors and item vectors.
    - Then we compute recommendations by dot product similarity.

#### We’ll use the Python library implicit.