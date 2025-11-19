# src/data_generation.py

# vector = [
#     -210.2, 98.4, -12.3, 45.1, -8.2,   # MFCC 1–5
#     125.4, 0.82,                        # tempo, beat_strength
#     0.91, -7.8,                          # energy, loudness
#     2700.2, 0.66, 0.48,                  # spectral centroid, rolloff, bandwidth
#     1, 0, 1, 0                            # emotions: happy, sad, energetic, calm
# ]

from sqlalchemy import text
import numpy as np
import json
from .database import engine

# ✅ Track metadata (titles, artists) — matches TRACK_FEATURES
TRACKS = {
    "H1": ("Happy Song 1", "Artist Pop A"),
    "H2": ("Happy Song 2", "Artist Pop B"),
    "H3": ("Happy Song 3", "Artist Pop C"),
    "H4": ("Happy Song 4", "Artist Pop D"),
    "H5": ("Happy Song 5", "Artist Pop E"),
    "H6": ("Happy Song 6", "Artist Pop F"),
    "H7": ("Happy Song 7", "Artist Pop G"),
    "H8": ("Happy Song 8", "Artist Pop H"),
    "S1": ("Sad Song 1", "Artist Ballad A"),
    "S2": ("Sad Song 2", "Artist Ballad B"),
    "S3": ("Sad Song 3", "Artist Ballad C"),
    "S4": ("Sad Song 4", "Artist Ballad D"),
    "S5": ("Sad Song 5", "Artist Ballad E"),
    "C1": ("Calm Song 1", "Artist Acoustic A"),
    "C2": ("Calm Song 2", "Artist Acoustic B"),
    "C3": ("Calm Song 3", "Artist Acoustic C"),
    "C4": ("Calm Song 4", "Artist Acoustic D"),
    "C5": ("Calm Song 5", "Artist Acoustic E"),
    "C6": ("Calm Song 6", "Artist Acoustic F"),
    "C7": ("Calm Song 7", "Artist Acoustic G"),
    "E1": ("EDM Song 1", "Artist EDM A"),
    "E2": ("EDM Song 2", "Artist EDM B"),
    "E3": ("EDM Song 3", "Artist EDM C"),
    "E4": ("EDM Song 4", "Artist EDM D"),
    "E5": ("EDM Song 5", "Artist EDM E"),
    "E6": ("EDM Song 6", "Artist EDM F"),
    "D1": ("Dark Song 1", "Artist Dark A"),
    "D2": ("Dark Song 2", "Artist Dark B"),
    "D3": ("Dark Song 3", "Artist Dark C"),
    "D4": ("Dark Song 4", "Artist Dark D"),
}

# ✅ Hardcoded feature vectors
TRACK_FEATURES = {
    "H1": [-210.2, 98.4, -12.3, 45.1, -8.2, 125.4, 0.82, 0.91, -7.8, 2700.2, 0.66, 0.48, 1, 0, 1, 0],
    "H2": [-205.1, 103.2, -15.0, 50.7, -10.1, 130.0, 0.79, 0.87, -8.5, 2600.5, 0.61, 0.42, 1, 0, 1, 0],
    "H3": [-220.8, 110.5, -18.1, 48.3, -9.2, 118.9, 0.83, 0.89, -7.2, 2850.0, 0.70, 0.52, 1, 0, 1, 0],
    "H4": [-230.4, 92.1, -10.7, 40.5, -11.4, 135.6, 0.76, 0.85, -9.0, 2550.7, 0.60, 0.40, 1, 0, 1, 0],
    "H5": [-215.1, 105.5, -22.3, 55.1, -13.0, 128.9, 0.80, 0.90, -6.7, 2900.4, 0.72, 0.53, 1, 0, 1, 0],
    "H6": [-225.3, 88.6, -9.4, 39.8, -7.9, 140.2, 0.78, 0.88, -8.3, 2450.1, 0.55, 0.39, 1, 0, 1, 0],
    "H7": [-210.0, 100.3, -12.2, 47.0, -10.0, 122.0, 0.81, 0.92, -7.0, 2750.4, 0.68, 0.50, 1, 0, 1, 0],
    "H8": [-217.3, 102.1, -17.4, 52.1, -12.5, 119.4, 0.84, 0.91, -6.9, 2950.9, 0.73, 0.55, 1, 0, 1, 0],
    # Sad
    "S1": [-310.4, 80.1, -40.2, 25.1, -20.5, 75.2, 0.42, 0.36, -14.2, 1800.3, 0.55, 0.28, 0, 1, 0, 0],
    "S2": [-295.0, 78.0, -38.1, 28.5, -18.3, 72.0, 0.40, 0.34, -15.0, 1700.1, 0.52, 0.25, 0, 1, 0, 0],
    "S3": [-305.9, 82.7, -42.6, 30.3, -22.1, 70.4, 0.45, 0.38, -13.8, 1850.9, 0.56, 0.29, 0, 1, 0, 0],
    "S4": [-290.2, 76.4, -36.5, 27.5, -19.8, 78.0, 0.39, 0.35, -14.8, 1600.3, 0.50, 0.23, 0, 1, 0, 0],
    "S5": [-300.7, 84.1, -44.0, 32.7, -21.4, 68.9, 0.43, 0.37, -13.5, 1750.7, 0.53, 0.27, 0, 1, 0, 0],
    # Calm
    "C1": [-260.1, 95.0, -20.4, 35.0, -15.0, 85.0, 0.50, 0.25, -12.0, 2100.5, 0.48, 0.30, 0, 0, 0, 1],
    "C2": [-270.3, 90.2, -18.0, 32.1, -14.1, 90.0, 0.48, 0.22, -13.0, 2000.2, 0.45, 0.28, 0, 0, 0, 1],
    "C3": [-255.9, 92.5, -21.5, 36.4, -13.7, 88.2, 0.53, 0.26, -11.8, 2150.3, 0.49, 0.31, 0, 0, 0, 1],
    "C4": [-265.0, 88.7, -19.0, 33.7, -16.2, 82.0, 0.47, 0.20, -14.0, 2050.0, 0.44, 0.27, 0, 0, 0, 1],
    "C5": [-258.4, 94.1, -22.1, 34.9, -17.0, 87.6, 0.52, 0.27, -12.5, 2200.7, 0.50, 0.32, 0, 0, 0, 1],
    "C6": [-275.6, 89.3, -19.8, 31.0, -15.5, 80.4, 0.46, 0.24, -15.0, 1950.4, 0.43, 0.26, 0, 0, 0, 1],
    "C7": [-262.7, 93.9, -23.7, 37.1, -14.8, 92.1, 0.51, 0.28, -11.5, 2250.2, 0.51, 0.33, 0, 0, 0, 1],
    # EDM
    "E1": [-180.2, 120.4, -5.1, 60.3, -3.8, 138.0, 0.95, 0.96, -5.0, 3000.0, 0.80, 0.60, 1, 0, 1, 0],
    "E2": [-175.0, 130.2, -6.7, 62.8, -4.5, 142.5, 0.97, 0.98, -4.5, 3100.4, 0.82, 0.62, 1, 0, 1, 0],
    "E3": [-185.1, 118.0, -4.0, 58.9, -3.0, 145.6, 0.93, 0.95, -6.0, 2950.7, 0.78, 0.58, 1, 0, 1, 0],
    "E4": [-190.5, 125.9, -7.2, 63.5, -5.0, 150.2, 0.96, 0.97, -4.8, 3200.3, 0.84, 0.65, 1, 0, 1, 0],
    "E5": [-178.3, 135.1, -3.5, 64.9, -2.8, 155.1, 0.98, 0.99, -3.9, 3300.8, 0.85, 0.67, 1, 0, 1, 0],
    "E6": [-192.1, 140.2, -8.1, 66.3, -6.0, 148.4, 0.94, 0.96, -5.2, 2850.2, 0.75, 0.55, 1, 0, 1, 0],
    # Dark
    "D1": [-320.5, 70.3, -50.1, 20.0, -25.0, 60.0, 0.30, 0.20, -18.0, 1500.0, 0.40, 0.22, 0, 1, 0, 0],
    "D2": [-330.1, 68.0, -52.2, 19.3, -26.1, 58.6, 0.28, 0.18, -19.0, 1400.3, 0.38, 0.20, 0, 1, 0, 0],
    "D3": [-315.4, 72.7, -48.8, 22.5, -23.9, 63.4, 0.32, 0.21, -17.5, 1550.1, 0.41, 0.23, 0, 1, 0, 0],
    "D4": [-325.7, 69.9, -51.3, 21.0, -24.5, 61.1, 0.29, 0.19, -18.5, 1450.9, 0.39, 0.21, 0, 1, 0, 0],
}


def create_tables():
    with engine.begin() as conn:
        # Tracks
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tracks (
                id SERIAL PRIMARY KEY,
                track_uuid VARCHAR(128) UNIQUE NOT NULL,
                title VARCHAR(512),
                artist VARCHAR(256),
                duration_ms INTEGER,
                track_features TEXT,
                tempo FLOAT,
                energy FLOAT,
                valence FLOAT
            );
        """))
        # Users
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                user_uuid VARCHAR(128) UNIQUE NOT NULL
            );
        """))
        # Interactions
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                track_id INTEGER REFERENCES tracks(id),
                play_count INTEGER,
                skipped INTEGER,
                completed INTEGER,
                UNIQUE(user_id, track_id)
            );
        """))
    print("✅ Tables created successfully!")


def populate_tracks():
    with engine.begin() as conn:
        for track_uuid, (title, artist) in TRACKS.items():
            features = TRACK_FEATURES[track_uuid]
            tempo, energy, valence = features[0], features[1], features[2]
            conn.execute(
                text("""
                    INSERT INTO tracks (track_uuid, title, artist, duration_ms, track_features, tempo, energy, valence)
                    VALUES (:uuid, :title, :artist, :duration, :vector, :tempo, :energy, :valence)
                    ON CONFLICT (track_uuid)
                    DO UPDATE SET
                        track_features = EXCLUDED.track_features,
                        tempo = EXCLUDED.tempo,
                        energy = EXCLUDED.energy,
                        valence = EXCLUDED.valence
                """),
                {
                    "uuid": track_uuid,
                    "title": title,
                    "artist": artist,
                    "duration": 180000,
                    "vector": json.dumps(features),
                    "tempo": float(tempo),
                    "energy": float(energy),
                    "valence": float(valence)
                }
            )
    print(f"✅ {len(TRACKS)} tracks inserted!")


def populate_users(num_users=50):
    with engine.begin() as conn:
        for i in range(1, num_users + 1):
            user_uuid = f"user-{i}"
            conn.execute(
                text("""
                    INSERT INTO users (user_uuid)
                    VALUES (:uuid)
                    ON CONFLICT (user_uuid) DO NOTHING
                """),
                {"uuid": user_uuid}
            )
    print(f"✅ {num_users} users inserted!")


def populate_interactions(num_interactions=2000):
    with engine.begin() as conn:
        user_ids = [int(r[0]) for r in conn.execute(
            text("SELECT id FROM users")).all()]
        track_ids = [int(r[0]) for r in conn.execute(
            text("SELECT id FROM tracks")).all()]
        for _ in range(num_interactions):
            user = int(np.random.choice(user_ids))
            track = int(np.random.choice(track_ids))
            play_count = int(np.random.randint(1, 15))
            skipped = int(np.random.randint(0, 2))
            completed = int(np.random.randint(0, 2))
            conn.execute(
                text("""
                    INSERT INTO interactions (user_id, track_id, play_count, skipped, completed)
                    VALUES (:user, :track, :plays, :skipped, :completed)
                    ON CONFLICT (user_id, track_id) DO NOTHING
                """),
                {
                    "user": user,
                    "track": track,
                    "plays": play_count,
                    "skipped": skipped,
                    "completed": completed
                }
            )
    print(f"✅ {num_interactions} interactions inserted!")


def generate_all_data():
    create_tables()
    populate_tracks()
    populate_users()
    populate_interactions()
    print("🎉 Database setup complete!")
