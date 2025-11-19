-- Users
CREATE TABLE IF NOT EXISTS users (
  id SERIAL PRIMARY KEY,
  user_uuid VARCHAR(128) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

-- Tracks (features included)
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

-- Interactions
CREATE TABLE IF NOT EXISTS interactions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
  play_count INTEGER DEFAULT 0,
  skipped INTEGER DEFAULT 0,
  completed INTEGER DEFAULT 0,
  last_played TIMESTAMP DEFAULT now(),
  UNIQUE(user_id, track_id)
);

-- Track emotions
CREATE TABLE IF NOT EXISTS track_emotions (
  id SERIAL PRIMARY KEY,
  track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
  emotion_label VARCHAR(64),
  emotion_confidence FLOAT
);
