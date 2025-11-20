# src/collaborative_als.py
import os
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text



load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# ---------------------------
# 1) Load interactions
# ---------------------------


def load_interactions():

    # Returns a DataFrame with columns: user_id(int), track_id(int), play_count(int)
    # Using DB integer IDs(not UUIDs) because sparse matrices need compact integer indices.

    query = text("""
        SELECT users.id as user_id, tracks.id as track_id, interactions.play_count
        FROM interactions
        JOIN users ON interactions.user_id = users.id
        JOIN tracks ON interactions.track_id = tracks.id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


# ---------------------------
# 2) Build user-item sparse matrix
# ---------------------------

def build_user_item_matrix(df, use_confidence=True, alpha=40.0):
    # Map DB user_ids → 0-based dense indices
    user_ids_in_matrix = sorted(df['user_id'].unique())
    user_idx_map = {uid: idx for idx, uid in enumerate(user_ids_in_matrix)}

    rows = df['user_id'].map(user_idx_map).to_numpy()  # dense 0-based row
    cols = df['track_id'].astype(int) - 1  # assuming track IDs start at 1
    plays = df['play_count'].astype(float).to_numpy()

    data = 1.0 + alpha * plays if use_confidence else plays

    num_users = len(user_ids_in_matrix)
    num_items = cols.max() + 1

    user_item = coo_matrix((data, (rows, cols)),
                           shape=(num_users, num_items)).tocsr()
    return user_item, user_idx_map  # return mapping too


# ---------------------------
# 3) Train ALS model
# ---------------------------

def train_als(user_item, factors=64, regularization=0.01, iterations=15):

    # Trains implicit ALS.
    # implicit expects item-user matrix for training -> transpose user_item.
    # Returns the trained model.

    # Convert to item-user (rows = items, cols = users) and as double
    item_user = user_item.T.tocsr().astype(np.float64)

    # Create the model (factors/regularization/iterations can be tweaked)
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=iterations, dtype=np.float64)

    # Fit model to item-user data
    model.fit(item_user)
    return model


# ---------------------------
# 4) Recommend for *user_id* (integer)
# ---------------------------
def recommend_for_user(model, user_item, user_idx, top_k=10, filter_played=True):

    # Returns list of (track_id, score) recommended for user_idx.
    # - model: trained ALS model
    # - user_item: csr_matrix (users x items)
    # - user_idx: row index in user_item matrix (0-based)
    # - top_k: number of recommendations
    # - filter_played: if True, removes items the user already played

    # Slice the user's row only (avoid full matrix issues)
    user_row = user_item[user_idx]  # shape (1 x num_items)

    recommendations = model.recommend(
        userid=0,  # because user_row is a single row
        user_items=user_row,  # CSR of shape (1 x num_items)
        N=top_k,
        filter_already_liked_items=filter_played
    )

    return recommendations

# ---------------------------
# 5) Full pipeline: UUID → recommendations (returns clean UUIDs): train on current DB and recommend for a user_uuid
# ---------------------------


def train_and_recommend_for_user_db(user_uuid, top_k=10, alpha=40.0, factors=64, iterations=15):
    # End-to-end helper:
    # - Map user_uuid -> user_id
    # - Load interactions, build matrix, train ALS, recommend top_k
    # - Returns recommended track_ids with scores

    # 1) map user_uuid -> user_id
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id FROM users WHERE user_uuid = :u"), {
                           "u": user_uuid}).fetchone()
        if row is None:
            raise ValueError(f"User UUID {user_uuid} not found in DB")
        user_id = int(row[0])

    # 2) load interactions
    df = load_interactions()
    if df.empty:
        print("No interactions found in DB !")
        return []

    # 3) build user-item matrix with confidence scaling
    user_item, user_idx_map = build_user_item_matrix(
        df, use_confidence=True, alpha=alpha)

    # 4) train ALS
    model = train_als(user_item, factors=factors,
                      regularization=0.01, iterations=iterations)

    # 5) Recommend (returns int track_ids)
    # map DB user_id → ALS matrix row
    if user_id not in user_idx_map:
        print(
            f"\n❌ User {user_uuid} (id={user_id}) has NO interactions in DB.")
        return []

    user_idx = user_idx_map[user_id]  # correct dense row index

    # recommend safely
    raw_recs = recommend_for_user(
        model, user_item, user_idx, top_k=top_k, filter_played=True
    )

    if not raw_recs:
        return []

    track_ids = [int(t[0]) for t in raw_recs]

    # 6) Convert int track_id -> track_uuid
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, track_uuid FROM tracks WHERE id = ANY(:ids)"),
            {"ids": track_ids}
        ).fetchall()

    id_to_uuid = {int(r[0]): r[1] for r in rows}

    # 7) Build final clean result
    item_ids, scores = raw_recs
    final = [(id_to_uuid.get(int(tid), f"id:{tid}"), float(score))
             for tid, score in zip(item_ids, scores)]

    return final
