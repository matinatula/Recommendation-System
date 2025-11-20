# main.py
from src.database import test_connection
from src import content_based_recommender
from src.collaborative_als import train_and_recommend_for_user_db
from sqlalchemy import text
import os
from dotenv import load_dotenv



# 1. Test DB
test_connection()


# 2. Content-based Recommendation
track_id = "S1"
top_k = 5

cb_results = content_based_recommender.recommend_content_based(
    track_id, top_k=top_k)

print(f"\nTop {top_k} content-based recommendations for track {track_id}:\n")
for uuid, score in cb_results:
    print(f"{uuid}: {score:.4f}")

# 3. Collaborative ALS Recommendation

user_uuid = "user-1"
collab_top_k = 5
print(
    f"\nTop {collab_top_k} Collaborative ALS recommendations for user {user_uuid}:\n")
recs = train_and_recommend_for_user_db(user_uuid, top_k=collab_top_k)

if not recs:
    print("No ALS recommendations returned. Check if user has interactions.")
else:
    for uuid, score in recs:
        print(f"{uuid}: {score:.4f}")
