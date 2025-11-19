# main.py
from src import content_based_recommender
from src.database import test_connection

# 1. Test DB
test_connection()


# 3️⃣ Test recommendation for a sample track
track_id = "S1"
top_k = 5
print(f"\nTop {top_k} recommendations for {track_id}:\n")
results = content_based_recommender.recommend_content_based(track_id, top_k=top_k)
for uuid, score in results:
    print(f"{uuid}: {score:.4f}")
