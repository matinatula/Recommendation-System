# main.py
from src import content_based_recommender, data_generation,database
from src.database import test_connection

# 1. Test DB
test_connection()

# 2. Populate database
data_generation.generate_all_data()

# 3. Test recommendation
print("\nTop 5 recommendations for H1:\n")
results = content_based_recommender.recommend("H1", top_k=5)
for uuid, score in results:
    print(f"{uuid}: {score:.4f}")
