from sqlalchemy import create_engine,text
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    query = text("""
        SELECT id, track_uuid, title, artist
        FROM tracks
        ORDER BY id
        LIMIT 20
""")
    
    rows = conn.execute(query).fetchall()

    if len(rows) == 0:
        print("No tracks found in the database.")
    else:
        print("Tracks in database:")
        for r in rows:
            print(r)