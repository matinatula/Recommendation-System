from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://postgres:yourpassword@localhost:5432/moodio")
with engine.begin() as conn:
    result = conn.execute("SELECT current_database(), version();")
    print(result.fetchall())
