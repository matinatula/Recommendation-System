# src/database.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env for DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")

# SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)

# Optional: session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Simple helper to test DB


def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Database connected! Result:", result.scalar())
    except Exception as e:
        print("Connection error:", e)


