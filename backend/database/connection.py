"""
PostgreSQL connection configuration with SQLAlchemy.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection URL
# If DATABASE_URL is not set, use SQLite as fallback for local development
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    # Fallback to SQLite for local development
    DATABASE_URL = "sqlite:///./stroke_prediction.db"
    print("⚠️  DATABASE_URL not set. Using SQLite for local development.")

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Change to False in production
    pool_pre_ping=True,  # Verifies connections before using them
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session creator
SessionLocal = sessionmaker(
    autocommit=False,  # autocommit=False: Transactions must be done manually
    autoflush=False,  # autoflush=False: Does not send changes automatically to the DB
    bind=engine
)
# Base class for models
Base = declarative_base()

# Database session generator
def get_db():
    """
    Database session generator.
    Used with Depends() in FastAPI to inject the session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Creates all tables in the database.
    Only for development. Use Alembic in production.
    """
    Base.metadata.create_all(bind=engine)