"""
PostgreSQL connection configuration with SQLAlchemy.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.config import settings

# PostgreSQL connection URL from settings
DATABASE_URL = settings.DATABASE_URL

# SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=settings.DEBUG,  # Only log SQL queries in debug mode
    pool_pre_ping=True,  # Verifies connections before using them
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