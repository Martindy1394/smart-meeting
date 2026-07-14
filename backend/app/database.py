"""SQLAlchemy engine / session setup.

Uses SQLite by default (zero-config local dev) but any SQLAlchemy URL works —
set ``DATABASE_URL`` to a PostgreSQL DSN for production.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Required for SQLite when used across threads (FastAPI runs handlers in a
    # threadpool, and background finalization runs on worker threads).
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called on application startup."""
    # Import models so they register with the metadata before create_all.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
