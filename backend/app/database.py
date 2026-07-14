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
    _apply_lightweight_migrations()


def _apply_lightweight_migrations() -> None:
    """Add newly introduced columns to existing tables (dev-friendly, no Alembic).

    ``create_all`` never alters existing tables, so when new columns are added to
    a model we add them here idempotently. This keeps existing SQLite databases
    working without a manual reset. For production, use real migrations.
    """
    from sqlalchemy import inspect, text

    # column name -> SQL definition (with default) for the meetings table.
    meeting_columns = {
        "venue": "VARCHAR(255) DEFAULT ''",
        "meeting_date": "TIMESTAMP NULL",
        "attendees": "TEXT DEFAULT '[]'",
    }

    inspector = inspect(engine)
    if "meetings" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("meetings")}
    with engine.begin() as conn:
        for name, ddl in meeting_columns.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE meetings ADD COLUMN {name} {ddl}"))
