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
    user_columns = {
        "username": "VARCHAR(64) DEFAULT ''",
        "first_name": "VARCHAR(128) DEFAULT ''",
        "last_name": "VARCHAR(128) DEFAULT ''",
        "position": "VARCHAR(128) DEFAULT ''",
        "workplace": "VARCHAR(255) DEFAULT ''",
    }

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "meetings" in tables:
        existing = {col["name"] for col in inspector.get_columns("meetings")}
        with engine.begin() as conn:
            for name, ddl in meeting_columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE meetings ADD COLUMN {name} {ddl}"))

    if "users" in tables:
        existing = {col["name"] for col in inspector.get_columns("users")}
        with engine.begin() as conn:
            for name, ddl in user_columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {name} {ddl}"))
            # Backfill usernames for legacy email-only accounts.
            rows = conn.execute(
                text(
                    "SELECT id, email, username, full_name FROM users "
                    "WHERE username IS NULL OR username = ''"
                )
            ).fetchall()
            used: set[str] = {
                r[0]
                for r in conn.execute(
                    text("SELECT username FROM users WHERE username IS NOT NULL AND username != ''")
                ).fetchall()
            }
            for user_id, email, _username, full_name in rows:
                base = (email or "user").split("@")[0].lower()
                base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base) or "user"
                candidate = base[:64]
                n = 1
                while candidate in used:
                    suffix = f"_{n}"
                    candidate = f"{base[: max(1, 64 - len(suffix))]}{suffix}"
                    n += 1
                used.add(candidate)
                first = ""
                last = ""
                if full_name:
                    parts = str(full_name).strip().split(None, 1)
                    first = parts[0] if parts else ""
                    last = parts[1] if len(parts) > 1 else ""
                conn.execute(
                    text(
                        "UPDATE users SET username = :u, "
                        "first_name = COALESCE(NULLIF(first_name, ''), :f), "
                        "last_name = COALESCE(NULLIF(last_name, ''), :l) "
                        "WHERE id = :id"
                    ),
                    {"u": candidate, "f": first, "l": last, "id": user_id},
                )
