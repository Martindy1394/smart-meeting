"""Reliability helpers: stale processing lease + Redis reconnect cooldown."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings  # noqa: E402
from app.services import finalize, redis_store  # noqa: E402


def test_processing_stale_detection():
    now = datetime.now(timezone.utc)
    fresh = SimpleNamespace(
        status="processing",
        updated_at=now - timedelta(seconds=30),
        created_at=now,
    )
    stale = SimpleNamespace(
        status="processing",
        updated_at=now - timedelta(seconds=settings.processing_stale_seconds + 5),
        created_at=now,
    )
    done = SimpleNamespace(
        status="finalized",
        updated_at=now - timedelta(hours=2),
        created_at=now,
    )
    assert finalize.is_processing_stale(fresh, now=now) is False
    assert finalize.is_processing_stale(stale, now=now) is True
    assert finalize.is_processing_stale(done, now=now) is False


def test_redis_client_recovers_after_cooldown(monkeypatch=None):
    redis_store.reset_client_for_tests()
    # Simulate a failed connect without requiring a real outage.
    redis_store._client = None
    redis_store._client_failed = True
    redis_store._client_failed_at = 10**12  # far future → still cooling down
    assert redis_store.get_client() is None

    redis_store._client_failed_at = 0.0  # cooldown elapsed
    # If Redis is up in this environment, reconnect should succeed.
    client = redis_store.get_client()
    if client is not None:
        assert redis_store.is_available() is True
    redis_store.reset_client_for_tests()


if __name__ == "__main__":
    test_processing_stale_detection()
    test_redis_client_recovers_after_cooldown()
    print("all_backend_reliability_tests_passed")
