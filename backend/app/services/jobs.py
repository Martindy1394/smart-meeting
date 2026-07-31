"""Background job queue for finalize / retranscribe / summarize.

Uses Redis lists when available (RQ-compatible key prefix). Falls back to
in-process ``asyncio.create_task`` so local/dev still works without a worker.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, Callable

from ..config import settings
from . import pipeline_metrics, redis_store

logger = logging.getLogger("smart_meeting.jobs")

_QUEUE_KEY = "smartmeeting:jobs:queue"
_JOB_PREFIX = "smartmeeting:jobs:status:"
_handlers: dict[str, Callable[[dict], dict]] = {}
_local_lock = threading.Lock()
_local_jobs: dict[str, dict] = {}


def register(job_type: str, handler: Callable[[dict], dict]) -> None:
    _handlers[job_type] = handler


def _job_key(job_id: str) -> str:
    return f"{_JOB_PREFIX}{job_id}"


def _save(job: dict) -> None:
    client = redis_store.get_client()
    payload = json.dumps(job, ensure_ascii=False)
    if client is not None:
        try:
            client.setex(_job_key(job["id"]), 86400, payload)
            return
        except Exception as exc:
            logger.debug("Redis job save failed: %s", exc)
    with _local_lock:
        _local_jobs[job["id"]] = dict(job)


def get_job(job_id: str) -> dict | None:
    client = redis_store.get_client()
    if client is not None:
        try:
            raw = client.get(_job_key(job_id))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    with _local_lock:
        job = _local_jobs.get(job_id)
        return dict(job) if job else None


def enqueue(job_type: str, payload: dict | None = None, *, dedupe_key: str | None = None) -> dict:
    """Enqueue a job. When ``dedupe_key`` matches a running/queued job, reuse it."""
    payload = dict(payload or {})
    if dedupe_key:
        existing = _find_by_dedupe(job_type, dedupe_key)
        if existing and existing.get("status") in {"queued", "running"}:
            return existing

    job = {
        "id": str(uuid.uuid4()),
        "type": job_type,
        "payload": payload,
        "dedupe_key": dedupe_key or "",
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
        "result": None,
        "error": None,
    }
    _save(job)
    client = redis_store.get_client()
    if client is not None and bool(getattr(settings, "jobs_use_redis_queue", True)):
        try:
            client.rpush(_QUEUE_KEY, job["id"])
            depth = int(client.llen(_QUEUE_KEY) or 0)
            pipeline_metrics.set_queue_depth("jobs", depth)
            # Eager local worker thread if no external RQ worker is assumed.
            if bool(getattr(settings, "jobs_inline_worker", True)):
                threading.Thread(target=_process_one, args=(job["id"],), daemon=True).start()
            return job
        except Exception as exc:
            logger.warning("Redis enqueue failed (%s); running inline", exc)
    # Inline fallback.
    threading.Thread(target=_process_one, args=(job["id"],), daemon=True).start()
    return job


def _find_by_dedupe(job_type: str, dedupe_key: str) -> dict | None:
    # Best-effort scan of recent local jobs; Redis workers set status keys.
    with _local_lock:
        for job in _local_jobs.values():
            if (
                job.get("type") == job_type
                and job.get("dedupe_key") == dedupe_key
                and job.get("status") in {"queued", "running"}
            ):
                return dict(job)
    return None


def _process_one(job_id: str) -> None:
    job = get_job(job_id)
    if not job or job.get("status") not in {"queued", "running"}:
        return
    if job.get("status") == "running":
        return
    job["status"] = "running"
    job["updated_at"] = time.time()
    _save(job)
    handler = _handlers.get(job.get("type") or "")
    if handler is None:
        job["status"] = "failed"
        job["error"] = f"No handler for job type {job.get('type')}"
        job["updated_at"] = time.time()
        _save(job)
        return
    try:
        with pipeline_metrics.track(f"job.{job['type']}"):
            result = handler(job.get("payload") or {})
        job["status"] = "completed"
        job["result"] = result
        job["error"] = None
    except Exception as exc:
        logger.exception("Job %s (%s) failed", job_id, job.get("type"))
        job["status"] = "failed"
        job["error"] = str(exc)
    job["updated_at"] = time.time()
    _save(job)


def queue_depth() -> int:
    client = redis_store.get_client()
    if client is None:
        return 0
    try:
        return int(client.llen(_QUEUE_KEY) or 0)
    except Exception:
        return 0
