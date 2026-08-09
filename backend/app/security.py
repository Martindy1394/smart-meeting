"""Password hashing (bcrypt) and JWT creation / verification / revocation."""
from __future__ import annotations

import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

logger = logging.getLogger("smart_meeting.security")

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=settings.bcrypt_rounds,
)

# In-process denylist fallback when Redis is down (single-worker / tests).
_local_denylist: dict[str, float] = {}


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except ValueError:
        return False


def _new_jti() -> str:
    return secrets.token_urlsafe(24)


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": "access",
        "jti": _new_jti(),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=max(1, int(settings.refresh_token_expire_days)))
    payload: dict[str, Any] = {
        "sub": subject,
        "type": "refresh",
        "jti": _new_jti(),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def issue_token_pair(subject: str, extra: dict[str, Any] | None = None) -> dict[str, str]:
    return {
        "access_token": create_access_token(subject, extra=extra),
        "refresh_token": create_refresh_token(subject),
        "token_type": "bearer",
    }


def _denylist_key(jti: str) -> str:
    return f"smartmeeting:jwt:deny:{jti}"


def revoke_token(token: str) -> None:
    """Revoke a JWT by jti until its natural expiry (Redis + local fallback)."""
    payload = decode_token_unverified(token)
    if not payload:
        return
    jti = payload.get("jti")
    exp = int(payload.get("exp") or 0)
    if not jti:
        return
    ttl = max(1, exp - int(time.time()))
    _local_denylist[jti] = float(exp)
    try:
        from .services import redis_store

        client = redis_store.get_client()
        if client is not None:
            client.setex(_denylist_key(jti), ttl, b"1")
    except Exception as exc:
        logger.warning("JWT revoke Redis write failed (%s); local denylist only.", exc)


def is_revoked(payload: dict[str, Any]) -> bool:
    jti = payload.get("jti")
    if not jti:
        return False
    exp = _local_denylist.get(jti)
    if exp is not None:
        if exp < time.time():
            _local_denylist.pop(jti, None)
            return False
        return True
    try:
        from .services import redis_store

        client = redis_store.get_client()
        if client is not None and client.exists(_denylist_key(jti)):
            return True
    except Exception:
        pass
    return False


def decode_token_unverified(token: str) -> dict[str, Any] | None:
    try:
        return jwt.get_unverified_claims(token)
    except JWTError:
        return None


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None
    if payload.get("type") not in {None, "access"}:
        # Legacy tokens without type are treated as access for one release.
        if payload.get("type") == "refresh":
            return None
    if is_revoked(payload):
        return None
    return payload


def decode_refresh_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None
    if payload.get("type") != "refresh":
        return None
    if is_revoked(payload):
        return None
    return payload


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
