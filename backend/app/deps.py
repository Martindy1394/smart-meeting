"""Authentication dependencies for protected routes."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .security import decode_access_token

_bearer = HTTPBearer(auto_error=False)

_credentials_exc = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None or not creds.credentials:
        raise _credentials_exc
    payload = decode_access_token(creds.credentials)
    if payload is None:
        raise _credentials_exc
    user_id = payload.get("sub")
    if not user_id:
        raise _credentials_exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _credentials_exc
    return user


def get_user_from_token(token: str, db: Session) -> User | None:
    """Resolve a user from a raw token string (used by the WebSocket handler)."""
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user
