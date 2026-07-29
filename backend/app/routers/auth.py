"""Authentication routes: signup, login, refresh, logout, current user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import get_current_user
from ..limiter import auth_rate_limit
from ..models import User
from ..schemas import (
    LoginRequest,
    LogoutRequest,
    ProfileUpdateRequest,
    RefreshRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
)
from ..security import (
    decode_refresh_token,
    hash_password,
    issue_token_pair,
    revoke_token,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        first_name=user.first_name or "",
        last_name=user.last_name or "",
        position=user.position or "",
        workplace=user.workplace or "",
        full_name=user.display_name(),
        created_at=user.created_at,
    )


def _token_response(user: User) -> TokenResponse:
    pair = issue_token_pair(
        subject=user.id, extra={"username": user.username, "email": user.email}
    )
    return TokenResponse(
        access_token=pair["access_token"],
        refresh_token=pair["refresh_token"],
        token_type="bearer",
        expires_in=int(settings.access_token_expire_minutes) * 60,
        user=_user_response(user),
    )


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_rate_limit())],
)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    email = str(payload.email).lower().strip()
    username = payload.username.strip().lower()

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That username is already taken.",
        )
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this working email already exists.",
        )

    full_name = f"{payload.first_name} {payload.last_name}".strip()
    user = User(
        username=username,
        email=email,
        first_name=payload.first_name.strip(),
        last_name=payload.last_name.strip(),
        position=payload.position.strip(),
        workplace=payload.workplace.strip(),
        full_name=full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit())],
)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    username = payload.username.strip().lower()
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled."
        )
    return _token_response(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    dependencies=[Depends(auth_rate_limit())],
)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    claims = decode_refresh_token(payload.refresh_token)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token.",
        )
    user = db.get(User, claims.get("sub"))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked refresh token.",
        )
    # Rotate: revoke the presented refresh token, issue a new pair.
    revoke_token(payload.refresh_token)
    return _token_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: LogoutRequest | None = None):
    """Revoke presented access/refresh tokens (best-effort). Auth optional so
    expired access tokens can still kill a refresh token.
    """
    body = payload or LogoutRequest()
    if body.access_token:
        revoke_token(body.access_token)
    if body.refresh_token:
        revoke_token(body.refresh_token)
    return None


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    return _user_response(current_user)


@router.patch("/me", response_model=UserResponse)
def update_me(
    payload: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = payload.model_dump(exclude_unset=True)

    if "username" in data and data["username"]:
        username = data["username"]
        clash = (
            db.query(User)
            .filter(User.username == username, User.id != current_user.id)
            .first()
        )
        if clash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="That username is already taken.",
            )
        current_user.username = username

    if "email" in data and data["email"]:
        email = str(data["email"]).lower().strip()
        clash = (
            db.query(User)
            .filter(User.email == email, User.id != current_user.id)
            .first()
        )
        if clash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this working email already exists.",
            )
        current_user.email = email

    for field in ("first_name", "last_name", "position", "workplace"):
        if field in data and data[field] is not None:
            setattr(current_user, field, str(data[field]).strip())

    if data.get("password"):
        current_user.hashed_password = hash_password(data["password"])

    current_user.full_name = (
        f"{current_user.first_name} {current_user.last_name}".strip()
        or current_user.full_name
    )
    db.commit()
    db.refresh(current_user)
    return _user_response(current_user)
