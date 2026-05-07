"""
Email-based authentication router.
- POST /api/auth/register  — create account
- POST /api/auth/login     — get JWT token
- GET  /api/auth/me        — get current user profile
- GET  /api/auth/usage     — get user usage/cost analytics
"""
import uuid
import hashlib
import hmac
import time
import json
import base64
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from database import get_db, User, UsageRecord, Report, ReportSet
from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS
from logger import get_logger

logger = get_logger("auth")
router = APIRouter()


# ── JWT utilities (no PyJWT dependency — pure stdlib) ──────────────────────

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s: str) -> bytes:
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)

def create_jwt(user_id: str, email: str) -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    now = int(time.time())
    payload = _b64url_encode(json.dumps({
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + JWT_EXPIRY_HOURS * 3600,
    }).encode())
    signature = _b64url_encode(
        hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"

def verify_jwt(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")
        header, payload, signature = parts
        expected_sig = _b64url_encode(
            hmac.new(JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected_sig):
            raise ValueError("Invalid signature")
        claims = json.loads(_b64url_decode(payload))
        if claims.get("exp", 0) < int(time.time()):
            raise ValueError("Token expired")
        return claims
    except Exception as e:
        raise ValueError(f"Invalid token: {e}")


# ── Password hashing (SHA-256 + salt, no bcrypt dependency) ────────────────

def hash_password(password: str) -> str:
    salt = uuid.uuid4().hex
    hashed = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, stored_hash: str) -> bool:
    salt, hashed = stored_hash.split(":", 1)
    return hmac.compare_digest(
        hashlib.sha256(f"{salt}:{password}".encode()).hexdigest(),
        hashed
    )


# ── Dependency: get current user from Authorization header ─────────────────

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header[7:]
    try:
        claims = verify_jwt(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    user = db.query(User).filter(User.id == claims["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated")
    return user


# ── Schemas ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=100)

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user: dict

class UserProfile(BaseModel):
    id: str
    email: str
    display_name: str | None
    created_at: str
    total_reports: int
    total_tokens_used: int
    total_cost_usd: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    # Normalize email
    email = payload.email.strip().lower()

    # Check if email exists
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name.strip() or email.split("@")[0],
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_jwt(user.id, user.email)
    logger.info(f"New user registered: {email}")

    return AuthResponse(
        token=token,
        user={
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
        }
    )


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account has been deactivated")

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_jwt(user.id, user.email)
    logger.info(f"User logged in: {email}")

    return AuthResponse(
        token=token,
        user={
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
        }
    )


@router.get("/me", response_model=UserProfile)
def get_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_reports = db.query(Report).filter(Report.user_id == user.id, Report.status == "ready").count()

    usage_agg = db.query(
        func.sum(UsageRecord.input_tokens),
        func.sum(UsageRecord.output_tokens),
        func.sum(UsageRecord.cost_usd),
    ).filter(UsageRecord.user_id == user.id).first()

    total_tokens = (usage_agg[0] or 0) + (usage_agg[1] or 0)
    total_cost = usage_agg[2] or 0.0

    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at.isoformat() if user.created_at else "",
        total_reports=total_reports,
        total_tokens_used=total_tokens,
        total_cost_usd=round(total_cost, 4),
    )


@router.get("/usage")
def get_usage(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return detailed usage analytics for the current user."""
    records = db.query(UsageRecord).filter(
        UsageRecord.user_id == user.id
    ).order_by(UsageRecord.created_at.desc()).limit(100).all()

    total_in = sum(r.input_tokens or 0 for r in records)
    total_out = sum(r.output_tokens or 0 for r in records)
    total_cost = sum(r.cost_usd or 0 for r in records)

    return {
        "summary": {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 4),
            "total_requests": len(records),
        },
        "recent": [
            {
                "operation": r.operation,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd or 0, 6),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records[:20]
        ]
    }

