"""
Database engine, session, and ORM models.
Supports SQLite (local dev) and PostgreSQL (Cloud SQL) via DATABASE_URL env var.
"""
from sqlalchemy import (
    create_engine, Column, String, DateTime, JSON, Integer, Text,
    Float, Boolean, ForeignKey, Index, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from config import DATABASE_URL

# ── Engine setup ──────────────────────────────────────────────────────────
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,  # Reconnect stale connections (important for Cloud SQL)
    pool_recycle=1800,     # Recycle connections every 30 min
)

# Enable WAL mode for SQLite (concurrent reads + single writer)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────

class User(Base):
    """Email-based auth user."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    report_sets = relationship("ReportSet", back_populates="user", lazy="dynamic")
    usage_records = relationship("UsageRecord", back_populates="user", lazy="dynamic")


class ReportSet(Base):
    __tablename__ = "report_sets"

    id = Column(String(36), primary_key=True, index=True)
    report_ids = Column(JSON)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ai_comparison = Column(JSON, nullable=True)

    # Relationships
    user = relationship("User", back_populates="report_sets")


class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    safe_filename = Column(String(255), nullable=False)  # sanitized filename
    storage_path = Column(String(500))
    status = Column(String(20), nullable=False, default="queued")
    content_hash = Column(String(64), index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    insights = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    extracted_text = Column(Text, nullable=True)
    token_usage_input = Column(Integer, nullable=True, default=0)
    token_usage_output = Column(Integer, nullable=True, default=0)
    processing_duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processing_started_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_report_user_hash", "user_id", "content_hash"),
        Index("idx_report_status", "status"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    report_set_id = Column(String(36), index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UsageRecord(Base):
    """Per-request token usage tracking for cost analytics."""
    __tablename__ = "usage_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    report_id = Column(String(36), nullable=True)
    operation = Column(String(50), nullable=False)  # "extraction", "chat", "summary"
    model = Column(String(100), nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)  # estimated cost
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="usage_records")

    __table_args__ = (
        Index("idx_usage_user_date", "user_id", "created_at"),
    )


# ── Init / Session ───────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
