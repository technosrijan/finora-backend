"""
Centralized configuration loaded from environment variables.
Designed for Google Cloud Run + Cloud SQL + ChromaDB Cloud Run deployment.
"""
import os
from dotenv import load_dotenv

# Load .env from current directory (for local dev — backend/ dir)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
# Also check parent directory (for monorepo dev setup)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Database ──────────────────────────────────────────────────────────────
# Local dev:  sqlite:///./finora.db
# Cloud SQL:  postgresql+pg8000://USER:PASSWORD@/finora?unix_sock=/cloudsql/PROJECT:REGION:INSTANCE/.s.PGSQL.5432
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./finora.db")

# ── Redis (Progress tracking & caching) ───────────────────────────────────
# Format: redis://host:port/db  or  redis://username:password@host:port/db
# On GCP: use Memorystore Redis or Redis Enterprise
# Leave empty to use in-memory fallback
REDIS_URL = os.environ.get("REDIS_URL", "")

# ── Auth ──────────────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("JWT_SECRET", "finora-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "72"))

# ── AI / LLM ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-oss-120b")

# ── Limits ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "25"))
MAX_FILES_PER_SET = int(os.environ.get("MAX_FILES_PER_SET", "3"))
MAX_TOKEN_BUDGET_PER_REPORT = int(os.environ.get("MAX_TOKEN_BUDGET_PER_REPORT", "500000"))

# ── Concurrency ───────────────────────────────────────────────────────────
# Max concurrent LLM map agents per report
MAX_MAP_CONCURRENCY = int(os.environ.get("MAX_MAP_CONCURRENCY", "4"))
# Max concurrent LLM calls GLOBALLY across all reports (prevents API rate limits)
MAX_GLOBAL_LLM_CONCURRENCY = int(os.environ.get("MAX_GLOBAL_LLM_CONCURRENCY", "6"))
# Max concurrent reports being processed globally
MAX_CONCURRENT_REPORTS = int(os.environ.get("MAX_CONCURRENT_REPORTS", "6"))
# Number of process pool workers for CPU-bound PDF extraction
# Set to 1 to disable multi-process extraction (useful for single-core local dev)
EXTRACTION_WORKERS = int(os.environ.get("EXTRACTION_WORKERS", "3"))

# ── CORS ──────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8080"
).split(",")

# ── Storage ───────────────────────────────────────────────────────────────
# Local dev: relative paths inside the backend directory
# Production: use GCS-mounted paths (managed by Cloud Run volume mount or GCS FUSE)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "reports"))
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))

# ── ChromaDB HTTP Server (Production) ────────────────────────────────────
# Set CHROMA_HOST to the Cloud Run URL of the finora-chromadb service to enable
# HTTP mode (required for production — local disk is ephemeral on Cloud Run).
# Leave empty to use local PersistentClient (local dev only).
# Example: CHROMA_HOST=finora-chromadb-xxxx-uc.a.run.app
CHROMA_HOST = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")

# ── Server ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")  # development | staging | production
