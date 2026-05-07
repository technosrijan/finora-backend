"""
Finora Backend — FastAPI application entry point.
Designed for Google Cloud Run deployment.

Key architectural decisions:
- Single uvicorn worker per container (Cloud Run scales horizontally)
- Lifespan events for graceful startup/shutdown
- Structured JSON logging in production
- Global exception handler with request IDs
"""
import os
import sys
from contextlib import asynccontextmanager

from config import GROQ_API_KEY, ALLOWED_ORIGINS, UPLOAD_DIR, ENVIRONMENT

if not GROQ_API_KEY:
    sys.stderr.write("FATAL: GROQ_API_KEY environment variable is not set. Exiting.\n")
    sys.exit(1)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from database import init_db
from logger import get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    init_db()
    # Initialize process pool for PDF extraction (called here so it binds
    # to the correct event loop and shuts down gracefully on SIGTERM)
    from routers.reports import init_extraction_pool
    init_extraction_pool()
    logger.info("Database initialized")
    logger.info(f"Finora API starting ({ENVIRONMENT})")
    yield
    # Shutdown
    from routers.reports import shutdown_extraction_pool
    shutdown_extraction_pool()
    logger.info("Finora API shutting down")


app = FastAPI(
    title="Finora Backend API",
    description="Enterprise Financial Intelligence — AI-powered annual report analysis",
    version="2.1.0",
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url=None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
_cors_origins = ["*"] if ENVIRONMENT == "development" else ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── Request ID middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
            "request_id": request_id,
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────
from routers import reports, chat, auth

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


# ── Health check (unauthenticated — used by Cloud Run) ────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "2.1.0",
        "environment": ENVIRONMENT,
    }


@app.get("/")
def root():
    return {"service": "Finora API", "version": "2.1.0", "status": "running"}
