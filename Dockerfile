# Finora Backend — Production Dockerfile
# Optimized for Google Cloud Run deployment
# ----------------------------------------

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies into a virtual environment
# This keeps the final image clean and small
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Final image ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create directories with proper permissions
RUN mkdir -p /app/reports /app/chroma_db && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Cloud Run sets PORT env var; default to 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + (__import__('os').environ.get('PORT', '8080')) + '/health')" || exit 1

# Pre-warm ChromaDB embedding model (~80 MB ONNX) so first upload isn't slow.
# Then start uvicorn. Workers=1 because Cloud Run handles horizontal scaling.
# Timeout-keep-alive ensures connections stay open for SSE/streaming.
CMD ["sh", "-c", "python -c 'from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()' && uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120 --loop uvloop"]
