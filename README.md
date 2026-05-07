# Finora Backend

> Enterprise Financial Intelligence — AI-powered annual report analysis API.

Built with **FastAPI** · **SQLAlchemy** · **ChromaDB** · **Groq** · **Redis**

---

## API Overview

```
POST /api/auth/register          →  Create account
POST /api/auth/login             →  Get JWT token
GET  /api/auth/me                →  Current user profile + usage totals
GET  /api/auth/usage             →  Detailed token usage analytics

POST /api/reports/upload         →  Upload up to 3 PDF files
POST /api/reports/process/:id    →  Run AI extraction pipeline
GET  /api/reports/progress/:id   →  Real-time progress (Redis-backed)
GET  /api/reports/set/:id        →  Fetch report set + insights
GET  /api/reports/history        →  Past report sets
GET  /api/reports/compare/:id    →  Side-by-side multi-company comparison
GET  /api/reports/compare_ai/:id →  AI-generated comparative analysis (cached)
DELETE /api/reports/set/:id      →  Delete a report set + cleanup

POST /api/chat/session           →  SSE streaming chat with RAG context
POST /api/chat/turn              →  Persist a chat turn
GET  /api/chat/messages          →  Load chat history
POST /api/chat/clear             →  Clear chat history

GET  /health                     →  Health check
GET  /                           →  Service info
```

---

## Local Development

### Prerequisites

- Python 3.12+
- A [Groq API key](https://console.groq.com/)

### Setup

```bash
git clone <repo-url>
cd finora-backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Environment Variables

Copy `.env.example` to `.env`. All variables have sensible defaults for local dev except `GROQ_API_KEY`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `AI_MODEL` | `openai/gpt-oss-120b` | Groq model for extraction |
| `DATABASE_URL` | `sqlite:///./finora.db` | SQLite or PostgreSQL |
| `REDIS_URL` | _(empty)_ | Redis for progress tracking. Empty = in-memory fallback |
| `JWT_SECRET` | dev-default | Change to a random 64-char string |
| `JWT_EXPIRY_HOURS` | `72` | Token lifetime |
| `ALLOWED_ORIGINS` | localhost | Comma-separated frontend URLs |
| `CHROMA_HOST` | _(empty)_ | Empty = local `./chroma_db/` |
| `CHROMA_PORT` | `8000` | ChromaDB HTTP server port |
| `ENVIRONMENT` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MAX_FILE_SIZE_MB` | `25` | PDF upload limit |
| `MAX_FILES_PER_SET` | `3` | Max PDFs per set |
| `MAX_TOKEN_BUDGET_PER_REPORT` | `500000` | Token cap per report |
| `MAX_MAP_CONCURRENCY` | `4` | Max concurrent LLM map agents |
| `MAX_GLOBAL_LLM_CONCURRENCY` | `6` | Global LLM call concurrency limit |
| `MAX_CONCURRENT_REPORTS` | `6` | Global processing concurrency limit |
| `EXTRACTION_WORKERS` | `3` | Process pool workers for CPU-bound PDF extraction |

---

## Project Structure

```
finora-backend/
├── main.py                    # FastAPI app, lifespan, middleware
├── config.py                  # Centralized env config
├── database.py                # SQLAlchemy engine, session, ORM models
├── models.py                  # Pydantic request/response schemas
├── ai_schema.py               # LLM structured output schemas (MapExtraction, ReportInsights, AIComparison)
├── logger.py                  # Structured JSON logging
├── requirements.txt           # Dependencies
├── Dockerfile                 # Multi-stage production container
├── .env.example               # Env template
├── routers/
│   ├── auth.py                # JWT auth (pure stdlib), registration, usage analytics
│   ├── reports.py             # Upload, process, history, compare, compare_ai, delete
│   └── chat.py                # SSE streaming, RAG retrieval, chat persistence
└── services/
    ├── financial_extractor.py # Intelligent PDF extraction engine (page scoring, table extraction, filtering)
    ├── table_extractor.py     # Legacy pdfplumber table-to-markdown helper
    ├── rag.py                 # Financial-aware chunking + ChromaDB storage/retrieval
    └── progress_tracker.py    # Redis-backed monotonic progress tracking
```

---

## How the AI Pipeline Works

### Phase 1 — Intelligent Financial Extraction
- Each page is scored for financial signal density using keyword lexicons, currency/percentage detection, and section classification
- Tables are selectively extracted only from financially relevant pages
- Low-value content (TOC, boilerplate, legal disclaimers) is aggressively filtered
- Output: prioritized LLM context (~150K chars max) and full RAG context (~300K chars max)

### Phase 2 — Parallel Downstream Processing
Two pipelines run concurrently after extraction:

**Pipeline A — Map-Reduce LLM Analysis**
- Dynamic agent count based on content volume (1–6 map agents)
- Each agent processes a semantic chunk with strict schema enforcement (`MapExtractionSchema`)
- Reduce agent synthesizes all extracts into structured `ReportInsights`
- Tables are injected into the reduce phase sorted by financial score

**Pipeline B — RAG Vectorization**
- Financial-aware chunking with metadata (page, section, table flag, financial score)
- Dense embedding storage in isolated ChromaDB collections (one per report)
- Retrieval with financial-score boosting for tables and statement sections

### Phase 3 — Dashboard & Chat
- Structured insights feed the frontend dashboard
- Chat uses RAG retrieval + full dashboard context for streaming SSE responses

---

## Production Deployment

The `Dockerfile` is optimized for **Google Cloud Run**:
- Multi-stage build for minimal image size
- Non-root user for security
- Health check endpoint
- Pre-warms ChromaDB embedding model on startup
- Single uvicorn worker (horizontal scaling via Cloud Run)

```bash
docker build -t finora-backend .
docker run -p 8080:8080 -e GROQ_API_KEY=... finora-backend
```

For full GCP deployment (Cloud SQL PostgreSQL, Memorystore Redis, ChromaDB Cloud Run service, Cloud Build CI/CD), set the appropriate environment variables and deploy the container.

---

## Performance Targets

| Metric | Target | How it's achieved |
|---|---|---|
| Concurrent PDFs | 3 | Global asyncio semaphore + Cloud Run concurrency |
| Max PDF size | 600 pages | Intelligent page filtering + tiered detail levels |
| Extraction speed | <60s for 600pp | Selective table extraction, fast text scoring |
| End-to-end latency | <2 min | Parallel map-reduce + vectorization pipelines |
| Progress accuracy | Real steps | Redis-backed progress with 10 granular pipeline stages |
| Horizontal scaling | Auto | Stateless design, Redis for shared state |

---

## Tech Stack

| Library | Purpose |
|---|---|
| **FastAPI** | ASGI web framework |
| **SQLAlchemy** | ORM — SQLite local, PostgreSQL prod |
| **ChromaDB** | Vector database for RAG |
| **Groq** | LLM API (async, streaming) |
| **PyMuPDF** | Fast PDF text extraction |
| **pdfplumber** | PDF table extraction |
| **langchain-text-splitters** | Financial semantic chunking |
| **json-repair** | Recovers malformed LLM JSON |
| **redis** | Production progress tracking |
| **pg8000** | PostgreSQL driver (Cloud SQL) |
| **uvicorn** | ASGI server |

---

## License

MIT
