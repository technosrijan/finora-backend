"""
Reports router — upload, process, fetch, compare, history.
All endpoints are auth-protected via get_current_user dependency.

Parallel processing architecture:
  1. Intelligent financial extraction (page scoring, table extraction, filtering)
  2. Parallel pipelines:
     A. Map-Reduce LLM analysis (dynamic agent count based on content)
     B. Vectorization (financial-aware chunking + ChromaDB embedding)
  3. Reduce synthesis and dashboard generation

Production robustness:
  - Redis-backed progress tracking (survives restarts)
  - Concurrency limiting (respects MAX_CONCURRENT_REPORTS)
  - Graceful degradation (partial failures don't kill the entire pipeline)
"""
import os
import re
import uuid
import json
import json_repair
import hashlib
import random
import asyncio
import time as _time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import groq
from pydantic import ValidationError

from database import get_db, Report, ReportSet, User, UsageRecord
from models import ReportRecordSchema, ReportSetSchema, HistoryItemSchema
from services.financial_extractor import analyze_pdf, get_dynamic_map_config, split_into_semantic_chunks
from services.rag import chunk_document_financial, store_chunks, retrieve_relevant_context, delete_collection
from services.progress_tracker import ProgressTracker, get_progress
from ai_schema import MapExtractionSchema, ReportInsights, AIComparison
from routers.auth import get_current_user
from config import (
    GROQ_API_KEY, AI_MODEL, MAX_MAP_CONCURRENCY, MAX_GLOBAL_LLM_CONCURRENCY, EXTRACTION_WORKERS,
    MAX_FILE_SIZE_MB, MAX_FILES_PER_SET, MAX_TOKEN_BUDGET_PER_REPORT, UPLOAD_DIR,
)
from logger import get_logger

logger = get_logger("reports")

SYSTEM_PROMPT = "You are Finora, an elite AI financial extraction engine."

router = APIRouter()

MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES = MAX_FILES_PER_SET

# Global concurrency limit for report processing
# This prevents overwhelming the LLM API and keeps memory under control
_report_processing_sem = asyncio.Semaphore(3)

# Process pool for CPU-bound PDF extraction (initialized in lifespan, see below)
_extraction_pool: ProcessPoolExecutor | None = None

# Global LLM API semaphore — ALL reports share this
# Prevents 3 PDFs × 4 map agents = 12 concurrent calls hammering Groq
_global_llm_sem = asyncio.Semaphore(MAX_GLOBAL_LLM_CONCURRENCY)


def init_extraction_pool():
    """Called from main.py lifespan startup."""
    global _extraction_pool
    if _extraction_pool is None:
        _extraction_pool = ProcessPoolExecutor(max_workers=EXTRACTION_WORKERS)
        logger.info(f"Extraction pool initialized with {EXTRACTION_WORKERS} workers")


def shutdown_extraction_pool():
    """Called from main.py lifespan shutdown."""
    global _extraction_pool
    if _extraction_pool is not None:
        _extraction_pool.shutdown(wait=True)
        logger.info("Extraction pool shut down gracefully")
        _extraction_pool = None


def secure_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-.]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename or "unnamed.pdf"


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def row_to_record(r: Report):
    return ReportRecordSchema(
        id=r.id,
        filename=r.filename,
        status=r.status,
        insights=r.insights,
        summary=r.summary,
        error=r.error,
        created_at=r.created_at.isoformat() if r.created_at else "",
        token_usage_input=r.token_usage_input or 0,
        token_usage_output=r.token_usage_output or 0,
    )


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate for tracking (Groq pricing varies)."""
    return (input_tokens * 0.0000005) + (output_tokens * 0.0000015)


# ── Upload Endpoints ────────────────────────────────────────────────────────

@router.post("/upload", response_model=ReportSetSchema)
async def upload_report_set(
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_FILES} PDFs allowed.")

    report_ids = []
    records = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"{file.filename}: only PDF files are accepted.")

        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(status_code=400, detail=f"{file.filename} exceeds {MAX_FILE_SIZE_MB}MB limit.")
        if len(content) < 100:
            raise HTTPException(status_code=400, detail=f"{file.filename} appears to be empty.")

        file_hash = sha256_hex(content)

        # Cache: reuse existing ready report for same user + same file hash
        existing = db.query(Report).filter(
            Report.content_hash == file_hash,
            Report.status == "ready",
            Report.user_id == user.id,
        ).order_by(Report.created_at.desc()).first()

        if existing:
            report_ids.append(existing.id)
            records.append(row_to_record(existing))
            logger.info(f"Cache hit for {file.filename} (hash={file_hash[:12]})")
            continue

        report_id = str(uuid.uuid4())
        safe_name = secure_filename(file.filename)
        storage_dir = os.path.join(UPLOAD_DIR, report_id)
        os.makedirs(storage_dir, exist_ok=True)
        storage_path = os.path.join(storage_dir, safe_name)

        with open(storage_path, "wb") as f:
            f.write(content)

        new_report = Report(
            id=report_id,
            filename=file.filename,
            safe_filename=safe_name,
            storage_path=storage_path,
            status="queued",
            content_hash=file_hash,
            user_id=user.id,
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)

        report_ids.append(report_id)
        records.append(row_to_record(new_report))
        logger.info(f"Uploaded {file.filename} -> {report_id}")

    set_id = str(uuid.uuid4())
    new_set = ReportSet(
        id=set_id,
        report_ids=report_ids,
        user_id=user.id,
    )
    db.add(new_set)
    db.commit()
    db.refresh(new_set)

    return ReportSetSchema(
        id=set_id,
        reports=records,
        created_at=new_set.created_at.isoformat() if new_set.created_at else datetime.now(timezone.utc).isoformat(),
    )


@router.get("/set/{set_id}", response_model=ReportSetSchema)
def get_report_set(set_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Report set not found.")

    reports = db.query(Report).filter(Report.id.in_(report_set.report_ids)).all()
    by_id = {r.id: r for r in reports}
    ordered = [row_to_record(by_id[rid]) for rid in report_set.report_ids if rid in by_id]

    return ReportSetSchema(
        id=report_set.id,
        reports=ordered,
        created_at=report_set.created_at.isoformat() if report_set.created_at else "",
    )


@router.get("/progress/{report_id}")
async def get_report_progress(
    report_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get real-time progress for a report being processed.

    The database is the source of truth for terminal states (ready/error).
    The progress tracker is only consulted for actively processing reports.
    This prevents stale progress from showing when users navigate back and forth.
    """
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == user.id,
    ).first()

    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    # BUG FIX #1: DB is the authoritative source for terminal states.
    # If the DB says ready/error, return that immediately regardless of
    # any stale in-memory/Redis progress tracker state.
    if report.status == "ready":
        return {
            "status": "ready",
            "progress": 100,
            "current_step": "Final completion",
            "total_steps": 10,
            "completed_steps": 10,
            "error": None,
        }

    if report.status == "error":
        return {
            "status": "error",
            "progress": 0,
            "current_step": "Error",
            "total_steps": 10,
            "completed_steps": 0,
            "error": report.error,
        }

    # For actively processing reports, use the live progress tracker
    progress = get_progress(report_id)
    if not progress:
        return {
            "status": report.status,
            "progress": 0,
            "current_step": report.status,
            "total_steps": 10,
            "completed_steps": 0,
            "error": report.error,
        }

    return progress


# ── Core Processing Pipeline ────────────────────────────────────────────────

async def _run_llm_pipeline(
    report_id: str,
    extraction,
    progress: ProgressTracker,
    client: groq.AsyncGroq,
    token_totals: dict,
) -> dict:
    """
    Pipeline A: Map-Reduce LLM analysis.
    Returns final insights dict or None on failure.
    """
    # Dynamic chunking based on extracted content
    map_config = get_dynamic_map_config(extraction)
    chunk_count = map_config["chunk_count"]
    concurrency = min(map_config["concurrency"], MAX_MAP_CONCURRENCY)

    logger.info(f"[MAP-CONFIG] report={report_id[:12]} chunks={chunk_count} concurrency={concurrency} "
                f"llm_chars={map_config['llm_context_length']:,}")

    # Split LLM context into semantic chunks
    super_chunks = split_into_semantic_chunks(extraction, chunk_count)
    if not super_chunks:
        raise Exception("No content available for LLM analysis after extraction.")

    progress.set_state(
        current_step=f"Launching {len(super_chunks)} map agents...",
        progress=40,
        completed_steps=4,
        pipeline_stage="llm_map",
    )

    sem = asyncio.Semaphore(concurrency)
    map_lock = asyncio.Lock()
    map_completed = 0

    async def bounded_map_agent(chunk, i):
        nonlocal map_completed
        await asyncio.sleep(i * 0.3)  # Slight stagger to avoid rate limits
        logger.info(f"[MAP] Agent {i+1}/{len(super_chunks)} starting...")
        async with sem:
            try:
                res = await _run_extraction_agent(
                    MapExtractionSchema, chunk, client, token_totals
                )
                if res:
                    logger.info(f"[MAP] Agent {i+1} success")
                    async with map_lock:
                        map_completed += 1
                        completed = map_completed
                    progress.set_state(
                        current_step=f"Map agent {i+1}/{len(super_chunks)} complete",
                        progress=40 + int((completed / max(1, len(super_chunks))) * 25),
                        completed_steps=5,
                        pipeline_stage="llm_map",
                    )
                else:
                    logger.warning(f"[MAP] Agent {i+1} returned empty")
                return res
            except Exception as e:
                logger.error(f"[MAP] Agent {i+1} failed: {e}")
                return None

    map_tasks = [bounded_map_agent(c, i) for i, c in enumerate(super_chunks)]
    map_results = await asyncio.gather(*map_tasks)
    valid_maps = [m for m in map_results if m]

    if len(valid_maps) == 0:
        raise Exception("All map agents failed. Could not extract any data.")

    logger.info(f"[MAP] {len(valid_maps)}/{len(super_chunks)} agents succeeded")

    # ── Reduce Phase ──
    progress.set_state(
        current_step="Analyzing and synthesizing results...",
        progress=70,
        completed_steps=6,
        pipeline_stage="llm_reduce",
    )

    map_json_str = json.dumps(valid_maps)
    reduce_context = f"--- MAP PHASE EXTRACTS ---\n{map_json_str}\n--- END MAP PHASE ---\n"

    tables_md = extraction.extracted_tables_markdown
    if tables_md:
        # Tables are already sorted by financial score (highest first).
        # We keep the top-scoring tables up to 80K chars.
        if len(tables_md) > 80_000:
            tables_md = tables_md[:80_000] + "\n...[ADDITIONAL TABLES TRUNCATED]..."
        reduce_context += "\n--- EXTRACTED FINANCIAL TABLES (sorted by importance) ---\n" + tables_md

    logger.info("[REDUCE] Starting synthesis...")
    final_insights = await _run_extraction_agent(ReportInsights, reduce_context, client, token_totals)

    if not final_insights:
        raise Exception("Reduce phase synthesis failed.")

    logger.info("[REDUCE] Synthesis complete")
    return final_insights


async def _run_vectorization_pipeline(
    report_id: str,
    extraction,
    progress: ProgressTracker,
) -> bool:
    """
    Pipeline B: Chunk and vectorize for RAG.
    Returns True on success, False on graceful failure.
    """
    try:
        progress.set_state(
            current_step="Chunking text for vector DB...",
            progress=45,
            completed_steps=4,
            pipeline_stage="vectorization",
        )

        # Financial-aware chunking with metadata
        chunks = chunk_document_financial(
            extraction.rag_context,
            tables_markdown=extraction.extracted_tables_markdown,
        )

        def _embedding_progress(batch_num: int, total_batches: int):
            base = 50
            span = 15
            pct = base + int((batch_num / max(1, total_batches)) * span)
            progress.set_state(
                current_step=f"Embedding chunks into vector DB ({batch_num}/{total_batches})...",
                progress=pct,
                completed_steps=7,
                pipeline_stage="vectorization",
            )

        progress.set_state(
            current_step=f"Embedding {len(chunks)} chunks into vector DB...",
            progress=50,
            completed_steps=7,
            pipeline_stage="vectorization",
        )

        store_chunks(report_id, chunks, progress_callback=_embedding_progress)
        logger.info(f"[RAG] Vectorization complete for {report_id[:12]} ({len(chunks)} chunks)")
        return True

    except Exception as e:
        logger.warning(f"[RAG] Vectorization failed for {report_id[:12]}: {e}")
        # Graceful degradation: RAG failure should not kill the whole report
        return False


async def _process_report_async(
    report_id: str,
    user_id: str,
    storage_path: str,
):
    """
    Async worker function for processing a single report.
    Orchestrates:
      1. Intelligent financial extraction
      2. Parallel LLM analysis + Vectorization
      3. Finalization
    """
    from database import SessionLocal

    progress = ProgressTracker(report_id)
    db = SessionLocal()
    start_time = _time.time()

    # Acquire global processing slot
    async with _report_processing_sem:
        try:
            progress.set_state(
                current_step="PDF preprocessing",
                progress=5,
                completed_steps=1,
                pipeline_stage="extraction",
            )

            if not os.path.exists(storage_path):
                raise Exception(f"PDF file not found at {storage_path}")

            # ── PHASE 1: Intelligent Financial Extraction ──
            progress.set_state(
                current_step="Financial page detection",
                progress=8,
                completed_steps=1,
                pipeline_stage="extraction",
            )

            # Run extraction in a separate process for true multi-core parallelism
            loop = asyncio.get_event_loop()
            pool = _extraction_pool or ProcessPoolExecutor(max_workers=1)
            extraction = await loop.run_in_executor(pool, analyze_pdf, storage_path)

            progress.set_state(
                current_step=f"Table extraction ({len(extraction.all_tables)} tables)",
                progress=15,
                completed_steps=2,
                pipeline_stage="extraction",
            )

            if extraction.financial_page_count == 0:
                raise Exception(
                    "Could not identify any financially relevant content in this PDF. "
                    "It may be an image-only PDF or contain no financial data."
                )

            progress.set_state(
                current_step=f"Financial prioritization ({extraction.financial_page_count}/{extraction.total_pages} pages)",
                progress=22,
                completed_steps=3,
                pipeline_stage="extraction",
            )

            logger.info(
                f"[EXTRACTION] report={report_id[:12]} pages={extraction.total_pages} "
                f"financial={extraction.financial_page_count} skipped={extraction.skipped_page_count} "
                f"llm_ctx={len(extraction.llm_context):,} rag_ctx={len(extraction.rag_context):,}"
            )

            if not GROQ_API_KEY:
                raise Exception("Groq API key not configured.")

            client = groq.AsyncGroq(api_key=GROQ_API_KEY)
            token_totals = {"input": 0, "output": 0}

            # ── PHASE 2: Parallel Pipelines ──
            # Launch LLM analysis and Vectorization concurrently
            progress.set_state(
                current_step="Launching parallel analysis and vectorization...",
                progress=30,
                completed_steps=4,
                pipeline_stage="parallel_processing",
            )

            llm_task = asyncio.create_task(
                _run_llm_pipeline(report_id, extraction, progress, client, token_totals)
            )
            rag_task = asyncio.create_task(
                _run_vectorization_pipeline(report_id, extraction, progress)
            )

            # Wait for both with individual exception handling
            llm_result = None
            rag_success = False

            try:
                llm_result = await llm_task
            except Exception as e:
                logger.error(f"[PIPELINE] LLM pipeline failed: {e}")
                # Cancel RAG if LLM fails (no point continuing)
                rag_task.cancel()
                raise

            try:
                rag_success = await rag_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[PIPELINE] RAG pipeline failed (non-critical): {e}")

            # ── PHASE 3: Finalize ──
            progress.set_state(
                current_step="Dashboard generation",
                progress=90,
                completed_steps=8,
                pipeline_stage="finalization",
            )

            total_in = token_totals["input"]
            total_out = token_totals["output"]
            duration = round(_time.time() - start_time, 2)

            # Prepare extracted text summary (trimmed for DB storage)
            trimmed = extraction.llm_context[:80_000]
            if len(extraction.llm_context) > 80_000:
                trimmed = trimmed[:40_000] + "\n\n...[MIDDLE TRUNCATED FOR STORAGE]...\n\n" + extraction.llm_context[-40_000:]

            report = db.query(Report).filter(Report.id == report_id).first()
            if report:
                report.status = "ready"
                report.insights = llm_result
                report.summary = llm_result.get("executive_summary") if llm_result else None
                report.extracted_text = trimmed
                report.token_usage_input = total_in
                report.token_usage_output = total_out
                report.processing_duration_seconds = duration
                report.error = None
                db.commit()

                # Track usage
                usage = UsageRecord(
                    user_id=user_id,
                    report_id=report_id,
                    operation="extraction",
                    model=AI_MODEL,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    cost_usd=_estimate_cost(total_in, total_out),
                )
                db.add(usage)
                db.commit()

            progress.complete()
            logger.info(
                f"[COMPLETE] report={report_id[:12]} | "
                f"Tokens: {total_in:,}+{total_out:,}={total_in+total_out:,} | "
                f"Duration: {duration}s | RAG: {'OK' if rag_success else 'FAILED'}"
            )

        except Exception as e:
            logger.error(f"[ERROR] Processing failed for {report_id[:12]}: {e}")
            progress.error(str(e)[:500])

            report = db.query(Report).filter(Report.id == report_id).first()
            if report:
                report.status = "error"
                report.error = str(e)[:500]
                report.processing_duration_seconds = round(_time.time() - start_time, 2)
                db.commit()

        finally:
            db.close()


# ── LLM Agent Utilities ─────────────────────────────────────────────────────

async def _run_extraction_agent(model_class, text, client: groq.AsyncGroq, token_totals: dict, max_retries: int = 5):
    """Run a single LLM extraction agent with retry logic."""
    schema_str = json.dumps(model_class.model_json_schema())

    if model_class.__name__ == "MapExtractionSchema":
        role_instruction = (
            "You are a Forensic Financial Analyst. Your goal is to extract every single "
            "numerical metric, ratio, operational KPI, and contingent liability from this text chunk. "
            "CRITICAL: If you see revenue, profit, net income, EBITDA, EPS, total assets, or debt figures, "
            "you MUST extract them with exact values, units, and labels. "
            "Also extract financial ratios and revenue segment breakdowns with segment names and amounts. "
            "Preserve all labels, units, dates, and contextual notes exactly as stated."
        )
    else:
        role_instruction = (
            "You are a Chief Financial Officer. Synthesize the raw data extracts into a definitive, "
            "board-level dashboard. Prioritize the most critical global metrics (Revenue, Profit, "
            "Margin, EPS, Cash Flow) and deduplicate findings. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. Look carefully at the EXTRACTED FINANCIAL TABLES section. Revenue, profit, and balance sheet "
            "data are often disclosed ONLY in tables, not in narrative text. If tables show revenue figures, "
            "you MUST include them in key_metrics and revenue_breakdown.\n"
            "2. For revenue_breakdown: if tables show absolute amounts by segment but not percentages, "
            "calculate the percentage share yourself (segment / total revenue × 100).\n"
            "3. You MUST provide: a financial_health_score (1-10), sentiment_score (-1 to 1), "
            "sector classification, 5-10 key_ratios, and a revenue_breakdown with 3-8 segments.\n"
            "4. For generated_charts, create 3-6 charts using 'bar', 'line', 'pie', or 'area' types.\n"
            "5. NEVER say revenue is 'not disclosed' if you see revenue figures in the tables or text."
        )

    base_prompt = (
        f"{role_instruction}\n\n"
        f"CRITICAL: Return ONLY a single raw JSON object. No text before or after it.\n"
        f"Do NOT use markdown code fences.\n"
        f"The JSON MUST strictly adhere to this schema:\n{schema_str}\n"
        f"Every object in every array must have ALL required keys.\n\n"
        f"--- CONTEXT START ---\n{text}\n--- CONTEXT END ---"
    )

    def _extract_json_str(raw: str) -> str:
        raw = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            return match.group(1)
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            return raw[start:end+1]
        return raw

    def _try_parse(raw_json: str):
        parsed = json_repair.loads(raw_json)
        validated = model_class.model_validate(parsed)
        return validated.model_dump()

    prompt = base_prompt
    for attempt in range(max_retries):
        try:
            if token_totals["input"] + token_totals["output"] > MAX_TOKEN_BUDGET_PER_REPORT:
                logger.warning(f"Token budget exceeded ({token_totals['input'] + token_totals['output']} tokens)")
                return None

            async with _global_llm_sem:
                completion = await client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT + " You ONLY output valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_completion_tokens=8192,
                    top_p=1,
                )

            if completion.usage:
                token_totals["input"] += completion.usage.prompt_tokens or 0
                token_totals["output"] += completion.usage.completion_tokens or 0

            resp_content = completion.choices[0].message.content
            if not resp_content:
                if attempt == max_retries - 1:
                    raise Exception("AI did not return any content.")
                continue

            json_str = _extract_json_str(resp_content)
            try:
                return _try_parse(json_str)
            except ValidationError as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Validation failed after {max_retries} attempts: {e}")
                prompt = base_prompt + f"\n\nPrevious response failed validation:\n{e}\nPlease fix the JSON."
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

        except groq.RateLimitError as e:
            retry_after = 2.0
            msg_str = str(e)
            wait_match = re.search(r'Please try again in (\d+\.?\d*)s', msg_str)
            if wait_match:
                retry_after = float(wait_match.group(1)) + 0.5
            else:
                retry_after = min(2 ** attempt + random.uniform(0.5, 1.5), 30)
            logger.info(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {retry_after:.1f}s...")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(retry_after)

        except groq.BadRequestError as e:
            msg_str = str(e)
            failed_gen = ""
            try:
                err_body = e.body if hasattr(e, 'body') else {}
                if isinstance(err_body, dict):
                    failed_gen = err_body.get("error", {}).get("failed_generation", "")
            except Exception:
                pass
            if failed_gen and len(failed_gen) > 50:
                try:
                    json_str = _extract_json_str(failed_gen)
                    result = _try_parse(json_str)
                    logger.info(f"Recovered failed_generation via json_repair (attempt {attempt+1})")
                    return result
                except Exception:
                    pass
            logger.warning(f"BadRequest (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(1)

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(0.5)


# ── Process Trigger ─────────────────────────────────────────────────────────

@router.post("/process/{report_id}")
async def process_report(
    report_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Trigger async processing for a report.
    Returns immediately; progress tracked via /progress/{report_id} endpoint.
    """
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == user.id,
        Report.status == "queued",
    ).first()

    if not report:
        return {"ok": True, "skipped": True}

    report.status = "processing"
    report.processing_started_at = datetime.now(timezone.utc)
    db.commit()

    # Initialize progress tracker immediately
    progress = ProgressTracker(report_id)
    progress.start()
    progress.set_state(
        current_step="Upload received",
        progress=2,
        completed_steps=0,
        pipeline_stage="processing",
    )

    # Schedule async processing in background
    background_tasks.add_task(
        _process_report_async,
        report_id=report_id,
        user_id=user.id,
        storage_path=report.storage_path,
    )

    return {
        "ok": True,
        "report_id": report_id,
        "message": "Processing started. Poll /progress/{report_id} for updates."
    }


# ── History & Comparison ────────────────────────────────────────────────────

@router.get("/history", response_model=list[HistoryItemSchema])
def list_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sets = db.query(ReportSet).filter(
        ReportSet.user_id == user.id
    ).order_by(ReportSet.created_at.desc()).limit(20).all()
    if not sets:
        return []

    all_ids = list(set(rid for s in sets for rid in s.report_ids))
    reports = db.query(Report).filter(Report.id.in_(all_ids)).all()
    by_id = {r.id: r for r in reports}

    result = []
    for s in sets:
        reps = [by_id[rid] for rid in s.report_ids if rid in by_id]
        result.append(HistoryItemSchema(
            id=s.id,
            created_at=s.created_at.isoformat() if s.created_at else "",
            filenames=[r.filename for r in reps],
            ready_count=len([r for r in reps if r.status == "ready"]),
            total_count=len(s.report_ids),
        ))
    return result


@router.delete("/set/{set_id}")
def delete_report_set(set_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Not found.")

    # Also clean up associated reports, files, and ChromaDB collections
    reports = db.query(Report).filter(Report.id.in_(report_set.report_ids)).all()
    for r in reports:
        try:
            if r.storage_path and os.path.exists(r.storage_path):
                os.remove(r.storage_path)
                # Try to remove parent directory
                parent = os.path.dirname(r.storage_path)
                if os.path.isdir(parent):
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to delete file for report {r.id}: {e}")

        # Delete ChromaDB collection
        try:
            delete_collection(r.id)
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection for report {r.id}: {e}")

        db.delete(r)

    db.delete(report_set)
    db.commit()
    return {"ok": True}


@router.get("/compare/{set_id}")
def compare_reports(set_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return structured comparison data for all ready reports in a set."""
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Report set not found.")

    reports = db.query(Report).filter(
        Report.id.in_(report_set.report_ids),
        Report.status == "ready",
    ).all()

    if len(reports) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 ready reports to compare.")

    comparison = {"reports": [], "common_metrics": [], "health_scores": []}

    all_metric_names = {}
    for r in reports:
        insights = r.insights or {}
        company = insights.get("company_name", r.filename)
        report_data = {
            "id": r.id,
            "company_name": company,
            "filename": r.filename,
            "period": insights.get("reporting_period", "N/A"),
            "sector": insights.get("sector", "General"),
            "health_score": insights.get("financial_health_score", 5.0),
            "sentiment": insights.get("sentiment_score", 0.0),
        }
        comparison["reports"].append(report_data)
        comparison["health_scores"].append({"company": company, "score": insights.get("financial_health_score", 5.0)})

        for metric in insights.get("key_metrics", []):
            name = metric.get("name", "").strip().lower()
            if name not in all_metric_names:
                all_metric_names[name] = {"name": metric.get("name", ""), "values": {}}
            all_metric_names[name]["values"][company] = {
                "value": metric.get("value", "N/A"),
                "trend": metric.get("trend", "N/A"),
            }

    for mname, mdata in all_metric_names.items():
        if len(mdata["values"]) >= 2:
            comparison["common_metrics"].append(mdata)

    return comparison


def _is_comparison_cache_valid(cached: dict, current_ready_reports: list) -> bool:
    """Check if cached AI comparison matches current set of ready reports."""
    if not cached or not isinstance(cached, dict):
        return False
    meta = cached.get("_meta", {})
    current_ids = sorted([r.id for r in current_ready_reports])
    cached_ids = sorted(meta.get("report_ids", []))
    return (
        meta.get("ready_count") == len(current_ready_reports)
        and cached_ids == current_ids
    )


@router.get("/compare_ai/{set_id}")
async def compare_reports_ai(set_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return an AI generated comparison for all ready reports in a set.

    The comparison is cached in ReportSet.ai_comparison but invalidated
    automatically when the set of ready reports changes (e.g., a new report
    finishes processing).
    """
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Report set not found.")

    # Always fetch current ready reports first — this is our cache key
    reports = db.query(Report).filter(
        Report.id.in_(report_set.report_ids),
        Report.status == "ready",
    ).all()

    # BUG FIX #2: Validate cache against current ready report set.
    # If a new report became ready since last cache, regenerate.
    if report_set.ai_comparison and _is_comparison_cache_valid(report_set.ai_comparison, reports):
        logger.info(f"Returning cached AI comparison for set {set_id} ({len(reports)} reports)")
        # Strip internal _meta before returning
        clean = {k: v for k, v in report_set.ai_comparison.items() if not k.startswith("_")}
        return clean

    if len(reports) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 ready reports to compare.")

    reports_data = []
    for r in reports:
        insights = r.insights or {}
        reports_data.append({
            "company_name": insights.get("company_name", r.filename),
            "financial_health_score": insights.get("financial_health_score"),
            "key_metrics": insights.get("key_metrics", []),
            "key_ratios": insights.get("key_ratios", []),
            "revenue_breakdown": insights.get("revenue_breakdown", []),
            "risk_analysis": insights.get("risk_analysis", []),
            "strategic_initiatives": insights.get("strategic_initiatives", []),
        })

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured.")

    client = groq.AsyncGroq(api_key=GROQ_API_KEY)
    schema_str = json.dumps(AIComparison.model_json_schema())

    prompt = (
        "You are an elite Financial Analyst. Compare the following companies based on their extracted financial reports.\n"
        "Return ONLY a single raw JSON object that strictly adheres to this schema:\n"
        f"{schema_str}\n\n"
        "--- COMPANIES DATA ---\n"
        f"{json.dumps(reports_data)}\n"
        "--- END DATA ---\n"
    )

    try:
        completion = await client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " You ONLY output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=4000,
            top_p=1,
        )

        resp_content = completion.choices[0].message.content

        raw = resp_content.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end+1]
            else:
                json_str = raw

        parsed = json_repair.loads(json_str)
        validated = AIComparison.model_validate(parsed)

        dumped = validated.model_dump()
        # Store cache metadata alongside the comparison data
        dumped["_meta"] = {
            "ready_count": len(reports),
            "report_ids": [r.id for r in reports],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        report_set.ai_comparison = dumped
        db.commit()

        # Return without internal metadata
        clean = {k: v for k, v in dumped.items() if not k.startswith("_")}
        return clean

    except Exception as e:
        logger.error(f"AI comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate AI comparison: {str(e)}")
