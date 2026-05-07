"""
Progress tracking for long-running operations (PDF processing, extraction, etc).

Production: Uses Redis if REDIS_URL is configured (survives restarts, multi-instance).
Local dev: Falls back to in-memory dict.

All progress entries have a 24-hour TTL.

Critical guarantees:
- Progress is STRICTLY MONOTONIC (never decreases)
- completed_steps never decreases
- Terminal states (ready/error) are IMMUTABLE — no further updates allowed
- DB is the source of truth for terminal states
"""
import json
import os
from typing import Dict, Optional
from datetime import datetime, timezone

from config import REDIS_URL
from logger import get_logger

logger = get_logger("progress_tracker")

# ── Redis or In-Memory backend ──────────────────────────────────────────────

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis as redis_lib
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)
        _redis_client.ping()
        logger.info("Progress tracker connected to Redis")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), falling back to in-memory progress")
        _redis_client = None
        return None


# In-memory fallback
_progress_store: Dict[str, Dict] = {}

_PROGRESS_TTL_SECONDS = 86400  # 24 hours


def _key(report_id: str) -> str:
    return f"finora:progress:{report_id}"


def _load_state(report_id: str) -> Optional[Dict]:
    r = _get_redis()
    if r:
        try:
            data = r.get(_key(report_id))
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis read error: {e}")
    return _progress_store.get(report_id)


def _save_state(report_id: str, state: Dict):
    r = _get_redis()
    if r:
        try:
            r.setex(_key(report_id), _PROGRESS_TTL_SECONDS, json.dumps(state))
            return
        except Exception as e:
            logger.warning(f"Redis write error: {e}")
    _progress_store[report_id] = state


def _delete_state(report_id: str):
    r = _get_redis()
    if r:
        try:
            r.delete(_key(report_id))
        except Exception:
            pass
    _progress_store.pop(report_id, None)


# ── Progress Tracker ────────────────────────────────────────────────────────

class ProgressTracker:
    """Persistent progress tracker for report processing.

    Guarantees:
    - progress NEVER decreases
    - completed_steps NEVER decreases
    - Terminal states (ready/error) reject ALL further updates
    """

    # Granular pipeline steps for frontend
    PIPELINE_STEPS = [
        "Upload received",
        "PDF preprocessing",
        "Financial page detection",
        "Table extraction",
        "Financial prioritization",
        "LLM map analysis",
        "Reduce synthesis",
        "Vectorization",
        "Dashboard generation",
        "Final completion",
    ]

    # Stage ordering for monotonic step detection
    STAGE_ORDER = {
        "queued": 0,
        "processing": 1,
        "extraction": 2,
        "parallel_processing": 3,
        "llm_map": 4,
        "vectorization": 5,
        "llm_reduce": 6,
        "finalization": 7,
        "complete": 8,
        "error": 9,
    }

    def __init__(self, report_id: str):
        self.report_id = report_id
        self._initialize()

    def _initialize(self):
        state = _load_state(self.report_id)
        if state is None:
            state = {
                "status": "queued",
                "progress": 2,
                "current_step": "Upload received",
                "total_steps": len(self.PIPELINE_STEPS),
                "completed_steps": 0,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "pipeline_stage": "queued",
                "_terminal": False,  # Internal flag: true when ready/error
            }
            _save_state(self.report_id, state)

    def _is_terminal(self, state: dict) -> bool:
        """Check if this report has reached a terminal state."""
        return state.get("_terminal", False) or state.get("status") in ("ready", "error")

    def start(self):
        state = _load_state(self.report_id)
        if not state or self._is_terminal(state):
            return
        state["status"] = "processing"
        state["started_at"] = datetime.now(timezone.utc).isoformat()
        state["pipeline_stage"] = "processing"
        _save_state(self.report_id, state)

    def set_state(
        self,
        *,
        current_step: str | None = None,
        progress: int | None = None,
        completed_steps: int | None = None,
        status: str | None = None,
        pipeline_stage: str | None = None,
    ):
        """Update progress state. ALL values are monotonic — never decrease."""
        state = _load_state(self.report_id)
        if not state:
            return

        # Terminal states are immutable — reject all updates
        if self._is_terminal(state):
            # Only allow re-setting the same terminal status (idempotent)
            if status in ("ready", "error"):
                state["status"] = status
                _save_state(self.report_id, state)
            return

        # Monotonic progress: never allow decrease
        if progress is not None:
            new_progress = max(0, min(100, int(progress)))
            if new_progress < state.get("progress", 0):
                # Silently drop regressive progress updates
                progress = None
            else:
                state["progress"] = new_progress

        # Monotonic completed_steps: never allow decrease
        if completed_steps is not None:
            new_steps = max(0, completed_steps)
            if new_steps < state.get("completed_steps", 0):
                completed_steps = None
            else:
                state["completed_steps"] = new_steps

        # Only update step text if progress is advancing (or no progress provided)
        if current_step is not None and progress is not None:
            state["current_step"] = current_step
            # Auto-advance completed_steps if step matches pipeline
            for i, step in enumerate(self.PIPELINE_STEPS):
                if step.lower() in current_step.lower():
                    state["completed_steps"] = max(state.get("completed_steps", 0), i)
                    break
        elif current_step is not None:
            state["current_step"] = current_step
            for i, step in enumerate(self.PIPELINE_STEPS):
                if step.lower() in current_step.lower():
                    state["completed_steps"] = max(state.get("completed_steps", 0), i)
                    break

        # Monotonic pipeline_stage: never move backward
        if pipeline_stage is not None:
            current_stage_order = self.STAGE_ORDER.get(state.get("pipeline_stage", "queued"), 0)
            new_stage_order = self.STAGE_ORDER.get(pipeline_stage, 0)
            if new_stage_order >= current_stage_order:
                state["pipeline_stage"] = pipeline_stage

        if status is not None:
            state["status"] = status

        _save_state(self.report_id, state)

    def update_step(self, step_index: int, step_name: str):
        state = _load_state(self.report_id)
        if not state or self._is_terminal(state):
            return
        new_progress = int((step_index / max(1, state["total_steps"])) * 100)
        if new_progress < state.get("progress", 0):
            return
        state["completed_steps"] = max(0, step_index - 1)
        state["current_step"] = step_name
        state["progress"] = new_progress
        _save_state(self.report_id, state)

    def complete(self):
        state = _load_state(self.report_id)
        if not state:
            return
        state["status"] = "ready"
        state["progress"] = 100
        state["completed_steps"] = state.get("total_steps", 10)
        state["current_step"] = "Final completion"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["pipeline_stage"] = "complete"
        state["_terminal"] = True
        _save_state(self.report_id, state)

    def error(self, error_msg: str):
        state = _load_state(self.report_id)
        if not state:
            return
        # Never overwrite a terminal state with error
        if self._is_terminal(state):
            return
        state["status"] = "error"
        state["error"] = error_msg[:500]
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["pipeline_stage"] = "error"
        state["_terminal"] = True
        _save_state(self.report_id, state)

    def get_state(self) -> dict:
        state = _load_state(self.report_id)
        return dict(state) if state else {}


def get_progress(report_id: str) -> Optional[dict]:
    state = _load_state(report_id)
    return dict(state) if state else None


def clear_progress(report_id: str):
    _delete_state(report_id)
