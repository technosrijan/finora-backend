from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# ── Report Schemas ────────────────────────────────────────────────────────

class ReportRecordSchema(BaseModel):
    id: str
    filename: str
    status: str
    insights: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    token_usage_input: Optional[int] = 0
    token_usage_output: Optional[int] = 0

class ReportSetSchema(BaseModel):
    id: str
    reports: List[ReportRecordSchema]
    created_at: str

class HistoryItemSchema(BaseModel):
    id: str
    created_at: str
    filenames: List[str]
    ready_count: int
    total_count: int

# ── Chat Schemas ──────────────────────────────────────────────────────────

class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatTurnInput(BaseModel):
    reportSetId: str
    userMessage: str
    assistantMessage: str

class ChatSessionInput(BaseModel):
    reportSetId: str
    messages: List[ChatMessageInput]
    context: Optional[str] = None
