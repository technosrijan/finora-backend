"""
Chat router — streaming AI chat with SSE, auth-protected.
"""
import os
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import groq
import asyncio

from database import SessionLocal, get_db, ReportSet, Report, ChatMessage, User, UsageRecord
from models import ChatSessionInput, ChatTurnInput
from services.rag import retrieve_relevant_context
from routers.auth import get_current_user
from config import GROQ_API_KEY, AI_MODEL
from logger import get_logger

logger = get_logger("chat")
router = APIRouter()


@router.post("/messages")
def get_chat_messages(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    set_id = payload.get("reportSetId")
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Not found.")

    messages = db.query(ChatMessage).filter(
        ChatMessage.report_set_id == set_id,
        ChatMessage.user_id == user.id,
    ).order_by(ChatMessage.created_at.asc()).limit(500).all()

    return {
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else "",
            }
            for m in messages
        ]
    }


@router.post("/turn")
def save_chat_turn(
    payload: ChatTurnInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report_set = db.query(ReportSet).filter(
        ReportSet.id == payload.reportSetId, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Not found.")

    user_msg = ChatMessage(
        report_set_id=payload.reportSetId, user_id=user.id,
        role="user", content=payload.userMessage,
    )
    asst_msg = ChatMessage(
        report_set_id=payload.reportSetId, user_id=user.id,
        role="assistant", content=payload.assistantMessage,
    )
    db.add(user_msg)
    db.add(asst_msg)
    db.commit()
    return {"ok": True}


@router.post("/clear")
def clear_chat_messages(
    payload: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    set_id = payload.get("reportSetId")
    report_set = db.query(ReportSet).filter(
        ReportSet.id == set_id, ReportSet.user_id == user.id
    ).first()
    if not report_set:
        raise HTTPException(status_code=404, detail="Not found.")

    db.query(ChatMessage).filter(
        ChatMessage.report_set_id == set_id,
        ChatMessage.user_id == user.id,
    ).delete()
    db.commit()
    return {"ok": True}


def _build_dashboard_context(reports: list[Report]) -> str:
    """Build structured summary of dashboard insights for the chat system prompt."""
    parts = []
    for r in reports:
        if r.status != "ready" or not r.insights:
            continue
        ins = r.insights
        company = ins.get("company_name", r.filename)
        period = ins.get("reporting_period", "N/A")
        sector = ins.get("sector", "N/A")
        health = ins.get("financial_health_score", "N/A")
        sentiment = ins.get("sentiment_score", "N/A")

        block = f"### {company} ({period})\n"
        block += f"Sector: {sector} | Health Score: {health}/10 | Sentiment: {sentiment}\n"
        block += f"Executive Summary: {ins.get('executive_summary', 'N/A')}\n"

        metrics = ins.get("key_metrics", [])
        if metrics:
            block += "\nKey Metrics:\n"
            for m in metrics[:15]:
                block += f"  - {m.get('name')}: {m.get('value')} ({m.get('trend')}) — {m.get('context', '')}\n"

        ratios = ins.get("key_ratios", [])
        if ratios:
            block += "\nKey Ratios:\n"
            for ratio in ratios:
                block += f"  - {ratio.get('name')}: {ratio.get('value')} [{ratio.get('assessment', 'neutral')}] — {ratio.get('context', '')}\n"

        segments = ins.get("revenue_breakdown", [])
        if segments:
            block += "\nRevenue Breakdown:\n"
            for seg in segments:
                block += f"  - {seg.get('segment')}: {seg.get('value')}% ({seg.get('amount', 'N/A')})\n"

        risks = ins.get("risk_analysis", [])
        if risks:
            block += "\nKey Risks:\n"
            for risk in risks[:8]:
                block += f"  - {risk}\n"

        strategies = ins.get("strategic_initiatives", [])
        if strategies:
            block += "\nStrategic Initiatives:\n"
            for s in strategies[:8]:
                block += f"  - {s}\n"

        parts.append(block)

    if not parts:
        return ""
    return "\n--- DASHBOARD INSIGHTS ---\n" + "\n---\n".join(parts) + "\n--- END ---"


@router.post("/session")
async def chat_session(payload: ChatSessionInput, user: User = Depends(get_current_user)):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="AI key not configured.")

    client = groq.AsyncGroq(api_key=GROQ_API_KEY)

    # Extract user query
    user_query = ""
    for m in reversed(payload.messages):
        if m.role == "user":
            user_query = m.content
            break

    # Retrieve context
    db = SessionLocal()
    report_set = db.query(ReportSet).filter(ReportSet.id == payload.reportSetId).first()
    context_parts = []
    dashboard_context = ""

    if report_set:
        all_reports = db.query(Report).filter(Report.id.in_(report_set.report_ids)).all()
        dashboard_context = _build_dashboard_context(all_reports)

        if user_query:
            for rid in report_set.report_ids:
                try:
                    chunk_str = retrieve_relevant_context(rid, user_query, k=8)
                    if chunk_str:
                        context_parts.append(chunk_str)
                except Exception as e:
                    logger.warning(f"RAG retrieval failed for {rid}: {e}")
    db.close()

    rag_context = ""
    if context_parts:
        combined = "\n\n...[NEXT REPORT]...\n\n".join(context_parts)
        rag_context = f"\n\n--- DOCUMENT CONTEXT ---\n{combined}\n--- END ---"

    system_prompt = (
        "You are Finora, a senior financial analyst assistant. You have access to:\n"
        "1. DASHBOARD INSIGHTS: Structured data from uploaded annual reports.\n"
        "2. DOCUMENT CONTEXT: Raw text chunks from the original PDFs.\n\n"
        "Answer questions using ONLY these contexts. If the answer is not available, say so.\n"
        "Use tables for comparisons. Bold key numbers. Format with markdown.\n"
        f"{dashboard_context}{rag_context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in payload.messages:
        messages.append({"role": m.role, "content": m.content})

    async def generate():
        try:
            stream = await client.chat.completions.create(
                model=AI_MODEL,
                messages=messages,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    payload_data = {"choices": [{"delta": {"content": chunk.choices[0].delta.content}}]}
                    yield f"data: {json.dumps(payload_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield f"data: {json.dumps({'error': 'Chat generation failed. Please try again.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
