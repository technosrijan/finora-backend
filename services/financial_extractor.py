"""
Intelligent Financial PDF Extraction Engine
============================================
Production-grade pipeline that extracts, scores, and prioritizes financial content
from PDFs before any downstream LLM or RAG processing.

Optimized for speed:
- Parallel text extraction (PyMuPDF)
- Selective table extraction (only financial-candidate pages)
- Fast heuristic pre-filtering before expensive table parsing
- Designed for 600-page PDFs, 3 concurrent uploads.

Goals:
- Maximize financial signal density
- Minimize downstream token load
- Preserve semantic structure (labels, units, dates, hierarchy)
- Intelligently filter low-value / repetitive content
- Dynamically adapt to document size
"""
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pymupdf
import pdfplumber
from logger import get_logger

logger = get_logger("financial_extractor")

# ── Financial Signal Lexicons ───────────────────────────────────────────────

FINANCIAL_KEYWORDS = {
    # High-value statement terms
    "balance sheet", "income statement", "cash flow statement", "statement of cash flows",
    "statement of financial position", "statement of comprehensive income",
    "consolidated financial statements", "notes to the financial statements",
    "auditor", "audit opinion", "gaap", "ifrs",
    # Metrics & KPIs
    "revenue", "total revenue", "net revenue", "gross profit", "operating profit",
    "net income", "net profit", "earnings", "ebitda", "ebit", "ebt",
    "earnings per share", "eps", "diluted eps", "basic eps",
    "free cash flow", "operating cash flow", "investing cash flow", "financing cash flow",
    "return on equity", "roe", "return on assets", "roa", "return on capital", "roce",
    "debt to equity", "debt ratio", "current ratio", "quick ratio", "interest coverage",
    "gross margin", "operating margin", "net margin", "ebitda margin", "profit margin",
    "working capital", "total assets", "total liabilities", "shareholders equity",
    "book value", "tangible book value", "goodwill", "intangible assets",
    "accounts receivable", "inventory", "accounts payable", "accrued liabilities",
    "capital expenditure", "capex", "depreciation", "amortization",
    "dividend", "dividend yield", "payout ratio", "share buyback", "repurchase",
    "wacc", "cost of capital", "beta", "ev/ebitda", "p/e ratio", "price to book",
    # Growth & Trend terms
    "yoy", "year over year", "year-on-year", "qoq", "quarter over quarter",
    "growth rate", "cagr", "organic growth", "constant currency",
    # Segment & Geography
    "segment", "business segment", "operating segment", "reportable segment",
    "geographic", "region", "north america", "europe", "asia pacific", "emerging markets",
    "product revenue", "service revenue", "subscription revenue", "recurring revenue",
    # Risk & Governance
    "risk factor", "risk management", "credit risk", "market risk", "liquidity risk",
    "operational risk", "cybersecurity risk", "regulatory risk", "climate risk",
    "internal control", "sarbanes-oxley", "sox", "coso",
    # MD&A & Outlook
    "management discussion", "md&a", "operating results", "financial condition",
    "liquidity", "solvency", "going concern", "outlook", "guidance", "forecast",
    "forward-looking", "expected", "projected", "target", "commitment",
    # Units & scale
    "million", "billion", "trillion", "thousand", "$", "usd", "eur", "gbp", "jpy",
    "%", "percent", "basis points", "bps",
}

# ── Table-specific keywords (shorter, cell-friendly) ────────────────────────
# These are more likely to match inside individual table cells
TABLE_FINANCIAL_KEYWORDS = {
    "revenue", "total revenue", "net revenue", "gross revenue",
    "income", "net income", "gross income", "operating income",
    "profit", "gross profit", "operating profit", "net profit",
    "loss", "net loss", "operating loss",
    "ebitda", "ebit", "ebt", "eps",
    "assets", "total assets", "current assets", "fixed assets",
    "liabilities", "total liabilities", "current liabilities",
    "equity", "shareholders equity", "total equity",
    "debt", "total debt", "net debt", "long-term debt",
    "cash", "cash flow", "free cash flow", "operating cash flow",
    "balance", "balance sheet",
    "statement", "income statement", "financial statement",
    "million", "billion", "thousand",
    "margin", "gross margin", "operating margin", "profit margin",
    "growth", "yoy", "increase", "decrease",
    "segment", "category", "division", "business",
    "subscription", "recurring", "hardware",
    "customer", "residential", "business", "wholesale",
    "chf", "usd", "eur", "gbp", "¥", "€", "$", "£",
}

STATEMENT_SECTION_MARKERS = {
    "balance sheet", "income statement", "cash flow", "statement of financial position",
    "statement of comprehensive income", "consolidated financial statements",
}

NOTES_SECTION_MARKERS = {
    "notes to the financial statements", "significant accounting policies",
    "basis of preparation", "summary of significant", "note ", "notes ",
}

MDA_SECTION_MARKERS = {
    "management discussion", "md&a", "management's discussion",
    "operating and financial review", "directors report",
}

RISK_SECTION_MARKERS = {
    "risk factor", "risk management", "principal risks", "risk and uncertainties",
    "risk overview", "risk appetite",
}

SEGMENT_SECTION_MARKERS = {
    "segment reporting", "business segment", "operating segment",
    "geographic information", "revenue by segment", "revenue by geography",
}

BOILERPLATE_MARKERS = {
    "table of contents", "index", "glossary", "definitions",
    "forward-looking statements", "cautionary statement", "safe harbor",
    "disclaimer", "legal proceedings", "corporate information",
    "board of directors", "corporate governance", "shareholder information",
    "contact us", "investor relations", "annual general meeting",
    "this page intentionally left blank",
}

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    page_num: int
    table_index: int
    markdown: str
    row_count: int
    col_count: int
    has_numeric_data: bool
    financial_score: float = 0.0


@dataclass
class PageAnalysis:
    page_num: int
    raw_text: str
    cleaned_text: str
    tables: List[ExtractedTable]
    word_count: int
    numeric_count: int
    currency_count: int
    percent_count: int
    financial_keyword_hits: int
    financial_score: float
    category: str  # statement | notes | mda | risk | segment | general | boilerplate
    is_high_value: bool = False


@dataclass
class FinancialExtractionResult:
    total_pages: int
    pages: List[PageAnalysis]
    all_tables: List[ExtractedTable]
    # Prioritized content ready for downstream
    llm_context: str = ""          # Optimized for map-reduce LLM pipeline
    rag_context: str = ""          # Full content for vectorization
    extracted_tables_markdown: str = ""  # All tables as markdown
    financial_page_count: int = 0
    skipped_page_count: int = 0
    # Metadata for progress/observability
    top_categories: Dict[str, int] = field(default_factory=dict)


# ── Text Cleaning ───────────────────────────────────────────────────────────

_EXTRA_WHITESPACE_RE = re.compile(r"\n{4,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _clean_page_text(raw_text: str, page_num: int) -> str:
    """
    Aggressively clean page text:
    - Remove repeated headers/footers/page numbers
    - Normalize whitespace
    - Remove URLs
    - Keep semantic structure
    """
    if not raw_text:
        return ""

    lines = raw_text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        # Skip obvious header/footer patterns
        if re.match(r"^\s*\d+\s*$", stripped) and len(stripped) < 5:
            continue
        if "www." in stripped.lower() and len(stripped) < 80:
            continue
        if stripped.lower().startswith("page ") and len(stripped) < 15:
            continue
        if "all amounts in" in stripped.lower() and len(stripped) < 100:
            cleaned_lines.append(f"[CURRENCY NOTE: {stripped}]")
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _URL_RE.sub("", text)
    text = _EXTRA_WHITESPACE_RE.sub("\n\n\n", text)
    return text.strip()


# ── Pattern Counters ────────────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"[\$€£¥]\s*[\d,]+(?:\.\d+)?\s*[BMKmbk]?|\b[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|mn|bn|m|b)\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"[\d,]+(?:\.\d+)?\s*%|\b\d+\s*basis points\b|\b\d+\s*bps\b", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\b[\d,]+(?:\.\d+)?\b")


# ── Fast Pre-Scoring (before table extraction) ──────────────────────────────

def _fast_prescore(text: str) -> Tuple[float, str]:
    """
    Very fast heuristic score to determine if a page is worth table-extracting.
    Returns (score, category_hint).
    """
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    if word_count < 15:
        return -10.0, "boilerplate"

    # Quick keyword scan
    keyword_hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in text_lower)
    currency_count = len(_CURRENCY_RE.findall(text))
    percent_count = len(_PERCENT_RE.findall(text))

    # Detect category
    category = "general"
    cat_scores = {
        "statement": sum(1 for m in STATEMENT_SECTION_MARKERS if m in text_lower),
        "notes": sum(1 for m in NOTES_SECTION_MARKERS if m in text_lower),
        "mda": sum(1 for m in MDA_SECTION_MARKERS if m in text_lower),
        "risk": sum(1 for m in RISK_SECTION_MARKERS if m in text_lower),
        "segment": sum(1 for m in SEGMENT_SECTION_MARKERS if m in text_lower),
        "boilerplate": sum(1 for m in BOILERPLATE_MARKERS if m in text_lower),
    }
    for cat, score in cat_scores.items():
        if score > 0 and (category == "general" or score > cat_scores.get(category, 0)):
            category = cat

    # Fast score
    score = keyword_hits * 1.5 + currency_count * 2.0 + percent_count * 1.5
    if category == "boilerplate" and cat_scores["boilerplate"] >= 2:
        score = -20.0
    elif category == "statement":
        score += 20.0
    elif category in ("notes", "mda", "segment"):
        score += 10.0

    return score, category


# ── Full Page Scoring (after table extraction) ──────────────────────────────

def _score_and_categorize_page(cleaned_text: str, tables: List[ExtractedTable]) -> Tuple[float, str]:
    """Score a page for financial signal density and assign a category."""
    text_lower = cleaned_text.lower()
    words = text_lower.split()
    word_count = len(words)

    if word_count < 10:
        return 0.0, "boilerplate"

    currency_count = len(_CURRENCY_RE.findall(cleaned_text))
    percent_count = len(_PERCENT_RE.findall(cleaned_text))
    numeric_count = len(_NUMERIC_RE.findall(cleaned_text))
    keyword_hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in text_lower)

    # Detect category
    category = "general"
    max_cat_score = 0
    cat_scores = {
        "statement": sum(1 for m in STATEMENT_SECTION_MARKERS if m in text_lower),
        "notes": sum(1 for m in NOTES_SECTION_MARKERS if m in text_lower),
        "mda": sum(1 for m in MDA_SECTION_MARKERS if m in text_lower),
        "risk": sum(1 for m in RISK_SECTION_MARKERS if m in text_lower),
        "segment": sum(1 for m in SEGMENT_SECTION_MARKERS if m in text_lower),
        "boilerplate": sum(1 for m in BOILERPLATE_MARKERS if m in text_lower),
    }
    for cat, score in cat_scores.items():
        if score > max_cat_score:
            max_cat_score = score
            category = cat

    if category == "boilerplate" and max_cat_score >= 2:
        return 0.0, "boilerplate"

    # Base score
    score = 0.0
    score += keyword_hits * 2.5
    score += currency_count * 3.0
    score += percent_count * 2.0
    score += numeric_count * 0.3
    score += len(tables) * 15.0

    category_bonus = {
        "statement": 35.0, "notes": 20.0, "mda": 25.0,
        "risk": 12.0, "segment": 18.0, "general": 0.0, "boilerplate": -100.0,
    }
    score += category_bonus.get(category, 0.0)

    for t in tables:
        if t.has_numeric_data:
            score += 10.0

    if word_count < 50:
        score *= 0.5
    if word_count < 30 and numeric_count < 3:
        score = 0.0
        category = "boilerplate"

    # Density normalization
    if word_count > 500:
        density_factor = 500 / word_count
        score = score * 0.7 + score * 0.3 * density_factor

    return max(0.0, score), category


# ── Table Extraction ────────────────────────────────────────────────────────

def _extract_tables_from_page(page, page_num: int) -> List[ExtractedTable]:
    """Extract tables from a single pdfplumber page object."""
    tables = []
    try:
        page_tables = page.extract_tables()
        if not page_tables:
            return tables

        for tidx, table in enumerate(page_tables):
            if not table or not table[0]:
                continue

            md_lines = []
            for ridx, row in enumerate(table):
                cleaned = [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
                md_lines.append("| " + " | ".join(cleaned) + " |")
                if ridx == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cleaned)) + " |")

            markdown = "\n".join(md_lines)
            flat = " ".join(str(c) for row in table for c in row if c)
            has_numeric = bool(_CURRENCY_RE.search(flat) or _PERCENT_RE.search(flat) or _NUMERIC_RE.search(flat))

            tscore = 0.0
            flat_lower = flat.lower()
            # Use table-specific keywords (shorter, cell-friendly)
            for kw in TABLE_FINANCIAL_KEYWORDS:
                if kw in flat_lower:
                    tscore += 2.0
            # Extra bonus for critical financial statement terms
            critical_terms = {"total revenue", "revenue", "net income", "net profit", "net loss",
                              "balance sheet", "income statement", "cash flow", "total assets",
                              "total liabilities", "shareholders equity", "ebitda", "eps"}
            for term in critical_terms:
                if term in flat_lower:
                    tscore += 8.0
            if has_numeric:
                tscore += 15.0
            # Bonus for large tables (more likely to be financial statements)
            if len(table) >= 5 and len(table[0]) >= 3:
                tscore += 10.0

            tables.append(ExtractedTable(
                page_num=page_num,
                table_index=tidx,
                markdown=markdown,
                row_count=len(table),
                col_count=len(table[0]) if table else 0,
                has_numeric_data=has_numeric,
                financial_score=tscore,
            ))
    except Exception:
        pass
    return tables


def extract_tables_structured(pdf_path: str, candidate_pages: Optional[List[int]] = None) -> Dict[int, List[ExtractedTable]]:
    """
    Extract tables with rich metadata per page.
    Opens the PDF once and iterates only candidate pages for speed.
    """
    tables_by_page: Dict[int, List[ExtractedTable]] = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if candidate_pages is None:
                candidate_pages = list(range(len(pdf.pages)))

            for pn in candidate_pages:
                if pn >= len(pdf.pages):
                    continue
                tables = _extract_tables_from_page(pdf.pages[pn], pn)
                if tables:
                    tables_by_page[pn] = tables
    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}")

    return tables_by_page


# ── Context Builders ────────────────────────────────────────────────────────

def _build_llm_context(pages: List[PageAnalysis], max_chars: int = 150_000) -> str:
    """Build LLM-optimized context from prioritized pages."""
    ordered = sorted(pages, key=lambda p: p.page_num)
    parts = []
    total_chars = 0

    for p in ordered:
        if p.category == "boilerplate" or p.financial_score < 3.0:
            continue

        if p.financial_score >= 50:
            max_page_chars = 5000
            prefix = f"\n--- PAGE {p.page_num + 1} [HIGH-VALUE {p.category.upper()}] ---\n"
        elif p.financial_score >= 20:
            max_page_chars = 3000
            prefix = f"\n--- PAGE {p.page_num + 1} [{p.category.upper()}] ---\n"
        elif p.financial_score >= 8:
            max_page_chars = 1200
            prefix = f"\n--- PAGE {p.page_num + 1} [{p.category.upper()}] ---\n"
        else:
            max_page_chars = 400
            prefix = f"\n--- PAGE {p.page_num + 1} [{p.category.upper()}] ---\n"

        text = p.cleaned_text[:max_page_chars]
        if len(p.cleaned_text) > max_page_chars:
            text = text.rsplit(" ", 1)[0] + " ...[truncated]"

        page_text = prefix + text

        for t in p.tables:
            if t.financial_score >= 8.0 or t.has_numeric_data:
                page_text += f"\n[TABLE {t.table_index + 1} from Page {p.page_num + 1}]\n" + t.markdown + "\n"

        if total_chars + len(page_text) > max_chars:
            remaining = max_chars - total_chars - len(prefix) - 50
            if remaining > 200:
                page_text = prefix + p.cleaned_text[:remaining].rsplit(" ", 1)[0] + " ...[truncated for length]\n"
                parts.append(page_text)
            break

        parts.append(page_text)
        total_chars += len(page_text)

    return "\n".join(parts)


def _build_rag_context(pages: List[PageAnalysis], max_chars: int = 300_000) -> str:
    """Build RAG-optimized context with MORE content than LLM context."""
    ordered = sorted(pages, key=lambda p: p.page_num)
    parts = []
    total_chars = 0

    for p in ordered:
        if p.category == "boilerplate" or p.financial_score < 2.0:
            continue

        if p.financial_score >= 30:
            max_page_chars = 6000
        elif p.financial_score >= 10:
            max_page_chars = 3500
        else:
            max_page_chars = 1500

        text = p.cleaned_text[:max_page_chars]
        if len(p.cleaned_text) > max_page_chars:
            text = text.rsplit(" ", 1)[0] + " ...[truncated]"

        prefix = f"\n--- PAGE {p.page_num + 1} [{p.category.upper()} | score={p.financial_score:.1f}] ---\n"
        page_text = prefix + text

        for t in p.tables:
            if t.financial_score >= 5.0:
                page_text += f"\n[TABLE {t.table_index + 1}]\n" + t.markdown + "\n"

        if total_chars + len(page_text) > max_chars:
            remaining = max_chars - total_chars - len(prefix) - 50
            if remaining > 300:
                parts.append(prefix + p.cleaned_text[:remaining].rsplit(" ", 1)[0] + " ...[truncated]\n")
            break

        parts.append(page_text)
        total_chars += len(page_text)

    return "\n".join(parts)


def _build_tables_markdown(pages: List[PageAnalysis]) -> str:
    """Build consolidated markdown of all financial tables.
    Tables are sorted by financial score (highest first) so that
    critical tables (revenue, P&L, balance sheet) survive truncation
    in the reduce phase.
    """
    all_tables = []
    for p in pages:
        for t in p.tables:
            if t.financial_score >= 5.0 or t.has_numeric_data:
                all_tables.append(t)

    # Sort by financial score descending — critical tables first
    all_tables.sort(key=lambda t: t.financial_score, reverse=True)

    parts = []
    for t in all_tables:
        parts.append(f"### Table from Page {t.page_num + 1} (score: {t.financial_score:.0f})\n{t.markdown}\n")
    return "\n".join(parts)


# ── Main Entry Point ────────────────────────────────────────────────────────

def analyze_pdf(pdf_path: str) -> FinancialExtractionResult:
    """
    Main entry point: analyze a PDF and return prioritized financial extraction.

    Optimized pipeline:
    1. Extract all text with PyMuPDF (fast)
    2. Fast pre-score to identify table-candidate pages
    3. Parallel table extraction only on candidate pages
    4. Full scoring with table data
    5. Build prioritized contexts
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import time
    t0 = time.time()
    logger.info(f"Starting financial extraction for {os.path.basename(pdf_path)}")

    # Step 1: Extract all text and pre-score (fast)
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    raw_pages = []
    prescores = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        raw_pages.append((page_num, text))
        cleaned_quick = _clean_page_text(text, page_num)
        pscore, pcat = _fast_prescore(cleaned_quick)
        prescores.append((page_num, pscore, pcat, cleaned_quick))
    doc.close()

    # Step 2: Identify table candidate pages
    # Only run pdfplumber on pages with meaningful financial signal.
    # Raise threshold to skip narrative-heavy pages that rarely have tables.
    candidate_pages = []
    for i, (pn, pscore, pcat, _) in enumerate(prescores):
        if pscore > 3 or pcat in ("statement", "notes", "segment"):
            candidate_pages.append(pn)
        # Include neighbors of high-signal pages (tables often span 2 pages)
        elif i > 0 and prescores[i-1][1] > 8:
            candidate_pages.append(pn)
        elif i < len(prescores) - 1 and prescores[i+1][1] > 8:
            candidate_pages.append(pn)

    candidate_pages = sorted(set(candidate_pages))
    logger.info(f"Table candidate pages: {len(candidate_pages)}/{total_pages}")

    # Step 3: Table extraction on filtered candidates
    tables_by_page = extract_tables_structured(pdf_path, candidate_pages)

    # Step 4: Full analysis
    analyzed_pages: List[PageAnalysis] = []
    for page_num, raw_text in raw_pages:
        tables = tables_by_page.get(page_num, [])
        cleaned = prescores[page_num][3]  # Reuse cleaned text from pre-score
        score, category = _score_and_categorize_page(cleaned, tables)

        numeric_count = len(_NUMERIC_RE.findall(cleaned))
        currency_count = len(_CURRENCY_RE.findall(cleaned))
        percent_count = len(_PERCENT_RE.findall(cleaned))
        keyword_hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in cleaned.lower())

        pa = PageAnalysis(
            page_num=page_num,
            raw_text=raw_text,
            cleaned_text=cleaned,
            tables=tables,
            word_count=len(cleaned.split()),
            numeric_count=numeric_count,
            currency_count=currency_count,
            percent_count=percent_count,
            financial_keyword_hits=keyword_hits,
            financial_score=score,
            category=category,
            is_high_value=(score >= 30 and category != "boilerplate"),
        )
        analyzed_pages.append(pa)

    # Step 5: Build contexts
    llm_context = _build_llm_context(analyzed_pages)
    rag_context = _build_rag_context(analyzed_pages)
    tables_md = _build_tables_markdown(analyzed_pages)

    financial_pages = [p for p in analyzed_pages if p.financial_score >= 5.0 and p.category != "boilerplate"]
    skipped_pages = [p for p in analyzed_pages if p.category == "boilerplate" or p.financial_score < 3.0]

    top_categories = defaultdict(int)
    for p in analyzed_pages:
        if p.financial_score >= 5.0:
            top_categories[p.category] += 1

    duration = round(time.time() - t0, 2)
    logger.info(
        f"Extraction complete: {total_pages} pages -> {len(financial_pages)} financial pages "
        f"({len(skipped_pages)} skipped) | LLM context: {len(llm_context):,} chars | "
        f"RAG context: {len(rag_context):,} chars | {duration}s"
    )

    all_tables = []
    for p in analyzed_pages:
        all_tables.extend(p.tables)

    return FinancialExtractionResult(
        total_pages=total_pages,
        pages=analyzed_pages,
        all_tables=all_tables,
        llm_context=llm_context,
        rag_context=rag_context,
        extracted_tables_markdown=tables_md,
        financial_page_count=len(financial_pages),
        skipped_page_count=len(skipped_pages),
        top_categories=dict(top_categories),
    )


# ── Helpers for downstream ──────────────────────────────────────────────────

def get_dynamic_map_config(extraction: FinancialExtractionResult) -> dict:
    """
    Determine optimal map-reduce configuration based on extracted content.
    """
    llm_len = len(extraction.llm_context)
    high_value_pages = sum(1 for p in extraction.pages if p.is_high_value)

    base_chunk_size = 18_000

    if llm_len <= 20_000:
        chunk_count = 1
    elif llm_len <= 60_000:
        chunk_count = 2
    elif llm_len <= 120_000:
        chunk_count = 3
    elif llm_len <= 200_000:
        chunk_count = 4
    elif llm_len <= 350_000:
        chunk_count = 5
    else:
        chunk_count = 6

    chunk_count = min(chunk_count, max(1, high_value_pages // 2))
    chunk_count = max(1, min(chunk_count, 8))
    concurrency = min(chunk_count, 4)

    return {
        "chunk_count": chunk_count,
        "chunk_size": base_chunk_size,
        "concurrency": concurrency,
        "llm_context_length": llm_len,
        "high_value_pages": high_value_pages,
    }


def split_into_semantic_chunks(extraction: FinancialExtractionResult, chunk_count: int) -> List[str]:
    """Split LLM context into semantic chunks based on page boundaries."""
    context = extraction.llm_context
    if chunk_count <= 1:
        return [context]

    page_pattern = re.compile(r"\n(--- PAGE \d+ \[[^\]]+\] ---\n)")
    pages_raw = page_pattern.split(context)
    page_headers = page_pattern.findall(context)

    chunks = []
    current_chunk = ""
    target_size = len(context) // chunk_count

    for i, page_text in enumerate(pages_raw):
        header = page_headers[i] if i < len(page_headers) else ""
        piece = header + page_text

        if not current_chunk:
            current_chunk = piece
        elif len(current_chunk) + len(piece) < target_size * 1.3:
            current_chunk += piece
        else:
            chunks.append(current_chunk)
            current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    while len(chunks) > chunk_count and len(chunks) > 1:
        min_idx = 0
        min_len = len(chunks[0]) + len(chunks[1])
        for i in range(1, len(chunks) - 1):
            combined = len(chunks[i]) + len(chunks[i + 1])
            if combined < min_len:
                min_len = combined
                min_idx = i
        chunks[min_idx] = chunks[min_idx] + chunks[min_idx + 1]
        chunks.pop(min_idx + 1)

    return chunks
