"""
RAG service — document chunking, ChromaDB storage, and retrieval.

Financial-aware chunking strategy:
- Preserves table structure (tables get larger chunks)
- Adds rich metadata (page, section, table flag, financial score)
- Uses financial semantic separators
- Optimized for financial Q&A retrieval

Per-user isolation is guaranteed by design:
  - Each report gets its own ChromaDB collection: "report_{report_id}"
  - chat.py only queries collections for report_ids in the user's ReportSet
  - ReportSet and Report rows are always scoped to user_id in the DB

In production (Cloud Run), ChromaDB runs as a separate HTTP server service
(finora-chromadb on Cloud Run) backed by GCS for persistence.
In local dev (CHROMA_HOST not set), falls back to a local PersistentClient.
"""
import os
import re
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHROMA_DB_DIR, CHROMA_HOST, CHROMA_PORT
from logger import get_logger

logger = get_logger("rag")

# ── ChromaDB Client ─────────────────────────────────────────────────────────

def _make_client() -> chromadb.ClientAPI:
    if CHROMA_HOST:
        logger.info(f"ChromaDB: connecting to HTTP server at {CHROMA_HOST}:{CHROMA_PORT}")
        return chromadb.HttpClient(
            host=CHROMA_HOST,
            port=int(CHROMA_PORT),
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        logger.info(f"ChromaDB: using local PersistentClient at {CHROMA_DB_DIR}")
        return chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )


_chroma_client_instance = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client_instance
    if _chroma_client_instance is None:
        _chroma_client_instance = _make_client()
    return _chroma_client_instance


embedding_fn = embedding_functions.DefaultEmbeddingFunction()


# ── Financial Semantic Chunking ─────────────────────────────────────────────

# Separators that respect financial document structure
FINANCIAL_SEPARATORS = [
    "\n--- PAGE ",           # Page boundaries (highest priority)
    "\n### ",                # Section headers
    "\n## ",                 # Larger sections
    "\n--- TABLE",           # Table boundaries
    "\n\n",                  # Paragraph breaks
    "\n",                    # Line breaks
    ". ",                    # Sentence boundaries
    " ",                     # Word boundaries
    "",                      # Character boundaries
]


def chunk_document_financial(text: str, tables_markdown: str = "") -> list[dict]:
    """
    Financial-aware chunking that returns chunks with rich metadata.

    Returns list of dicts: {text, metadata}
    metadata contains: page_num, section_type, is_table, financial_score
    """
    chunks_with_meta = []

    # Split text by page markers first
    page_pattern = re.compile(r"\n(--- PAGE (\d+) \[([^\]]+)\] ---\n)")
    splits = page_pattern.split(text)

    current_page = 0
    current_category = "general"
    current_buffer = ""

    # Process page by page
    i = 0
    while i < len(splits):
        if splits[i].startswith("--- PAGE "):
            # Save previous buffer
            if current_buffer.strip():
                chunks_with_meta.append({
                    "text": current_buffer.strip(),
                    "metadata": {
                        "page_num": current_page,
                        "section_type": current_category,
                        "is_table": False,
                        "financial_score": 0,
                    }
                })
            # Parse header
            match = page_pattern.match("\n" + splits[i])
            if match:
                current_page = int(match.group(2))
                cat_raw = match.group(3).lower()
                # Extract category from bracket text like "HIGH-VALUE STATEMENT"
                for cat in ["statement", "notes", "mda", "risk", "segment", "general", "boilerplate"]:
                    if cat in cat_raw:
                        current_category = cat
                        break
            current_buffer = splits[i + 1] if i + 1 < len(splits) else ""
            i += 2
        else:
            current_buffer += splits[i]
            i += 1

    # Don't forget the last buffer
    if current_buffer.strip():
        chunks_with_meta.append({
            "text": current_buffer.strip(),
            "metadata": {
                "page_num": current_page,
                "section_type": current_category,
                "is_table": False,
                "financial_score": 0,
            }
        })

    # Now apply recursive splitting to each page-level chunk
    # Use different parameters based on content type
    splitter_standard = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=FINANCIAL_SEPARATORS,
    )

    splitter_dense = RecursiveCharacterTextSplitter(
        chunk_size=3500,   # Larger chunks for dense financial content
        chunk_overlap=300,
        length_function=len,
        separators=FINANCIAL_SEPARATORS,
    )

    final_chunks = []
    for chunk_meta in chunks_with_meta:
        meta = chunk_meta["metadata"]
        text = chunk_meta["text"]

        # Choose splitter based on content type
        if meta["section_type"] in ("statement", "notes", "segment"):
            splitter = splitter_dense
            meta["financial_score"] = 8
        elif meta["section_type"] == "mda":
            splitter = splitter_standard
            meta["financial_score"] = 6
        elif meta["section_type"] == "risk":
            splitter = splitter_standard
            meta["financial_score"] = 4
        else:
            splitter = splitter_standard
            meta["financial_score"] = 2

        sub_chunks = splitter.split_text(text)
        for sub in sub_chunks:
            if len(sub.strip()) < 50:
                continue
            final_chunks.append({
                "text": sub.strip(),
                "metadata": {
                    **meta,
                    "char_count": len(sub),
                }
            })

    # Add table chunks separately with high financial score
    if tables_markdown.strip():
        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n### ", "\n\n", "\n", " ", ""],
        )
        table_chunks = table_splitter.split_text(tables_markdown)
        for tc in table_chunks:
            if len(tc.strip()) < 50:
                continue
            # Extract page number from "### Table from Page X"
            page_match = re.search(r"Page (\d+)", tc)
            table_page = int(page_match.group(1)) if page_match else 0
            final_chunks.append({
                "text": tc.strip(),
                "metadata": {
                    "page_num": table_page,
                    "section_type": "table",
                    "is_table": True,
                    "financial_score": 10,
                    "char_count": len(tc),
                }
            })

    return final_chunks


# ── Legacy wrapper for backward compatibility ───────────────────────────────

def chunk_document(text: str) -> list[str]:
    """Legacy interface: returns plain text chunks."""
    chunks = chunk_document_financial(text)
    return [c["text"] for c in chunks]


# ── Storage ─────────────────────────────────────────────────────────────────

def store_chunks(report_id: str, chunks: list[dict] | list[str], progress_callback=None):
    """
    Store chunks in an isolated ChromaDB collection for this report.
    Accepts either list of strings (legacy) or list of dicts with metadata.
    """
    safe_id = report_id.replace("-", "_")
    collection_name = f"report_{safe_id}"

    try:
        collection = get_chroma_client().get_collection(name=collection_name, embedding_function=embedding_fn)
        if collection.count() > 0:
            logger.info(f"Collection {collection_name} already populated, skipping re-index.")
            return
    except Exception:
        pass

    collection = get_chroma_client().get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    # Normalize to dict format
    if chunks and isinstance(chunks[0], str):
        chunks = [{"text": c, "metadata": {}} for c in chunks]

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for i, c in enumerate(chunks):
        meta = dict(c.get("metadata", {}))
        meta["report_id"] = report_id
        meta["chunk_index"] = i
        metadatas.append(meta)

    batch_size = 500
    total_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
        if progress_callback:
            progress_callback(min(total_batches, i // batch_size + 1), total_batches)

    logger.info(f"Stored {len(chunks)} chunks for report {report_id[:12]}...")


# ── Retrieval ───────────────────────────────────────────────────────────────

def retrieve_relevant_context(report_id: str, query: str, k: int = 10) -> str:
    """
    Retrieve top-k relevant chunks with financial-aware boosting.
    Boosts table chunks and high-financial-score chunks.
    """
    safe_id = report_id.replace("-", "_")
    collection_name = f"report_{safe_id}"
    try:
        collection = get_chroma_client().get_collection(name=collection_name, embedding_function=embedding_fn)
    except Exception:
        return ""

    # Use a slightly larger k to allow for reranking
    query_k = min(k + 4, 20)
    results = collection.query(query_texts=[query], n_results=query_k)

    if not results or not results["documents"] or not results["documents"][0]:
        return ""

    retrieved = results["documents"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(retrieved)

    # Simple rerank: boost tables and high-score chunks, then take top k
    scored = []
    for doc, meta in zip(retrieved, metadatas):
        score = 0.0
        if meta.get("is_table"):
            score += 3.0
        score += meta.get("financial_score", 0) * 0.2
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:k]]

    return "\n\n...[CONTEXT BREAK]...\n\n".join(top_docs)


def delete_collection(report_id: str):
    """Delete ChromaDB collection for a report."""
    safe_id = report_id.replace("-", "_")
    collection_name = f"report_{safe_id}"
    try:
        get_chroma_client().delete_collection(name=collection_name)
        logger.info(f"Deleted collection {collection_name}")
    except Exception as e:
        logger.warning(f"Failed to delete collection {collection_name}: {e}")
