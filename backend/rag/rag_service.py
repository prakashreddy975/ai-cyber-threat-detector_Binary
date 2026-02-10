import os
import json
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from sqlalchemy.orm import Session
from db.models import RagChunk
from .text_splitter import chunk_text

RUNBOOK_DIR = os.path.join(os.path.dirname(__file__), "runbooks")

# Where to store FAISS on disk
INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_store")
FAISS_PATH = os.path.join(INDEX_DIR, "rag.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "google/flan-t5-base")  # small & works

# Lazy singletons
_embedder = None
_llm = None
_index = None
_meta = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def get_llm():
    global _llm
    if _llm is None:
        # text2text-generation works for flan-t5
        _llm = pipeline("text2text-generation", model=LLM_MODEL_NAME)
    return _llm

def _ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def load_faiss_if_exists() -> bool:
    """Loads FAISS + meta if present."""
    global _index, _meta
    if os.path.exists(FAISS_PATH) and os.path.exists(META_PATH):
        _index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        return True
    return False

def save_faiss(index, meta):
    _ensure_index_dir()
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def read_runbooks() -> List[Tuple[str, str]]:
    docs = []
    if not os.path.isdir(RUNBOOK_DIR):
        return docs

    for name in sorted(os.listdir(RUNBOOK_DIR)):
        if not name.endswith(".md"):
            continue
        path = os.path.join(RUNBOOK_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((name, f.read()))
    return docs

def rebuild_index(db: Session) -> Dict[str, Any]:
    """
    1) Load runbooks
    2) Chunk them
    3) Embed
    4) Build FAISS
    5) Store chunk metadata in MySQL + meta mapping on disk
    """
    docs = read_runbooks()
    if not docs:
        return {"ok": False, "error": f"No markdown files found in {RUNBOOK_DIR}"}

    # Clear previous chunks in DB
    db.query(RagChunk).delete()
    db.commit()

    all_texts = []
    meta = []  # aligns with FAISS vector order

    for doc_name, content in docs:
        chunks = chunk_text(content, max_chars=1200, overlap=150)
        for i, ch in enumerate(chunks):
            all_texts.append(ch)
            meta.append({"doc": doc_name, "chunk_index": i})

    embedder = get_embedder()
    vecs = embedder.encode(all_texts, normalize_embeddings=True)

    vecs = np.array(vecs, dtype="float32")
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine since normalized
    index.add(vecs)

    # Insert chunk rows in DB with same order as meta
    for idx, m in enumerate(meta):
        rc = RagChunk(
            doc_name=m["doc"],
            chunk_index=m["chunk_index"],
            chunk_text=all_texts[idx],
            last_score=None,
        )
        db.add(rc)
    db.commit()

    save_faiss(index, meta)

    # keep in memory too
    global _index, _meta
    _index = index
    _meta = meta

    return {
        "ok": True,
        "docs": len(docs),
        "chunks": len(all_texts),
        "embed_model": EMBED_MODEL_NAME,
        "faiss_path": FAISS_PATH,
    }

def _ensure_loaded():
    global _index, _meta
    if _index is None or _meta is None:
        ok = load_faiss_if_exists()
        if not ok:
            raise RuntimeError("RAG index not built. Call POST /rag/index first.")

def retrieve(db: Session, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    _ensure_loaded()
    embedder = get_embedder()

    qv = embedder.encode([query], normalize_embeddings=True)
    qv = np.array(qv, dtype="float32")

    scores, ids = _index.search(qv, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = _meta[idx]
        # fetch the chunk text from DB (source of truth)
        row = (
            db.query(RagChunk)
            .filter(RagChunk.doc_name == m["doc"], RagChunk.chunk_index == m["chunk_index"])
            .first()
        )
        if row is None:
            continue

        # store last score snapshot (optional)
        row.last_score = float(score)
        results.append({
            "doc": m["doc"],
            "chunk_index": m["chunk_index"],
            "score": float(score),
            "text": row.chunk_text[:1200],  # cap to keep responses smaller
        })
    db.commit()
    return results

def answer_with_llm(query: str, sources: List[Dict[str, Any]]) -> str:
    llm = get_llm()

    # Grounded prompt: only use the sources
    context = "\n\n".join(
        [f"[{i+1}] ({s['doc']}#{s['chunk_index']}) {s['text']}" for i, s in enumerate(sources)]
    )

    prompt = f"""
You are a SOC assistant. Answer the user query using ONLY the context below.
If context is insufficient, say what’s missing and suggest what to check next.

Query: {query}

Context:
{context}

Answer (concise, actionable, include bullet points):
""".strip()

    out = llm(prompt, max_new_tokens=220, do_sample=False)
    return out[0]["generated_text"].strip()
