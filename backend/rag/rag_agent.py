import os
import json
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from sqlalchemy.orm import Session
from db.models import RagChunk
from .text_splitter import smart_chunk_markdown

RUNBOOK_DIR = os.path.join(os.path.dirname(__file__), "runbooks")

INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_store")
FAISS_PATH = os.path.join(INDEX_DIR, "rag.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "google/flan-t5-base")

# Confidence gating (fixes “same answer for hi”)
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

# Lazy singletons
_embedder = None
_llm = None
_index = None
_meta = None

CYBER_KEYWORDS = [
    "attack","intrusion","ddos","dos","malware","phishing","scan","port","probe",
    "tcp","udp","tls","ssh","http","dns","firewall","ids","ips","siem",
    "incident","mitigation","alert","risk","threat","exploit","recon",
    "backdoor","botnet","payload","c2","command and control","lateral","privilege",
    "brute","password","credential","exfil","leak","anomaly","anomalous",
    "fin","syn","rst","ack","pcap","wireshark","nginx","apache"
]

SMALL_TALK = {"hi","hello","hey","yo","thanks","thank you","ok","okay","cool","nice"}

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder

def get_llm():
    global _llm
    if _llm is None:
        _llm = pipeline("text2text-generation", model=LLM_MODEL_NAME)
    return _llm

def _ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)

def load_faiss_if_exists() -> bool:
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
        if name.endswith(".md"):
            path = os.path.join(RUNBOOK_DIR, name)
            with open(path, "r", encoding="utf-8") as f:
                docs.append((name, f.read()))
    return docs

def rebuild_index(db: Session) -> Dict[str, Any]:
    docs = read_runbooks()
    if not docs:
        return {"ok": False, "error": f"No .md found in {RUNBOOK_DIR}"}

    # wipe chunk table
    db.query(RagChunk).delete()
    db.commit()

    all_texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for doc_name, content in docs:
        chunks = smart_chunk_markdown(content, max_chars=1400, overlap=180)
        for i, ch in enumerate(chunks):
            all_texts.append(ch)
            meta.append({"doc": doc_name, "chunk_index": i})

    embedder = get_embedder()
    vecs = embedder.encode(all_texts, normalize_embeddings=True)
    vecs = np.array(vecs, dtype="float32")
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(vecs)

    # store chunks in MySQL (metadata + text)
    for idx, m in enumerate(meta):
        db.add(RagChunk(
            doc_name=m["doc"],
            chunk_index=m["chunk_index"],
            chunk_text=all_texts[idx],
            last_score=None
        ))
    db.commit()

    save_faiss(index, meta)

    global _index, _meta
    _index, _meta = index, meta

    return {
        "ok": True,
        "docs": len(docs),
        "chunks": len(all_texts),
        "embed_model": EMBED_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
        "min_score": RAG_MIN_SCORE
    }

def _ensure_loaded():
    global _index, _meta
    if _index is None or _meta is None:
        if not load_faiss_if_exists():
            raise RuntimeError("RAG index not built. Call POST /rag/index first.")

def retrieve(db: Session, query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Returns: { best_score, sources[] }.
    Confidence gating: if best_score < RAG_MIN_SCORE, sources=[]
    """
    _ensure_loaded()
    embedder = get_embedder()

    qv = embedder.encode([query], normalize_embeddings=True)
    qv = np.array(qv, dtype="float32")

    top_k = max(1, min(top_k, 8))
    scores, ids = _index.search(qv, top_k)

    best = float(scores[0][0]) if len(scores[0]) else 0.0
    if best < RAG_MIN_SCORE:
        return {"best_score": best, "sources": []}

    sources = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        m = _meta[idx]
        row = (db.query(RagChunk)
               .filter(RagChunk.doc_name == m["doc"], RagChunk.chunk_index == m["chunk_index"])
               .first())
        if not row:
            continue
        row.last_score = float(score)
        sources.append({
            "doc": m["doc"],
            "chunk_index": m["chunk_index"],
            "score": float(score),
            "text": row.chunk_text[:1500],
        })
    db.commit()
    return {"best_score": best, "sources": sources}

def is_small_talk(q: str) -> bool:
    ql = q.strip().lower()
    return ql in SMALL_TALK or len(ql) <= 2

def is_cyber_query(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in CYBER_KEYWORDS)

def llm_grounded_answer(query: str, sources: List[Dict[str, Any]]) -> str:
    """
    Generates answer with citations [1], [2] mapping to sources order.
    """
    llm = get_llm()
    context = "\n\n".join(
        [f"[{i+1}] ({s['doc']}#{s['chunk_index']})\n{s['text']}" for i, s in enumerate(sources)]
    )

    prompt = f"""
You are a SOC Copilot. Answer the user's question using ONLY the context sources below.
Rules:
- If the answer is not in the sources, say: "Not found in runbooks." Then suggest what document/section should be added.
- Keep it concise and actionable.
- Add citations like [1], [2] after relevant sentences.
- Use bullets when helpful.

User question: {query}

Sources:
{context}

Answer:
""".strip()

    out = llm(prompt, max_new_tokens=240, do_sample=False)
    return out[0]["generated_text"].strip()

def agent_ask(db: Session, query: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Agent router:
    - small talk → normal assistant response (no RAG)
    - non-cyber → normal response + suggestion
    - cyber → RAG retrieval with confidence gating + grounded answer
    """
    q = query.strip()
    if not q:
        return {"mode": "chat", "answer": "Ask me about an alert, a risk level, or an incident response step.", "sources": []}

    if is_small_talk(q):
        return {"mode": "chat", "answer": "Hey! 👋 Ask me a cybersecurity question (e.g., HIGH risk mitigation, DDoS indicators, TCP FIN meaning).", "sources": []}

    if not is_cyber_query(q):
        return {
            "mode": "chat",
            "answer": "I can help with cybersecurity runbooks and alert triage. Try asking about incidents, mitigations, protocols, or IDS alerts.",
            "sources": []
        }

    ret = retrieve(db, q, top_k=top_k)
    sources = ret["sources"]

    if not sources:
        return {
            "mode": "rag",
            "answer": f"I couldn't find relevant runbook context for that (best_score={ret['best_score']:.3f}). "
                      f"Add/expand a runbook on that topic, then re-index.",
            "sources": [],
            "best_score": ret["best_score"]
        }

    answer = llm_grounded_answer(q, sources)
    return {
        "mode": "rag",
        "answer": answer,
        "sources": sources,
        "best_score": ret["best_score"]
    }
