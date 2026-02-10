import nest_asyncio
import uvicorn
import joblib
import pandas as pd
import numpy as np

from pydantic import BaseModel
from pyngrok import ngrok
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import sys, time, asyncio, traceback, logging
from collections import deque
from typing import List, Optional
from fastapi import FastAPI, Request
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import time
import traceback

from dotenv import load_dotenv
load_dotenv()

from db.database import engine, get_db
from db.database import Base
from db import models, crud

from rag.rag_service import rebuild_index, retrieve, answer_with_llm


Base.metadata.create_all(bind=engine)

MODEL_PATH = "/Users/prakashreddypasham/Desktop/PRAKASH/SELF_PROJECTS/Binary_AI_CYBER_DETECTOR/backend/data/rf_threat_detector.joblib"
THRESHOLD = 0.30

# ============================================================
# MODEL_PATH = os.getenv("MODEL_PATH", "rf_threat_detector.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.30"))

# local default port
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Optional ngrok
USE_NGROK = os.getenv("USE_NGROK", "false").lower() in ("1", "true", "yes", "y")
NGROK_AUTHTOKEN = os.getenv("392vSfugNWnv7hVTLyvRPkItjL1_6VGdCqWzZngEjkTqP1NPY", "")

# In-memory alert store (last N)
ALERTS = deque(maxlen=500)

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ThreatDetector")


def risk_level(score: float) -> str:
    return "HIGH" if score >= 0.7 else "MEDIUM" if score >= 0.4 else "LOW"

logger.info("Loading model...")
model = joblib.load(MODEL_PATH)
logger.info(f"✅ Model loaded from: {MODEL_PATH}")

app = FastAPI(
    title="AI Cyber Threat Detector",
    description="Random Forest based IDS",
    version="1.0"
)

EXPECTED_COLS: Optional[List[str]] = None
try:
    prep = model.named_steps.get("prep", None)
    if prep is not None and hasattr(prep, "feature_names_in_"):
        EXPECTED_COLS = list(prep.feature_names_in_)
        logger.info(f"✅ Detected expected raw columns: {len(EXPECTED_COLS)}")
    else:
        logger.warning("⚠️ Could not detect expected columns (prep.feature_names_in_ missing).")
except Exception:
    logger.error("⚠️ Column detection failed", exc_info=True)



# ----------------------------
# REQUEST LOGGER
# ----------------------------
# Live request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️  {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"⬅️  {request.method} {request.url.path} → {response.status_code}")
    return response
# ----------------------------
# IN-MEMORY ALERT STORE
# ----------------------------
ALERTS = deque(maxlen=500)  # store last 500 alerts


# ----------------------------
# SCHEMAS
# ----------------------------
from pydantic import BaseModel, ConfigDict

class TrafficEvent(BaseModel):
    record: dict


class BatchRequest(BaseModel):

    records: List[Dict[str, Any]]

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    actual_label: str | None = None

class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 4
    use_llm: bool = True


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast JSON message to all active clients"""
        dead = []
        payload = json.dumps(message)

        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

def align_record_to_expected_cols(record: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    if EXPECTED_COLS is not None:
        df = df.reindex(columns=EXPECTED_COLS, fill_value=0)
    return df

def make_response(pred_int: int, score: float, latency_ms: float) -> dict:
    lvl = risk_level(score)
    threat_detected = bool(pred_int == 1 and score >= 0.4)

    return {
        "prediction": int(pred_int),                 # 0/1
        "threat_score": float(score),                # 0..1
        "risk_level": lvl,                           # LOW/MEDIUM/HIGH
        "threat_detected": threat_detected,          # bool
        "threshold": THRESHOLD,
        "latency": round(float(latency_ms), 2),      # ms
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

@app.get("/health")
def health():
    logger.info("Health check.......")
    return {"status": "running"}

@app.get("/template")
def template():
    t = {c: 0 for c in EXPECTED_COLS} if EXPECTED_COLS else {}

    # small sensible defaults
    if "id" in t: t["id"] = 1
    if "dur" in t: t["dur"] = 0.2
    if "proto" in t: t["proto"] = "tcp"
    if "service" in t: t["service"] = "http"
    if "state" in t: t["state"] = "FIN"
    if "spkts" in t: t["spkts"] = 4
    if "dpkts" in t: t["dpkts"] = 2
    if "sbytes" in t: t["sbytes"] = 200
    if "dbytes" in t: t["dbytes"] = 100

    return {"record": t}



@app.get("/expected_columns")
def expected_columns():
    logger.info("Expected columns requested")
    if EXPECTED_COLS is None:
        return {"available": False}
    return {
        "available": True,
        "n_cols": len(EXPECTED_COLS),
        "columns": EXPECTED_COLS
    }

from fastapi import Depends
from sqlalchemy.orm import Session

# Prediction Endpoint
# ----------------------------
@app.post("/predict")
async def predict_intrusion(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Input:  { "features": { ... 43 cols ... } }
    Output: result dict + broadcasts same dict to /ws clients
    """
    try:
        start = time.perf_counter()

        df = align_record_to_expected_cols(request.features)

        # Score
        proba = float(model.predict_proba(df)[:, 1][0])
        pred = int(proba >= THRESHOLD)

        latency_ms = (time.perf_counter() - start) * 1000.0

        actual = getattr(request, "actual_label", None)  # if extra="allow"
        result = make_response(pred, proba, latency_ms)
        if actual is not None:
            result["actual"] = actual  # ✅ include in response + WS broadcast

        # Store alert if attack with medium/high risk
        event = crud.create_event(
            db,
            features=request.features,
            result=result,
            actual_label=actual
        )

        if result["threat_detected"]:
            crud.create_alert(db, event_id=event.id, result=result)



        # Broadcast to WS clients (Grafana-like stream)
        await manager.broadcast(result)

        return result

    except Exception as e:
        err = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure request.features includes correct keys/types. Try GET /template then POST /predict using that record."
        }
        return err

@app.post("/predict_batch")
async def predict_batch(payload: BatchRequest):
    """
    Input: { "records": [ {..}, {..} ] }
    Output: summary + first 200 results
    Broadcasts alerts to /ws clients as they occur.
    """
    try:
        start = time.perf_counter()

        df = pd.DataFrame(payload.records)
        if EXPECTED_COLS is not None:
            df = df.reindex(columns=EXPECTED_COLS, fill_value=0)

        probas = model.predict_proba(df)[:, 1]
        preds = (probas >= THRESHOLD).astype(int)

        latency_ms = (time.perf_counter() - start) * 1000.0

        results = []
        attack_count = 0

        for i, (p, s) in enumerate(zip(preds, probas)):
            s = float(s)
            p = int(p)
            r = make_response(p, s, latency_ms=latency_ms)  # batch latency shown same for simplicity
            r["index"] = i
            results.append(r)




            if r["prediction"] == 1:
                attack_count += 1

            # store & broadcast only meaningful alerts
            if r["threat_detected"]:
                ALERTS.appendleft(r)
                await manager.broadcast(r)

        return {
            "n": len(results),
            "attack_count": int(attack_count),
            "normal_count": int(len(results) - attack_count),
            "threshold": THRESHOLD,
            "latency": round(float(latency_ms), 2),
            "results": results[:200],
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

def make_response(pred_int: int, score: float, latency_ms: float) -> dict:
    lvl = risk_level(score)
    threat_detected = bool(pred_int == 1 and score >= 0.4)
    now = time.time()

    return {
        "prediction": int(pred_int),
        "threat_score": float(score),
        "risk_level": lvl,
        "threat_detected": threat_detected,
        "threshold": THRESHOLD,
        "latency": round(float(latency_ms), 2),
        "ts": now,  # ✅ add this
        "timestamp": datetime.utcfromtimestamp(now).isoformat() + "Z",
    }

@app.get("/alerts")
def get_alerts(limit: int = 50, risk: str | None = None, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    rows = crud.list_alerts(db, limit=limit, risk=risk)
    return {
        "count": len(rows),
        "alerts": [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat(),
                "event_id": a.event_id,
                "prediction": a.prediction,
                "threat_score": a.threat_score,
                "risk_level": a.risk_level,
                "threshold": a.threshold,
                "latency": a.latency_ms,
                "note": a.note,
            }
            for a in rows
        ],
    }

@app.get("/metrics")
def metrics():
    return {
        "threshold": THRESHOLD,
        "stored_alerts": len(ALERTS),
        "expected_cols_detected": EXPECTED_COLS is not None,
        "n_expected_cols": (len(EXPECTED_COLS) if EXPECTED_COLS else None),
        "ws_clients": len(manager.active_connections),
        "model_path": MODEL_PATH,
    }

@app.get("/events")
def get_events(limit: int = 100, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    rows = crud.list_events(db, limit=limit)
    return {
        "count": len(rows),
        "events": [
            {
                "id": e.id,
                "created_at": e.created_at.isoformat(),
                "prediction": e.prediction,
                "threat_score": e.threat_score,
                "risk_level": e.risk_level,
                "threat_detected": e.threat_detected,
                "threshold": e.threshold,
                "latency": e.latency_ms,
                "actual_label": e.actual_label,
            }
            for e in rows
        ],
    }

# Rag End Points

@app.get("/rag/status")
def rag_status():
    from rag.rag_service import FAISS_PATH, META_PATH
    return {
        "faiss_exists": os.path.exists(FAISS_PATH),
        "meta_exists": os.path.exists(META_PATH),
        "embed_model": os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "llm_model": os.getenv("RAG_LLM_MODEL", "google/flan-t5-base"),
    }

@app.post("/rag/index")
def rag_index(db: Session = Depends(get_db)):
    return rebuild_index(db)

@app.post("/rag/query")
def rag_query(payload: RagQueryRequest, db: Session = Depends(get_db)):
    sources = retrieve(db, payload.query, top_k=max(1, min(payload.top_k, 8)))

    if payload.use_llm:
        answer = answer_with_llm(payload.query, sources)
    else:
        # simple fallback: just return top sources
        answer = "Top relevant sources retrieved. Enable use_llm=true to generate a grounded answer."

    return {
        "query": payload.query,
        "answer": answer,
        "sources": sources
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Clients connect and receive broadcast messages whenever /predict or /predict_batch runs.
    We also keep the connection alive with a periodic server ping message.
    """
    await manager.connect(websocket)

    try:
        # Send a hello message immediately
        await websocket.send_text(json.dumps({
            "type": "hello",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": "connected to /ws"
        }))

        while True:
            # Keep-alive: wait for client ping OR timeout and send server ping
            try:
                # If client sends something, read it (optional)
                _ = await websocket.receive_text()
            except Exception:
                # If receive fails, break
                pass

            # Optional server keepalive message
            await websocket.send_text(json.dumps({
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }))
            import asyncio
            await asyncio.sleep(10)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# ----------------------------
# Run Server
# ----------------------------

# from pyngrok import ngrok
# import threading # Added import for running in a separate thread

# # Paste your ngrok token here
# NGROK_AUTHTOKEN = "392vSfugNWnv7hVTLyvRPkItjL1_6VGdCqWzZngEjkTqP1NPY"
# ngrok.set_auth_token(NGROK_AUTHTOKEN)

# PORT = 8000
# public_url = ngrok.connect(PORT, bind_tls=True)
# https_url = public_url.public_url
# wss_url = https_url.replace("https://", "wss://") + "/ws"
# api_url = https_url + "/predict"

# # Print URLs for frontend/backend
# print("\n" + "=" * 60)
# print("🚀 SERVER STARTED")
# print("=" * 60)
# print(f"🌍 Public HTTPS URL : {https_url}")
# print(f"🔌 WebSocket WSS   : {wss_url}")
# print(f"📡 Predict API     : {api_url}")
# print("\n👉 Put this in React .env:")
# print(f"VITE_WS_URL={wss_url}")
# print("=" * 60 + "\n")

# logger.info("Starting FastAPI server...")

# nest_asyncio.apply()

# config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="debug", access_log=True)
# server = uvicorn.Server(config)

# logger.info("🚀 Starting server on http://127.0.0.1:8000")

# server.serve()


# if __name__ == "__main__":
#     import uvicorn
#     from pyngrok import ngrok
#     import os

#     PORT = int(os.getenv("PORT", "8000"))
#     HOST = os.getenv("HOST", "0.0.0.0")

#     USE_NGROK = os.getenv("USE_NGROK", "false").lower() in ("1", "true", "yes", "y")
#     NGROK_AUTHTOKEN = os.getenv("392vSfugNWnv7hVTLyvRPkItjL1_6VGdCqWzZngEjkTqP1NPY", "")  # ✅ correct

#     if USE_NGROK:
#         if not NGROK_AUTHTOKEN:
#             raise RuntimeError("USE_NGROK=true but NGROK_AUTHTOKEN is not set.")
#         ngrok.set_auth_token(NGROK_AUTHTOKEN)
#         public_url = ngrok.connect(PORT, bind_tls=True).public_url
#         https_url = public_url
#         wss_url = https_url.replace("https://", "wss://") + "/ws"
#         api_url = https_url + "/predict"

     

#         logger.info("\n" + "=" * 60)
#         logger.info("🚀 SERVER STARTED")
#         logger.info("=" * 60)
#         logger.info(f"🌍 Public HTTPS URL : {https_url}")
#         logger.info(f"🔌 WebSocket WSS   : {wss_url}")
#         logger.info(f"📡 Predict API     : {api_url}")
#         logger.info("\n👉 Put this in React .env:")
#         logger.info(f"VITE_WS_URL={wss_url}")
#         logger.info("=" * 60 + "\n")

#     uvicorn.run(app, host=HOST, port=PORT, log_level="debug", access_log=True)


if __name__ == "__main__":
    import os
    import uvicorn

    PORT = int(os.getenv("PORT", "8000"))
    HOST = os.getenv("HOST", "0.0.0.0")

    logger.info(f" Starting server on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="debug", access_log=True)