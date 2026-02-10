# 🛡️ AI-Enhanced Cybersecurity Threat Detector (Binary) + Real-Time SOC Dashboard + SOC Copilot (RAG)
Phase-wise, step-wise workflow documentation (from Phase 0 → current)

> This README is written so you can **teach juniors** and also use it as a **project handover doc**.  
> It explains **what we built**, **why we built it**, **how components talk**, and **how to run** everything end-to-end.

---

## 0) What we built (one-line)
A full-stack cybersecurity system that:
- **Scores network traffic** using an ML model (Random Forest binary IDS)
- **Streams detections live** to a React SOC dashboard via **WebSockets**
- **Stores alerts** in **MySQL** for history/audit
- Adds a **SOC Copilot** (RAG agent) grounded in **runbooks (.md)** to answer analyst questions

---

## 1) System workflow (end-to-end)
### 1.1 Data flow (runtime)
1. **Traffic record** arrives (from curl / streamer / batch)
2. Backend **aligns features** to the model’s expected columns
3. Backend runs **predict_proba → threat_score**
4. Backend computes:
   - `prediction` (0/1)
   - `risk_level` (LOW/MEDIUM/HIGH)
   - `threat_detected` boolean
5. Backend:
   - Saves risky alerts to **MySQL**
   - Broadcasts event to all **WebSocket** clients (`/ws`)
6. React dashboard:
   - Receives events via WS
   - Updates metrics, charts, and feed in real-time
7. SOC Copilot:
   - If asked a security question, retrieves relevant runbook chunks
   - Answers with citations OR refuses if low confidence

### 1.2 Why this architecture matters
This mirrors real enterprise SOC systems:
- **Model inference** = detection engine
- **WebSocket** = live monitoring stream (Grafana/Splunk-like)
- **DB** = audit + history
- **RAG Copilot** = analyst productivity + safe AI

---

## 2) Tech stack
### Backend
- FastAPI (REST + WebSockets)
- scikit-learn RandomForest pipeline + joblib
- pandas / numpy for feature alignment
- SQLAlchemy for DB models and CRUD
- MySQL (docker)

### Frontend
- React (Vite)
- chart.js + react-chartjs-2
- WebSocket for live feed

### SOC Copilot (RAG)
- Markdown runbooks (`.md`)
- Chunking (heading-aware) + embeddings
- Similarity thresholding (avoid hallucination)
- `/agent/ask` endpoint

---

## 3) Repo structure (recommended)
> Your actual repo may differ slightly; this is the intended structure.

Binary_AI_CYBER_DETECTOR/
backend/
main.py
database.py
models.py
crud.py
ml/
rf_threat_detector.joblib (NOT committed; local only)
data/ (NOT committed)
rag_docs/
dos_ddos.md
exploits.md
reconnaissance.md
false_positives.md
incident_response.md
assistant_policy.md
streamer/
stream_demo.py
requirements.txt

frontend/
src/
App.jsx
App.css
index.css
components/
CopilotWidget.jsx
CopilotWidget.css
.env
package.json

docker-compose.yml
.gitignore
README.md



---

# ✅ PHASE-WISE WORKFLOW 

---

## PHASE 0 — Planning (Scope + threat definition)
### Goal
Build a system that is not only “accuracy on a dataset” but a **usable SOC product**.

### Decisions made
- Start with **binary IDS** (Normal vs Attack)
- Add:
  - threshold tuning
  - risk levels
  - real-time streaming
  - persistent DB
  - SOC UI
  - RAG agent

---

## PHASE 1 — ML threat detector (Random Forest binary)
### Step 1.1 Dataset preparation
- Use UNSW-NB15 processed dataset with ~43 features (proto, service, state, rates, ttl, ct_* etc.)
- Split train/test

### Step 1.2 Train model
- Train Logistic Regression and Random Forest as baselines
- Evaluate using:
  - precision/recall/F1
  - ROC-AUC

### Step 1.3 Select RF + threshold
- RF performed much better than LR for this dataset.
- We used **threshold tuning** to choose behavior trade-off:
  - lower threshold → more recall (detect more attacks), more false positives
  - higher threshold → fewer false positives, may miss attacks

### Step 1.4 Save artifacts
- Save the trained pipeline:
  - `rf_threat_detector.joblib`
- IMPORTANT: do not commit large binaries to GitHub; use `.gitignore` or Git LFS.

---

## PHASE 2 — Backend API (FastAPI inference service)
### Step 2.1 Why FastAPI
- Fast + typed + great for ML inference
- Supports **WebSockets** natively

### Step 2.2 Implement endpoints
Core endpoints (you already have most of these):

- `GET /health`  
  Returns `{ "status": "running" }` for uptime checks.

- `GET /template`  
  Returns a full feature template to help clients send correct payloads.

- `GET /expected_columns`  
  Returns the list of expected model input columns (detected from pipeline).

- `POST /predict`  
  Takes one record, returns:
  - `prediction` 0/1
  - `threat_score` 0..1
  - `risk_level` LOW/MEDIUM/HIGH
  - `threat_detected` bool
  - `latency` ms
  - `timestamp`

- `POST /predict_batch`  
  Takes an array of records, returns summary + results (and can broadcast alerts).

- `GET /alerts`  
  Fetch alerts stored in DB (or memory earlier).

- `GET /metrics`  
  Helpful debug info (threshold, ws clients, stored alerts count, etc.)

- `WS /ws`  
  Clients connect once and receive broadcast events from `/predict`.

### Step 2.3 Feature alignment (very important)
Because real events may miss some columns:
- Convert input dict → DataFrame
- Reindex to `EXPECTED_COLS`
- Fill missing columns with 0

This prevents model input mismatch errors.

---

## PHASE 3 — Real-time streaming (WebSockets)
### Step 3.1 Why WebSockets
- SOC dashboards must update instantly
- Polling `/alerts` is slow and expensive

### Step 3.2 ConnectionManager pattern
- Maintain `active_connections`
- Broadcast JSON to all connected clients
- Remove dead sockets safely

### Step 3.3 Message types
- optional `hello` (on connect)
- optional `ping` (keepalive)
- inference events (the ones your UI uses)

---

## PHASE 4 — Public access (Colab + ngrok)
### Step 4.1 Running backend on Colab
Colab has a running event loop; `uvicorn.run()` can fail with:
- `asyncio.run() cannot be called from a running event loop`

We used a Colab-friendly startup pattern:
- `uvicorn.Server(Config)` with `await server.serve()`
- `nest_asyncio.apply()` when needed

### Step 4.2 ngrok
- ngrok exposes the Colab port publicly.
- We printed:
  - Public HTTPS URL
  - WebSocket WSS URL
  - Predict endpoint URL

Example:
- Public: `https://<id>.ngrok-free.dev`
- WS: `wss://<id>.ngrok-free.dev/ws`

---

## PHASE 5 — React SOC dashboard (Frontend)
### Step 5.1 Vite env setup
`.env` (frontend):
- VITE_API_BASE=https://<ngrok-id>.ngrok-free.dev
- VITE_WS_URL=wss://<ngrok-id>.ngrok-free.dev/ws


### Step 5.2 WebSocket stream
React opens a WebSocket:
- onopen → “CONNECTED”
- onmessage → update charts + logs
- onclose/onerror → show backend down

### Step 5.3 StrictMode bug (teachable moment)
React StrictMode double-mounts effects in dev → WebSocket connects twice → can appear “down”.
Fix used:
- remove StrictMode wrapper for local dev
- or implement WS reconnect logic + cleanup

### Step 5.4 Full screen layout
Vite default `index.css` centers content (`display:flex; place-items:center`) which breaks dashboard.
Fix:
- set `html, body, #root` to 100% height/width
- body `display:block`
- allow scrolling by removing fixed constraints

---

## PHASE 6 — Streamer (simulate real traffic)
### Step 6.1 Why streamer
To demonstrate live SOC behavior, we push 1 record/sec into `/predict`.

### Step 6.2 What it does
- reads test CSV
- samples N rows
- removes label columns
- sends to backend with correct JSON shape
- sleeps 1 second between requests

This makes UI continuously update.

---

## PHASE 7 — Database (MySQL) persistence
### Step 7.1 Why DB
Without DB:
- alerts disappear on restart
- no audit trail
- no historical analytics

### Step 7.2 Docker compose
We used docker compose to run MySQL locally:
- container maps port 3306
- persistent volume keeps data

### Step 7.3 SQLAlchemy layer (how to teach juniors)
- `database.py`: creates engine + SessionLocal
- `models.py`: defines tables (Alert, RagChunk, etc.)
- `crud.py`: reusable DB functions (create_alert, get_alerts)

Backend calls CRUD inside endpoints:
- `/predict` inserts alerts when `threat_detected == true`
- `/alerts` reads from DB

---

## PHASE 8 — SOC Copilot (RAG agent)
### Step 8.1 Why RAG
LLMs hallucinate. Security must be grounded.

RAG ensures:
- retrieve authoritative runbook chunks
- answer using only that context
- refuse if confidence low

### Step 8.2 Runbooks
We created multiple `.md` runbooks (dos, exploits, recon, IR, etc.) used as source of truth.

### Step 8.3 Agent routing
- Small talk → normal response (no retrieval)
- Security question → retrieve + answer with citations
- Low similarity → refuse and ask to expand docs

This is why you saw:
- “what is ddos?” → best_score too low (needs indexing or stronger definition chunk)

---

# ✅ HOW TO RUN (Local workflow)


```bash
## 1) Start MySQL
### From repo root (where docker-compose.yml exists):
docker compose up -d
docker ps

## Backend setup

cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000


verify 

curl http://127.0.0.1:8000/health

## frontend setup
cd frontend
npm install
npm run dev

VITE_API_BASE=http://127.0.0.1:8000
VITE_WS_URL=ws://127.0.0.1:8000/ws


## Testing 

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/template
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "dur": 0.2,
      "proto": "tcp",
      "service": "http",
      "state": "FIN",
      "spkts": 4,
      "dpkts": 2,
      "sbytes": 200,
      "dbytes": 100
    }
  }'
curl "http://127.0.0.1:8000/alerts?limit=20"
