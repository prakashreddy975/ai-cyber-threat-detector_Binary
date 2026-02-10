import pandas as pd
import requests
import time

# ✅ 1) UPDATE THIS URL (ngrok or local)
# NGROK EXAMPLE:
API_URL = "http://127.0.0.1:8000/predict"

# LOCAL EXAMPLE:
# API_URL = "http://127.0.0.1:8000/predict"

# ✅ 2) PATH TO YOUR TEST DATA (local path on Mac)
DATA_PATH = "/Users/prakashreddypasham/Desktop/PRAKASH/SELF_PROJECTS/Binary_AI_CYBER_DETECTOR/backend/data/raw/UNSW_NB15_testing-set.csv"

# ✅ 3) Columns to drop (labels + metadata). Edit if your CSV differs.
DROP_COLS = ["id", "label", "attack_cat", "stime", "ltime"]

# ✅ 4) Stream config
STREAM_ROWS = 50       # how many rows to sample for looping
INTERVAL_SEC = 1.0     # 1 record per second
TIMEOUT_SEC = 10

def start_demo_stream():
    print("📡 Loading dataset for live stream...")
    df = pd.read_csv(DATA_PATH)

    # --- choose a mix so UI looks active ---
    # If you have 'label' (0/1): take half attacks + half normals
    if "label" in df.columns:
        attacks = df[df["label"] == 1]
        normals = df[df["label"] == 0]

        # If dataset is imbalanced, protect from small counts
        n_attack = min(len(attacks), STREAM_ROWS // 2)
        n_normal = min(len(normals), STREAM_ROWS - n_attack)

        sample_df = pd.concat([
            attacks.sample(n=n_attack, random_state=42) if n_attack > 0 else attacks.head(0),
            normals.sample(n=n_normal, random_state=42) if n_normal > 0 else normals.head(0),
        ]).sample(frac=1.0, random_state=42)  # shuffle
    else:
        # fallback: random sample
        sample_df = df.sample(n=min(STREAM_ROWS, len(df)), random_state=42)

    print(f"🚀 Starting Stream ({INTERVAL_SEC} packet/sec) -> {API_URL}")
    print("Keep your React Dashboard open to see live updates!\n")

    for idx, row in sample_df.iterrows():
        payload = row.to_dict()

        # actual label for display (optional)
        actual = payload.get("attack_cat", "Normal")

        # remove labels/meta
        features = {k: v for k, v in payload.items() if k not in DROP_COLS}

        # IMPORTANT: ensure JSON-serializable
        for k, v in list(features.items()):
            # pandas sometimes uses numpy types which requests can serialize,
            # but we make it safer:
            if pd.isna(v):
                features[k] = 0

        try:
            resp = requests.post(
                API_URL,
                json={"features": features, "actual_label": actual},
                timeout=TIMEOUT_SEC,
            )

            if resp.status_code == 200:
                result = resp.json()

                # Your backend returns: prediction (0/1), threat_score, risk_level, threat_detected, latency...
                status = "🚩 ATTACK" if result.get("prediction") == 1 else "✅ NORMAL"
                score = result.get("threat_score", None)
                risk = result.get("risk_level", None)
                latency = result.get("latency", None)

                print(
                    f"Row {idx}: {status} | score={score:.4f} | risk={risk} "
                    f"| actual={actual} | latency={latency}ms"
                )
            else:
                print(f"❌ Server Error {resp.status_code}: {resp.text[:200]}")

        except Exception as e:
            print(f"📡 Connection Error: {e}")

        time.sleep(INTERVAL_SEC)

    print("\n✅ Stream finished. Re-run to stream again.")

if __name__ == "__main__":
    start_demo_stream()
