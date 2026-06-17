"""
OTT Recommendation API — FastAPI

Endpoints:
    GET /health                         Health check
    GET /recommend/{user_id}            Multi-Stage ML recommendations (LightGBM)
    GET /models                         List available saved models
"""
from fastapi import FastAPI, HTTPException, Query
import time
import threading

import pandas as pd

from data.load_auditlog import load_auditlog
from data.load_content import load_content
from data.preprocessing import clean_and_merge, build_rating_matrix
from utils.logger import log


# ================= CONFIG =================
DATA_TTL = 15 * 60  # seconds before data reload
# =========================================


app = FastAPI(
    title="OTT Recommendation API",
    version="3.0.0",
    description="Multi-Stage Machine Learning Recommendation System",
)


# ================= GLOBAL CACHE =================
data_lock = threading.Lock()
last_loaded = 0

auditlog = None
content = None
R = None
# ================================================


# ================= DATA LOADER =================
def load_data():
    global auditlog, content, R, last_loaded

    log("Reloading data...")

    raw_auditlog = load_auditlog()
    raw_content = load_content()
    auditlog, content = clean_and_merge(raw_auditlog, raw_content)

    R = build_rating_matrix(auditlog)

    last_loaded = time.time()
    log("Data loaded successfully")


def ensure_data_loaded():
    global last_loaded

    now = time.time()
    if last_loaded == 0 or now - last_loaded > DATA_TTL:
        with data_lock:
            if last_loaded == 0 or time.time() - last_loaded > DATA_TTL:
                load_data()
# ================================================


# ================= HEALTH =================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_loaded": last_loaded,
    }
# =========================================


# ================= PIPELINE RECOMMEND =================
@app.get("/recommend/{user_id}")
def recommend_endpoint(user_id: str, top_n: int = Query(10, ge=1, le=100)):
    """Get Multi-Stage Machine Learning (LightGBM) recommendations."""
    ensure_data_loaded()

    from pipeline.serve import recommend_pipeline
    
    import math
    recs = recommend_pipeline(user_id, auditlog, content, top_n)
    
    records = recs.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
                
    return {
        "user_id": user_id,
        "model": "multi_stage_pipeline",
        "recommendations": records,
    }


# ================= MODELS =================
@app.get("/models")
def list_models():
    """List all saved models."""
    from models.persistence import list_saved_models
    return {"models": list_saved_models()}


# ================= LOCAL RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
# =========================================
