"""
OTT Recommendation API — FastAPI

Endpoints:
    GET /health                         Health check
    GET /recommend/{user_id}/svd        SVD-based recommendations
    GET /recommend/{user_id}/fpgrowth   FP-Growth-based recommendations
    GET /recommend/{user_id}/similarity Item-item similarity recommendations
    GET /recommend/{user_id}/pipeline   Multi-Stage ML recommendations (LightGBM)
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
    version="2.0.0",
    description="Multi-algorithm recommendation system powered by your database",
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


# ================= SVD RECOMMEND =================
@app.get("/recommend/{user_id}/svd")
def recommend_svd_endpoint(user_id: str, top_n: int = Query(10, ge=1, le=100)):
    """Get SVD-based recommendations."""
    ensure_data_loaded()

    from models.svd_recommender import train_svd, recommend_svd
    from models.persistence import load_latest_model

    svd_model = load_latest_model("svd")
    if svd_model is None:
        svd_model = train_svd(R)

    seen = set(auditlog[auditlog["userId"] == user_id]["itemid"])

    recs = recommend_svd(svd_model, user_id, seen, top_n)
    recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")

    return {
        "user_id": user_id,
        "model": "svd",
        "recommendations": recs.to_dict(orient="records"),
    }


# ================= FP-GROWTH RECOMMEND =================
@app.get("/recommend/{user_id}/fpgrowth")
def recommend_fpgrowth_endpoint(user_id: str, top_n: int = Query(10, ge=1, le=100)):
    """Get FP-Growth association-rule-based recommendations."""
    ensure_data_loaded()

    from models.fpgrowth_recommender import train_fpgrowth, recommend_fpgrowth
    from models.persistence import load_latest_model

    rules = load_latest_model("fpgrowth_rules")
    if rules is None:
        _, rules = train_fpgrowth(auditlog)

    seen = set(auditlog[auditlog["userId"] == user_id]["itemid"])
    recs = recommend_fpgrowth(rules, seen, top_n)
    recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")

    return {
        "user_id": user_id,
        "model": "fpgrowth",
        "recommendations": recs.to_dict(orient="records"),
    }


# ================= SIMILARITY RECOMMEND =================
@app.get("/recommend/{user_id}/similarity")
def recommend_similarity_endpoint(
    user_id: str,
    top_n: int = Query(10, ge=1, le=100),
    metric: str = Query("cosine", regex="^(cosine|pearson)$"),
):
    """Get item-item similarity-based recommendations."""
    ensure_data_loaded()

    from models.similarity_recommender import compute_similarity, recommend_similarity

    sim_matrix = compute_similarity(R, mode="item", metric=metric)
    recs = recommend_similarity(user_id, R, sim_matrix, mode="item", top_n=top_n)
    recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")

    return {
        "user_id": user_id,
        "model": f"similarity_{metric}",
        "recommendations": recs.to_dict(orient="records"),
    }


# ================= PIPELINE RECOMMEND =================
@app.get("/recommend/{user_id}/pipeline")
def recommend_pipeline_endpoint(user_id: str, top_n: int = Query(10, ge=1, le=100)):
    """Get Multi-Stage Machine Learning (LightGBM) recommendations."""
    ensure_data_loaded()

    from pipeline.serve import recommend_pipeline
    
    recs = recommend_pipeline(user_id, auditlog, content, top_n)
    
    return {
        "user_id": user_id,
        "model": "multi_stage_pipeline",
        "recommendations": recs.to_dict(orient="records"),
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
