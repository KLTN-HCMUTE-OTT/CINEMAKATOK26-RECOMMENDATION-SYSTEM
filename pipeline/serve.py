"""
Serving Pipeline Orchestrator (Inference).

Applies the two-stage Multi-Stage ML RecSys:
1. Candidate Generation (from SVD or Popularity for cold start).
2. Rank generated candidates using the trained LightGBM model.
"""
import pandas as pd
import numpy as np

from utils.logger import log
from pipeline.features import build_user_features, build_item_features, build_candidate_features
from models.persistence import load_latest_model
from models.svd_recommender import recommend_svd


def recommend_pipeline(user_id: str, auditlog: pd.DataFrame, content: pd.DataFrame, top_n: int = 10):
    """
    Generate recommendations using the 2-Stage Pipeline (SVD + LightGBM).
    
    Args:
        user_id: Target user UUID
        auditlog: Historical interactions (used for real-time feature extraction)
        content: Metadata catalog
        top_n: Number of final items to return
        
    Returns:
        DataFrame top N items sorted by LightGBM probability.
    """
    # Load all pipeline artifacts
    svd_model = load_latest_model("pipeline_svd_generator")
    pop_model = load_latest_model("pipeline_pop_generator") # used for Cold Start
    ranker    = load_latest_model("pipeline_lgb_ranker")
    
    if ranker is None:
        log("No trained pipeline found! Run 'train-pipeline' first.")
        return pd.DataFrame()
        
    # Get user's seen items
    user_history = auditlog[auditlog["userId"] == user_id]
    seen_items = set(user_history["itemid"])
    
    # -----------------------------------------------------
    # STAGE 1: CANDIDATE GENERATION (~200 items)
    # -----------------------------------------------------
    candidates_pool = []
    
    # Check if user is Cold Start (no history in SVD)
    if svd_model and user_id in svd_model["user_means"]:
        log(f"Stage 1: Generating candidates using SVD for known user {user_id}")
        svd_recs = recommend_svd(svd_model, user_id, seen_items, top_n=200)
        candidates_pool = svd_recs["itemid"].tolist()
    else:
        log(f"Stage 1: User {user_id} is Cold-Start. Using Popularity candidates.")
        if pop_model is not None:
            # Filter out seen and take top 200 popular
            unseen_pop = pop_model[~pop_model["itemid"].isin(seen_items)]
            candidates_pool = unseen_pop.head(200)["itemid"].tolist()
            
    if not candidates_pool:
        log("No candidates could be generated.")
        return pd.DataFrame()

    # -----------------------------------------------------
    # STAGE 2: FEATURE EXTRACTION (Just-in-Time)
    # -----------------------------------------------------
    # 1. Build User Profile (aggregate their actual history)
    # We pass the full user history to extract features
    user_feats = build_user_features(user_history)
    
    # 2. Build Item Profiles
    # Filter content to only candidate items to speed up computation
    cand_auditlog = auditlog[auditlog["itemid"].isin(candidates_pool)]
    cand_content  = content[content["itemid"].isin(candidates_pool)]
    current_date  = pd.Timestamp.now(tz="UTC")
    
    if "createdAt" not in auditlog.columns:
        # Fallback date if no real timestamps
        current_date = pd.Timestamp("2026-01-01", tz="UTC") 
        
    item_feats = build_item_features(cand_auditlog, cand_content, current_date)
    
    # 3. Combine pairs
    cand_df = pd.DataFrame({"userId": user_id, "itemid": candidates_pool})
    X_infer = build_candidate_features(cand_df, user_feats, item_feats)
    
    # Ensure columns match training order exactly
    feature_names = ranker.feature_name()
    
    # If a feature is missing (e.g. they watched no movies so 'is_movie' didn't generate), fill with 0
    for col in feature_names:
        if col not in X_infer.columns:
            X_infer[col] = 0
            
    # Re-order and drop IDs
    X_ready = X_infer[feature_names]
    
    # -----------------------------------------------------
    # STAGE 3: RANKING & SCORING
    # -----------------------------------------------------
    log(f"Stage 2: Ranking {len(X_ready)} candidates via LightGBM...")
    scores = ranker.predict(X_ready)
    
    # Attach scores and sort
    cand_df["lgb_score"] = scores
    final_recs = cand_df.sort_values(by="lgb_score", ascending=False).head(top_n)
    
    # Merge rich metadata for output
    final_recs = final_recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")
    
    return final_recs
