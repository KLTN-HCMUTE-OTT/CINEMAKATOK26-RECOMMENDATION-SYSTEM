"""
Training Pipeline Orchestrator.

1. Chronological Train/Val split of audit logs.
2. Candidate generation on Train.
3. Feature construction (with Negative Sampling).
4. Train LightGBM Ranker.
5. Save models and feature transformers.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, ndcg_score

from utils.logger import log
from utils.timer import Timer
from models.svd_recommender import train_svd, recommend_svd
from models.persistence import save_model
from pipeline.features import build_user_features, build_item_features, build_candidate_features

def run_training_pipeline(auditlog: pd.DataFrame, content: pd.DataFrame, val_days: int = 7):
    """
    Run the full multi-stage training pipeline.
    """
    log("\n" + "=" * 60)
    log("STARTING MULTI-STAGE TRAINING PIPELINE")
    log("=" * 60)

    # Ensure timestamp exists for time-based split
    if "createdAt" not in auditlog.columns:
        log("No 'createdAt' found in auditlog! Adding synthetic timestamps for simulation...")
        # Simulate timestamps if missing (for the sake of the project structure)
        auditlog["createdAt"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(auditlog))
    else:
        auditlog["createdAt"] = pd.to_datetime(auditlog["createdAt"], utc=True)

    # 1. TIME-BASED SPLIT
    with Timer("1. Chronological Split"):
        max_date = auditlog["createdAt"].max()
        split_date = max_date - pd.Timedelta(days=val_days)
        
        train_logs = auditlog[auditlog["createdAt"] <= split_date].copy()
        val_logs   = auditlog[auditlog["createdAt"] > split_date].copy()
        
        log(f"Train interactions: {len(train_logs)} (<= {split_date.date()})")
        log(f"Val interactions:   {len(val_logs)} (> {split_date.date()})")

    if len(train_logs) == 0:
        log("Not enough training data! Decrease val_days.")
        return

    # Build matrix for Candidate Generation
    # We only train Candidate Generators on TRAIN data
    R_train = train_logs.pivot_table(index="userId", columns="itemid", values="rating", fill_value=0)
    
    # Ensure all items match the content catalog
    for col in content["itemid"]:
        if col not in R_train.columns:
            R_train[col] = 0

    # 2. TRAIN CANDIDATE GENERATOR (SVD used as primary generator)
    with Timer("2. Train Candidate Generator (SVD)"):
        svd_model = train_svd(R_train, n_factors=50)
        save_model(svd_model, "pipeline_svd_generator")
        
        # Also compute popularity for Cold-Start candidates
        pop_counts = train_logs["itemid"].value_counts().reset_index()
        pop_counts.columns = ["itemid", "score"]
        save_model(pop_counts, "pipeline_pop_generator")

    # 3. FEATURE ENGINEERING & NEGATIVE SAMPLING
    with Timer("3. Feature Construction & Sampling"):
        # Features MUST only be built from train_logs to prevent leakage
        user_feats = build_user_features(train_logs)
        item_feats = build_item_features(train_logs, content, split_date) # pretend split_date is "now"
        
        # Build Positive samples from train_logs
        pos_df = train_logs[["userId", "itemid"]].copy()
        pos_df["label"] = 1
        
        # Build Negative samples (Items users never interacted with)
        # For each user, sample N random items they haven't seen in train
        neg_samples = []
        all_items = set(content["itemid"].unique())
        
        user_groups = train_logs.groupby("userId")["itemid"].apply(set)
        for uid, seen_items in user_groups.items():
            unseen = list(all_items - seen_items)
            # Sample 4 negatives for every positive (ratio 1:4 bias correction)
            n_neg = len(seen_items) * 4
            if n_neg > 0 and unseen:
                sampled = np.random.choice(unseen, size=min(n_neg, len(unseen)), replace=False)
                for iid in sampled:
                    neg_samples.append({"userId": uid, "itemid": iid, "label": 0})
        
        neg_df = pd.DataFrame(neg_samples)
        train_candidates = pd.concat([pos_df, neg_df], ignore_index=True)
        
        # Merge features into training dataset
        X_train = build_candidate_features(train_candidates, user_feats, item_feats)
        y_train = X_train.pop("label")
        X_train = X_train.drop(columns=["userId", "itemid"]) # Drop IDs, keep only features
        
        log(f"Training set: {len(X_train)} rows ({y_train.sum()} pos, {len(y_train) - y_train.sum()} neg)")

    # 4. LIGHTGBM RANKER TRAINING
    with Timer("4. Train LightGBM Ranker"):
        # We model this as a binary classification problem (predicting interaction probability)
        lgb_train = lgb.Dataset(X_train, y_train)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7,
            "feature_fraction": 0.8,
            "verbose": -1,
            "seed": 42
        }
        
        ranker_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=100
        )
        
        save_model(ranker_model, "pipeline_lgb_ranker")
        
        # Log feature importance
        importances = ranker_model.feature_importance(importance_type='gain')
        feat_names = X_train.columns
        for name, imp in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]:
            log(f"  Feature '{name}': gain={imp:.1f}")

    log("=" * 60)
    log("MULTI-STAGE PIPELINE TRAINED & SAVED")
    log("=" * 60 + "\n")
