"""
Feature Engineering Module for the Multi-Stage Pipeline.

Extracts historical behavior features for users and popularity/trend
features for items based on a specific cut-off date (to avoid data leaks).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import log


def build_user_features(auditlog: pd.DataFrame) -> pd.DataFrame:
    """
    Build user features from historical interactions.
    
    Args:
        auditlog: Interaction data (must only contain events BEFORE the split date).
    
    Returns:
        DataFrame with index = userId and columns for features.
    """
    if auditlog.empty:
        return pd.DataFrame(columns=["user_interaction_count", "user_avg_rating", "user_std_rating"])

    # Base aggregations
    user_stats = auditlog.groupby("userId").agg(
        user_interaction_count=("itemid", "count"),
        user_avg_rating=("rating", "mean"),
        user_std_rating=("rating", "std")
    ).fillna(0)
    
    return user_stats


def build_item_features(auditlog: pd.DataFrame, content: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build item features (popularity, trends, content metadata).
    
    Args:
        auditlog: Interaction data (BEFORE split date).
        content: Metadata for all items.
        current_date: The "now" timestamp used to calculate recency/trends.
        
    Returns:
        DataFrame with index = itemid and columns for features.
    """
    if auditlog.empty:
        item_stats = pd.DataFrame(index=content["itemid"].unique())
        item_stats["item_popularity_all_time"] = 0
        item_stats["item_popularity_7d"] = 0
        item_stats["item_avg_rating"] = 0.0
    else:
        # All time popularity
        stats_all = auditlog.groupby("itemid").agg(
            item_popularity_all_time=("userId", "count"),
            item_avg_rating=("rating", "mean")
        )

        # 7-day trend (interactions in the 7 days prior to current_date)
        cutoff_7d = current_date - pd.Timedelta(days=7)
        if "createdAt" in auditlog.columns:
            recent_logs = auditlog[pd.to_datetime(auditlog["createdAt"], utc=True) >= cutoff_7d]
        else:
            # Fallback if no timestamps available
            recent_logs = auditlog

        stats_7d = recent_logs.groupby("itemid").agg(
            item_popularity_7d=("userId", "count")
        )

        item_stats = stats_all.join(stats_7d, how="left").fillna(0)

    # Content features
    content_features = content.set_index("itemid")[["type"]].copy()
    
    # One-hot encode type (MOVIE vs SERIES)
    if "type" in content_features.columns:
        content_features["is_movie"] = (content_features["type"] == "MOVIE").astype(int)
        content_features["is_series"] = (content_features["type"] == "SERIES").astype(int)
        content_features = content_features.drop(columns=["type"])

    # Merge stats and content
    features = item_stats.join(content_features, how="right").fillna(0)
    
    return features


def build_candidate_features(
    candidates_df: pd.DataFrame, 
    user_features: pd.DataFrame, 
    item_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine user, item, and cross features for a set of candidate pairs.
    
    Args:
        candidates_df: DataFrame with ['userId', 'itemid', 'label' (optional)]
        user_features: DataFrame from build_user_features
        item_features: DataFrame from build_item_features
        
    Returns:
        DataFrame ready for LightGBM training/inference.
    """
    # Merge User features
    df = candidates_df.join(user_features, on="userId", how="left")
    
    # Cold users get 0 for historical stats
    df["user_interaction_count"] = df["user_interaction_count"].fillna(0)
    df["user_avg_rating"] = df["user_avg_rating"].fillna(df["user_avg_rating"].mean() if not df["user_avg_rating"].isna().all() else 3.0)
    df["user_std_rating"] = df["user_std_rating"].fillna(0)

    # Merge Item features
    df = df.join(item_features, on="itemid", how="left")
    
    # Cold items get 0 for popularity
    df["item_popularity_all_time"] = df["item_popularity_all_time"].fillna(0)
    df["item_popularity_7d"] = df["item_popularity_7d"].fillna(0)
    df["item_avg_rating"] = df["item_avg_rating"].fillna(df["item_avg_rating"].mean() if not df["item_avg_rating"].isna().all() else 3.0)
    df["is_movie"] = df["is_movie"].fillna(0)
    df["is_series"] = df["is_series"].fillna(0)

    # Cross Features
    # User's deviation from item avg rating (if they rate high generically, maybe they will rate this high)
    df["user_item_rating_diff"] = df["user_avg_rating"] - df["item_avg_rating"]

    return df
