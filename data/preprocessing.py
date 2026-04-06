"""
Data preprocessing utilities.

Handles:
- Missing value analysis and cleaning
- Rating matrix construction
- Data quality reporting
"""
import pandas as pd
import numpy as np
from utils.logger import log


def analyze_data_quality(auditlog: pd.DataFrame, content: pd.DataFrame) -> dict:
    """
    Print a data quality report for the loaded datasets.

    Returns:
        dict with summary statistics
    """
    log("\n" + "=" * 60)
    log("DATA QUALITY REPORT")
    log("=" * 60)

    # --- Auditlog ---
    log("\n[Auditlog]")
    log(f"  Total interactions:  {len(auditlog)}")
    log(f"  Unique users:        {auditlog['userId'].nunique()}")
    log(f"  Unique items:        {auditlog['itemid'].nunique()}")
    log(f"  Missing userId:      {auditlog['userId'].isna().sum()}")
    log(f"  Missing itemid:      {auditlog['itemid'].isna().sum()}")
    log(f"  Rating distribution:")
    for rating, count in auditlog["rating"].value_counts().sort_index().items():
        log(f"    {rating}: {count}")

    # --- Content ---
    log("\n[Content]")
    log(f"  Total items:         {len(content)}")
    for col in ["title", "genres", "director", "cast", "keywords", "description"]:
        if col in content.columns:
            missing = content[col].isna().sum()
            pct = missing / len(content) * 100
            log(f"  Missing {col:12s}: {missing} ({pct:.1f}%)")

    # --- Overlap ---
    audit_items = set(auditlog["itemid"].unique())
    content_items = set(content["itemid"].unique())
    overlap = audit_items & content_items
    log(f"\n[Overlap]")
    log(f"  Items in auditlog:   {len(audit_items)}")
    log(f"  Items in content:    {len(content_items)}")
    log(f"  Items in both:       {len(overlap)}")
    log(f"  Audit-only items:    {len(audit_items - content_items)}")
    log(f"  Content-only items:  {len(content_items - audit_items)}")
    log("=" * 60 + "\n")

    return {
        "n_interactions": len(auditlog),
        "n_users": auditlog["userId"].nunique(),
        "n_items_audit": auditlog["itemid"].nunique(),
        "n_items_content": len(content),
        "n_overlap": len(overlap),
    }


def build_rating_matrix(auditlog: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user-item rating matrix from the auditlog.

    Args:
        auditlog: DataFrame with columns [userId, itemid, rating]

    Returns:
        Pivot table (users × items), missing entries filled with 0.
    """
    user_item = (
        auditlog.groupby(["userId", "itemid"])["rating"]
        .max()
        .reset_index()
    )

    R = user_item.pivot_table(
        index="userId",
        columns="itemid",
        values="rating",
        fill_value=0,
    )

    sparsity = (1 - R.astype(bool).sum().sum() / (R.shape[0] * R.shape[1])) * 100
    log(f"Rating matrix shape: {R.shape}  |  Sparsity: {sparsity:.2f}%")

    return R


def clean_and_merge(
    auditlog: pd.DataFrame,
    content: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean both datasets and keep only overlapping items.

    Returns:
        (cleaned_auditlog, cleaned_content)
    """
    # Keep only items present in both datasets
    common_items = set(auditlog["itemid"].unique()) & set(content["itemid"].unique())
    log(f"Keeping {len(common_items)} items present in both auditlog and content")

    auditlog = auditlog[auditlog["itemid"].isin(common_items)].copy()
    content = content[content["itemid"].isin(common_items)].copy()

    # Fill missing text fields
    text_cols = ["genres", "director", "cast", "keywords", "description"]
    for col in text_cols:
        if col in content.columns:
            content[col] = content[col].fillna("")

    return auditlog, content
