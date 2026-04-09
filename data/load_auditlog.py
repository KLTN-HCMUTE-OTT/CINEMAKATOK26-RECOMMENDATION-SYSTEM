"""
Load user interaction data from the audit_logs table.

Schema:
    userId, sessionId, action, resourceType, resourceId, signalWeight, metadata

signalWeight: 2=strong, 1=medium, -1=negative, 0=ignored
resourceType: MOVIE | SERIES
"""
import pandas as pd
from data.db import engine_audit
from utils.logger import log

# Actions that represent meaningful user–content interactions
USER_ACTIONS = [
    # Positive
    'PLAY_MOVIE', 'PLAY_EPISODE_OF_SERIES',
    'LIKE_MOVIE', 'LIKE_SERIES',
    'ADD_MOVIE_TO_WATCHLIST', 'ADD_SERIES_TO_WATCHLIST',
    # Negative (kept for potential use; filtered by signalWeight)
    'UNLIKE_MOVIE', 'UNLIKE_SERIES',
    'REMOVE_MOVIE_FROM_WATCHLIST', 'REMOVE_SERIES_FROM_WATCHLIST',
]

AUDITLOG_QUERY = """
SELECT
    "userId",
    "sessionId",
    action,
    "resourceType",
    "resourceId",
    "signalWeight",
    "createdAt"
FROM audit_logs
WHERE "resourceType" IN ('MOVIE', 'SERIES')
  AND "resourceId" IS NOT NULL
  AND "signalWeight" != 0
ORDER BY "createdAt" DESC
"""


def load_auditlog(include_negative: bool = False) -> pd.DataFrame:
    """
    Load and clean user interaction data from PostgreSQL.

    Args:
        include_negative: If True, keep signalWeight=-1 rows (unlikes/removes).
                          If False (default), only keep positive signals (1, 2).

    Returns:
        DataFrame with columns: userId, itemid, rating, action, resourceType
    """
    df = pd.read_sql(AUDITLOG_QUERY, engine_audit)
    log(f"Raw audit log rows loaded: {len(df)}")

    # Rename resourceId → itemid for downstream compatibility
    df = df.rename(columns={"resourceId": "itemid"})

    # Ensure UUIDs are strings
    df["itemid"] = df["itemid"].astype(str)
    df["userId"] = df["userId"].astype(str)

    # Filter out negative signals unless explicitly requested
    if not include_negative:
        df = df[df["signalWeight"] > 0]

    # Use signalWeight as the rating signal
    # Normalize: signalWeight 1 → rating 3, signalWeight 2 → rating 5
    df["rating"] = df["signalWeight"].map({1: 3, 2: 5, -1: 1})
    df = df.dropna(subset=["rating"])

    # Deduplicate: keep the strongest signal per user-item pair
    df = (
        df.sort_values("rating", ascending=False)
        .drop_duplicates(subset=["userId", "itemid"], keep="first")
    )

    log(f"Cleaned interactions: {len(df)} (users={df['userId'].nunique()}, items={df['itemid'].nunique()})")

    return df[["userId", "itemid", "rating", "action", "resourceType", "createdAt"]]
