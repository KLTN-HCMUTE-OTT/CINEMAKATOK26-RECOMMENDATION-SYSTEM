"""
Item-Item and User-User similarity-based recommendation.

Supports:
  - Cosine similarity
  - Pearson correlation
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import log


def _pearson_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute row-wise Pearson correlation as a similarity matrix."""
    # Center each row (subtract mean)
    means = matrix.mean(axis=1, keepdims=True)
    centered = matrix - means

    # Cosine similarity on centered data = Pearson correlation
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normalized = centered / norms
    return normalized @ normalized.T


def compute_similarity(
    rating_matrix: pd.DataFrame,
    mode: str = "item",
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Compute a similarity matrix.

    Args:
        rating_matrix:  Users × Items pivot table
        mode:           "item" (item-item) or "user" (user-user)
        metric:         "cosine" or "pearson"

    Returns:
        DataFrame — similarity matrix with appropriate labels
    """
    if mode == "item":
        data = rating_matrix.values.T  # items × users
        labels = rating_matrix.columns
    else:
        data = rating_matrix.values    # users × items
        labels = rating_matrix.index

    if metric == "cosine":
        sim = cosine_similarity(data)
    elif metric == "pearson":
        sim = _pearson_similarity(data)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'pearson'.")

    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    log(f"Similarity matrix ({mode}, {metric}): shape {sim_df.shape}")
    return sim_df


def recommend_similarity(
    user_id: str,
    rating_matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    mode: str = "item",
    top_k_neighbors: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate top-N recommendations using similarity-based CF.

    Args:
        user_id:          Target user
        rating_matrix:    Users × Items matrix
        sim_matrix:       Pre-computed similarity matrix
        mode:             "item" or "user"
        top_k_neighbors:  Number of neighbors to consider
        top_n:            Number of recommendations

    Returns:
        DataFrame with [itemid, predicted_rating]
    """
    if user_id not in rating_matrix.index:
        log(f"User {user_id} not found in rating matrix")
        return pd.DataFrame(columns=["itemid", "predicted_rating"])

    user_ratings = rating_matrix.loc[user_id]
    seen_items = set(user_ratings[user_ratings > 0].index)
    unseen_items = [i for i in rating_matrix.columns if i not in seen_items]

    predictions = []

    if mode == "item":
        # Item-Item CF
        for item in unseen_items:
            if item not in sim_matrix.index:
                continue

            # Get similarities to items the user has rated
            sims = sim_matrix.loc[item, list(seen_items)]
            top_sims = sims.nlargest(top_k_neighbors)
            top_sims = top_sims[top_sims > 0]

            if top_sims.empty:
                continue

            # Weighted average of user's ratings for similar items
            weights = top_sims.values
            ratings = user_ratings[top_sims.index].values
            pred = np.dot(weights, ratings) / weights.sum()
            predictions.append((item, pred))

    else:
        # User-User CF
        if user_id not in sim_matrix.index:
            log(f"User {user_id} not in similarity matrix")
            return pd.DataFrame(columns=["itemid", "predicted_rating"])

        user_sims = sim_matrix.loc[user_id].drop(user_id)
        top_users = user_sims.nlargest(top_k_neighbors)
        top_users = top_users[top_users > 0]

        if top_users.empty:
            return pd.DataFrame(columns=["itemid", "predicted_rating"])

        for item in unseen_items:
            # Get ratings from similar users for this item
            neighbor_ratings = rating_matrix.loc[top_users.index, item]
            rated = neighbor_ratings[neighbor_ratings > 0]

            if rated.empty:
                continue

            weights = top_users[rated.index].values
            pred = np.dot(weights, rated.values) / weights.sum()
            predictions.append((item, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(predictions[:top_n], columns=["itemid", "predicted_rating"])
    log(f"Similarity ({mode}): {len(df)} recommendations for user {user_id}")
    return df


def evaluate_similarity(
    rating_matrix: pd.DataFrame,
    sim_matrix: pd.DataFrame,
    mode: str = "item",
    test_ratio: float = 0.2,
) -> dict:
    """
    Evaluate similarity-based model using random masking.

    For each user, hide `test_ratio` of their rated items and
    measure RMSE on predictions for those items.
    """
    np.random.seed(42)
    y_true, y_pred = [], []

    for user_id in rating_matrix.index:
        user_ratings = rating_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index.tolist()

        if len(rated_items) < 3:
            continue

        n_test = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, size=n_test, replace=False)

        for item in test_items:
            true_rating = user_ratings[item]
            seen = set(rated_items) - {item}

            if mode == "item" and item in sim_matrix.index:
                sims = sim_matrix.loc[item, list(seen)]
                top_sims = sims.nlargest(20)
                top_sims = top_sims[top_sims > 0]
                if not top_sims.empty:
                    w = top_sims.values
                    r = user_ratings[top_sims.index].values
                    pred = np.dot(w, r) / w.sum()
                    y_true.append(true_rating)
                    y_pred.append(pred)

    if not y_true:
        return {"model_name": f"Similarity ({mode})", "rmse": float("inf"), "mae": float("inf")}

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    log(f"Similarity ({mode}) Eval — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return {
        "model_name": f"Similarity ({mode})",
        "rmse": rmse,
        "mae": mae,
        "n_samples": len(y_true),
    }
