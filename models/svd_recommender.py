"""
SVD-based Collaborative Filtering using scipy's truncated SVD.

Decomposes the user-item rating matrix with SVD and generates
top-N recommendations from the reconstructed matrix.
"""
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.logger import log


# ─── Configuration ──────────────────────────────────────────
SVD_N_FACTORS = 50   # Number of latent factors (k)
# ────────────────────────────────────────────────────────────


def train_svd(rating_matrix: pd.DataFrame, n_factors: int = SVD_N_FACTORS) -> dict:
    """
    Train an SVD model by decomposing the user-item rating matrix.

    Args:
        rating_matrix: Users × Items pivot table (0 = unrated)
        n_factors:     Number of latent factors

    Returns:
        dict with keys: U, sigma, Vt, predicted, user_index, item_index, user_means
    """
    # Center by user mean (only over rated items)
    R = rating_matrix.values.astype(float)
    user_means = []
    for i in range(R.shape[0]):
        rated = R[i][R[i] > 0]
        user_means.append(rated.mean() if len(rated) > 0 else 0)
    user_means = np.array(user_means)

    R_centered = R.copy()
    for i in range(R.shape[0]):
        mask = R[i] > 0
        R_centered[i][mask] -= user_means[i]

    # Truncated SVD
    k = min(n_factors, min(R.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(R_centered), k=k)

    # Reconstruct predictions
    sigma_diag = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma_diag), Vt) + user_means.reshape(-1, 1)

    predicted_df = pd.DataFrame(
        predicted,
        index=rating_matrix.index,
        columns=rating_matrix.columns,
    )

    log(f"SVD trained — {k} factors, matrix shape {R.shape}")

    return {
        "U": U,
        "sigma": sigma,
        "Vt": Vt,
        "predicted": predicted_df,
        "user_means": user_means,
    }


def evaluate_svd(auditlog: pd.DataFrame, rating_matrix: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Evaluate SVD using train/test split on the interaction data.

    Returns:
        dict with rmse, mae, and sample count
    """
    # Build user-item pairs
    user_item = auditlog.groupby(["userId", "itemid"])["rating"].max().reset_index()

    train_df, test_df = train_test_split(user_item, test_size=test_size, random_state=42)

    # Build training rating matrix
    R_train = train_df.pivot_table(index="userId", columns="itemid", values="rating", fill_value=0)

    # Ensure all items from full matrix are present
    for col in rating_matrix.columns:
        if col not in R_train.columns:
            R_train[col] = 0
    for idx in rating_matrix.index:
        if idx not in R_train.index:
            R_train.loc[idx] = 0
    R_train = R_train.reindex(index=rating_matrix.index, columns=rating_matrix.columns, fill_value=0)

    # Train SVD on training data
    model = train_svd(R_train)
    predicted = model["predicted"]

    # Evaluate on test data
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        uid, iid, rating = row["userId"], row["itemid"], row["rating"]
        if uid in predicted.index and iid in predicted.columns:
            pred = predicted.loc[uid, iid]
            if not np.isnan(pred):
                y_true.append(rating)
                y_pred.append(np.clip(pred, 1, 5))

    if not y_true:
        return {"model_name": "SVD", "rmse": float("inf"), "mae": float("inf")}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    log(f"SVD Evaluation — RMSE: {rmse:.4f}, MAE: {mae:.4f}, samples: {len(y_true)}")

    return {
        "model_name": "SVD",
        "rmse": rmse,
        "mae": mae,
        "n_samples": len(y_true),
    }


def recommend_svd(
    model: dict,
    user_id: str,
    seen_item_ids: set[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate top-N recommendations for a user from the SVD model.

    Args:
        model:          dict returned by train_svd()
        user_id:        Target user
        seen_item_ids:  Items the user has already interacted with
        top_n:          Number of recommendations

    Returns:
        DataFrame with columns [itemid, predicted_rating]
    """
    predicted = model["predicted"]

    if user_id not in predicted.index:
        log(f"User {user_id} not found in SVD predictions")
        return pd.DataFrame(columns=["itemid", "predicted_rating"])

    scores = predicted.loc[user_id].drop(labels=list(seen_item_ids), errors="ignore")
    scores = scores.clip(lower=1, upper=5)
    top = scores.nlargest(top_n).reset_index()
    top.columns = ["itemid", "predicted_rating"]

    log(f"SVD: {len(top)} recommendations for user {user_id}")
    return top
