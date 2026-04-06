"""
FP-Growth (Frequent Pattern Mining) based recommendation.

Converts user-item interactions into "transaction baskets" and mines
association rules to find co-occurring items.
Recommendation: "Users who interacted with item X also interacted with Y."
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from utils.logger import log


# ─── Configuration ──────────────────────────────────────────
MIN_SUPPORT = 0.01        # Minimum support for frequent itemsets
MIN_CONFIDENCE = 0.1      # Minimum confidence for association rules
MIN_LIFT = 1.0            # Minimum lift
# ────────────────────────────────────────────────────────────


def _build_transactions(auditlog: pd.DataFrame) -> list[list[str]]:
    """
    Group interactions by user to form transaction baskets.
    Each basket = set of items a single user interacted with.
    """
    transactions = (
        auditlog.groupby("userId")["itemid"]
        .apply(list)
        .tolist()
    )
    log(f"FP-Growth: {len(transactions)} user baskets created")
    return transactions


def train_fpgrowth(auditlog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mine frequent itemsets and association rules.

    Args:
        auditlog: DataFrame with [userId, itemid, rating]

    Returns:
        (frequent_itemsets_df, rules_df)
    """
    # Only use positive interactions (rating > 0)
    positive = auditlog[auditlog["rating"] > 0]
    transactions = _build_transactions(positive)

    # One-hot encode transactions
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    log(f"FP-Growth: transaction matrix shape {df_encoded.shape}")

    # Mine frequent itemsets
    freq_itemsets = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    log(f"FP-Growth: {len(freq_itemsets)} frequent itemsets found")

    if freq_itemsets.empty:
        log("FP-Growth: No frequent itemsets found. Try lowering min_support.")
        return freq_itemsets, pd.DataFrame()

    # Generate association rules
    rules = association_rules(
        freq_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
        num_itemsets=len(freq_itemsets),
    )

    # Filter by lift
    rules = rules[rules["lift"] >= MIN_LIFT]
    rules = rules.sort_values("lift", ascending=False)

    log(f"FP-Growth: {len(rules)} association rules generated (lift >= {MIN_LIFT})")

    return freq_itemsets, rules


def recommend_fpgrowth(
    rules: pd.DataFrame,
    user_items: set[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Recommend items based on association rules.

    For each rule where the antecedent matches the user's history,
    score the consequent items by confidence × lift.

    Args:
        rules:      Association rules DataFrame from train_fpgrowth
        user_items: Set of item IDs the user has interacted with
        top_n:      Number of recommendations

    Returns:
        DataFrame with columns [itemid, confidence, lift, score]
    """
    if rules.empty:
        log("FP-Growth: No rules available for recommendation")
        return pd.DataFrame(columns=["itemid", "confidence", "lift", "score"])

    scored = {}

    for _, rule in rules.iterrows():
        antecedent = set(rule["antecedents"])
        consequent = set(rule["consequents"])

        # Check if user has ALL antecedent items
        if antecedent.issubset(user_items):
            # Recommend consequent items the user hasn't seen
            new_items = consequent - user_items
            for item in new_items:
                score = rule["confidence"] * rule["lift"]
                if item not in scored or scored[item]["score"] < score:
                    scored[item] = {
                        "confidence": rule["confidence"],
                        "lift": rule["lift"],
                        "score": score,
                    }

    if not scored:
        log("FP-Growth: No matching rules for this user's history")
        return pd.DataFrame(columns=["itemid", "confidence", "lift", "score"])

    df = pd.DataFrame([
        {"itemid": iid, **vals}
        for iid, vals in scored.items()
    ])
    df = df.sort_values("score", ascending=False).head(top_n)

    log(f"FP-Growth: {len(df)} recommendations generated")
    return df


def evaluate_fpgrowth(
    rules: pd.DataFrame,
    auditlog: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Evaluate FP-Growth using leave-one-out style Precision@K and Recall@K.

    For each user, hide one item, recommend using the rest, check if the
    hidden item appears in top-K.
    """
    if rules.empty:
        return {"model_name": "FP-Growth", "precision_at_k": 0, "recall_at_k": 0, "hit_rate": 0}

    user_groups = auditlog.groupby("userId")["itemid"].apply(set).to_dict()

    hits = 0
    total_precision = 0.0
    total_recall = 0.0
    n_evaluated = 0

    for user_id, items in user_groups.items():
        if len(items) < 2:
            continue

        # Hide one random item
        items_list = list(items)
        hidden = items_list[-1]
        visible = set(items_list[:-1])

        recs = recommend_fpgrowth(rules, visible, top_n=k)
        if recs.empty:
            continue

        rec_items = set(recs["itemid"].values)
        hit = 1 if hidden in rec_items else 0
        hits += hit
        total_precision += hit / k
        total_recall += hit  # Recall = hit / 1 (only 1 hidden item)
        n_evaluated += 1

    if n_evaluated == 0:
        return {"model_name": "FP-Growth", "precision_at_k": 0, "recall_at_k": 0, "hit_rate": 0}

    metrics = {
        "model_name": "FP-Growth",
        "precision_at_k": total_precision / n_evaluated,
        "recall_at_k": total_recall / n_evaluated,
        "hit_rate": hits / n_evaluated,
        "n_evaluated": n_evaluated,
        "k": k,
    }

    log(f"FP-Growth Eval — P@{k}: {metrics['precision_at_k']:.4f}, "
        f"R@{k}: {metrics['recall_at_k']:.4f}, Hit Rate: {metrics['hit_rate']:.4f}")

    return metrics
