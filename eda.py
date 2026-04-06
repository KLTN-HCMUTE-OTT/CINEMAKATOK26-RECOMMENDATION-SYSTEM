"""
Exploratory Data Analysis (EDA) and Visualization.

Generates plots for rating distribution, user activity, item popularity,
sparsity heatmap, and saves them to data/plots/.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from utils.logger import log

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "data", "plots")


def _ensure_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_rating_distribution(auditlog: pd.DataFrame):
    """Bar chart of rating counts."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    auditlog["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_title("Rating Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Rating (signalWeight mapped)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "rating_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def plot_user_activity(auditlog: pd.DataFrame, top_n: int = 30):
    """Horizontal bar chart of most active users."""
    _ensure_dir()
    user_counts = auditlog["userId"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    user_counts.plot(kind="barh", ax=ax, color="#55A868", edgecolor="black")
    ax.set_title(f"Top {top_n} Most Active Users", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Interactions")
    ax.set_ylabel("User ID")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "user_activity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def plot_item_popularity(auditlog: pd.DataFrame, content: pd.DataFrame, top_n: int = 20):
    """Horizontal bar chart of most popular items."""
    _ensure_dir()
    item_counts = auditlog["itemid"].value_counts().head(top_n)

    # Try to map item IDs to titles
    if "title" in content.columns:
        id_to_title = content.set_index("itemid")["title"].to_dict()
        labels = [id_to_title.get(iid, iid[:8]) for iid in item_counts.index]
    else:
        labels = [iid[:8] for iid in item_counts.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, item_counts.values, color="#C44E52", edgecolor="black")
    ax.set_title(f"Top {top_n} Most Popular Items", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Interactions")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "item_popularity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def plot_sparsity_heatmap(rating_matrix: pd.DataFrame, max_users: int = 50, max_items: int = 50):
    """Heatmap showing a sample of the user-item interaction matrix."""
    _ensure_dir()
    sample = rating_matrix.iloc[:max_users, :max_items]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        sample.astype(float),
        cmap="YlOrRd",
        ax=ax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Rating"},
    )
    ax.set_title(f"Rating Matrix Sparsity (sample {sample.shape})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Items")
    ax.set_ylabel("Users")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sparsity_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def plot_interactions_per_user(auditlog: pd.DataFrame):
    """Histogram showing distribution of interactions per user."""
    _ensure_dir()
    interactions = auditlog.groupby("userId").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(interactions.values, bins=30, color="#8172B2", edgecolor="black", alpha=0.8)
    ax.set_title("Interactions per User", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Interactions")
    ax.set_ylabel("Number of Users")
    ax.axvline(interactions.mean(), color="red", linestyle="--", label=f"Mean: {interactions.mean():.1f}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "interactions_per_user.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def plot_action_breakdown(auditlog: pd.DataFrame):
    """Pie chart showing breakdown of action types."""
    _ensure_dir()
    if "action" not in auditlog.columns:
        return

    action_counts = auditlog["action"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        action_counts.values,
        labels=action_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(action_counts)),
    )
    ax.set_title("Action Type Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "action_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Plot saved: {path}")


def run_full_eda(auditlog: pd.DataFrame, content: pd.DataFrame, rating_matrix: pd.DataFrame):
    """Run all EDA plots."""
    log("\n" + "=" * 60)
    log("RUNNING EDA — GENERATING PLOTS")
    log("=" * 60)

    plot_rating_distribution(auditlog)
    plot_user_activity(auditlog)
    plot_item_popularity(auditlog, content)
    plot_sparsity_heatmap(rating_matrix)
    plot_interactions_per_user(auditlog)
    plot_action_breakdown(auditlog)

    log(f"\nAll plots saved to: {PLOTS_DIR}")
    log("=" * 60 + "\n")
