"""
CLI Interface for the Recommendation System.

Commands:
    train       Train a model (svd, fpgrowth, similarity)
    recommend   Get top-N recommendations for a user
    evaluate    Evaluate one or all models
    eda         Generate EDA visualizations
    models      List saved models

Usage:
    python cli.py train --model svd
    python cli.py recommend --user-id <UUID> --model svd --top-n 10
    python cli.py evaluate --model all
    python cli.py eda
    python cli.py models
"""
import typer
import pandas as pd
from typing import Optional

from data.load_auditlog import load_auditlog
from data.load_content import load_content
from data.preprocessing import analyze_data_quality, build_rating_matrix, clean_and_merge
from utils.logger import log
from utils.timer import Timer

app = typer.Typer(
    name="recsys",
    help="OTT Recommendation System CLI",
    add_completion=False,
)


# ─── Shared data loading ──────────────────────────────────────
def _load_all():
    """Load and preprocess all data from the database."""
    with Timer("Load data"):
        auditlog = load_auditlog()
        content = load_content()

    auditlog, content = clean_and_merge(auditlog, content)
    analyze_data_quality(auditlog, content)
    R = build_rating_matrix(auditlog)

    return auditlog, content, R


# ─── TRAIN ─────────────────────────────────────────────────────
@app.command()
def train(
    model: str = typer.Option("svd", help="Model to train: svd, fpgrowth, similarity"),
    save: bool = typer.Option(True, help="Save the trained model to disk"),
):
    """Train a recommendation model on the database data."""
    auditlog, content, R = _load_all()

    if model == "svd":
        from models.svd_recommender import train_svd
        from models.persistence import save_model

        with Timer("Train SVD"):
            svd_model = train_svd(R)

        if save:
            save_model(svd_model, "svd", {"n_interactions": len(auditlog)})

    elif model == "fpgrowth":
        from models.fpgrowth_recommender import train_fpgrowth
        from models.persistence import save_model

        with Timer("Train FP-Growth"):
            freq_itemsets, rules = train_fpgrowth(auditlog)

        log(f"Frequent itemsets: {len(freq_itemsets)}")
        log(f"Association rules: {len(rules)}")

        if save and not rules.empty:
            save_model(rules, "fpgrowth_rules", {"n_rules": len(rules)})

    elif model == "similarity":
        from models.similarity_recommender import compute_similarity
        from models.persistence import save_model

        with Timer("Compute Item-Item Similarity"):
            sim_matrix = compute_similarity(R, mode="item", metric="cosine")

        if save:
            save_model(sim_matrix, "similarity_item", {"shape": sim_matrix.shape})

    else:
        typer.echo(f"Unknown model: {model}. Options: svd, fpgrowth, similarity")
        raise typer.Exit(1)

    typer.echo(f"\n✅ Model '{model}' trained successfully!")


@app.command()
def train_pipeline(val_days: int = typer.Option(7, help="Days to hold out for validation split")):
    """Run the modern Multi-Stage ML training pipeline (Candidate Gen + Feature Eng + LightGBM)."""
    from pipeline.train import run_training_pipeline
    auditlog, content, _ = _load_all()
    run_training_pipeline(auditlog, content, val_days)


# ─── RECOMMEND ─────────────────────────────────────────────────
@app.command()
def recommend(
    user_id: str = typer.Option(..., "--user-id", "-u", help="User UUID"),
    model: str = typer.Option("svd", help="Model to use: svd, fpgrowth, similarity"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Number of recommendations"),
):
    """Get top-N recommendations for a user."""
    auditlog, content, R = _load_all()

    # Items this user has already seen
    seen = set(auditlog[auditlog["userId"] == user_id]["itemid"])

    if model == "svd":
        from models.svd_recommender import train_svd, recommend_svd
        from models.persistence import load_latest_model

        svd_model = load_latest_model("svd")
        if svd_model is None:
            log("No saved SVD model found. Training a new one...")
            svd_model = train_svd(R)

        recs = recommend_svd(svd_model, user_id, seen, top_n)
        recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")
        _print_recommendations(recs, "predicted_rating")

    elif model == "fpgrowth":
        from models.fpgrowth_recommender import train_fpgrowth, recommend_fpgrowth
        from models.persistence import load_latest_model

        rules = load_latest_model("fpgrowth_rules")
        if rules is None:
            log("No saved FP-Growth rules found. Training...")
            _, rules = train_fpgrowth(auditlog)

        recs = recommend_fpgrowth(rules, seen, top_n)
        recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")
        _print_recommendations(recs, "score")

    elif model == "similarity":
        from models.similarity_recommender import compute_similarity, recommend_similarity
        from models.persistence import load_latest_model

        sim_matrix = load_latest_model("similarity_item")
        if sim_matrix is None:
            log("No saved similarity matrix. Computing...")
            sim_matrix = compute_similarity(R, mode="item", metric="cosine")

        recs = recommend_similarity(user_id, R, sim_matrix, mode="item", top_n=top_n)
        recs = recs.merge(content[["itemid", "title", "type"]], on="itemid", how="left")
        _print_recommendations(recs, "predicted_rating")

    else:
        typer.echo(f"Unknown model: {model}. Options: svd, fpgrowth, similarity")
        raise typer.Exit(1)


@app.command()
def recommend_pipeline(
    user_id: str = typer.Option(..., "--user-id", "-u", help="User UUID"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Number of recommendations"),
):
    """Get top-N recommendations using the full Multi-Stage ML RecSys."""
    from pipeline.serve import recommend_pipeline
    auditlog, content, _ = _load_all()

    recs = recommend_pipeline(user_id, auditlog, content, top_n)
    _print_recommendations(recs, "lgb_score")


def _print_recommendations(df: pd.DataFrame, score_col: str):
    """Pretty-print a recommendations DataFrame."""
    if df.empty:
        typer.echo("No recommendations generated.")
        return

    typer.echo(f"\n{'Rank':<6}{'Title':<50}{'Type':<12}{score_col:<15}")
    typer.echo("-" * 83)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        title = str(row.get("title", "N/A"))[:48]
        rtype = str(row.get("type", "?"))
        score = row.get(score_col, 0)
        typer.echo(f"{i:<6}{title:<50}{rtype:<12}{score:<15.4f}")


# ─── EVALUATE ──────────────────────────────────────────────────
@app.command()
def evaluate(
    model: str = typer.Option("all", help="Model to evaluate: svd, fpgrowth, similarity, all"),
):
    """Evaluate recommendation model(s) with standard metrics."""
    auditlog, content, R = _load_all()
    results = []

    if model in ("svd", "all"):
        from models.svd_recommender import evaluate_svd
        with Timer("Evaluate SVD"):
            results.append(evaluate_svd(auditlog, R))

    if model in ("fpgrowth", "all"):
        from models.fpgrowth_recommender import train_fpgrowth, evaluate_fpgrowth
        with Timer("Evaluate FP-Growth"):
            _, rules = train_fpgrowth(auditlog)
            results.append(evaluate_fpgrowth(rules, auditlog))

    if model in ("similarity", "all"):
        from models.similarity_recommender import compute_similarity, evaluate_similarity
        with Timer("Evaluate Similarity"):
            sim_matrix = compute_similarity(R, mode="item", metric="cosine")
            results.append(evaluate_similarity(R, sim_matrix, mode="item"))

    # Summary table
    if results:
        typer.echo(f"\n{'Model':<25}{'RMSE':<12}{'MAE':<12}")
        typer.echo("-" * 49)
        for r in results:
            rmse = r.get("rmse", r.get("precision_at_k", "N/A"))
            mae = r.get("mae", r.get("recall_at_k", "N/A"))
            name = r["model_name"]
            if isinstance(rmse, float):
                typer.echo(f"{name:<25}{rmse:<12.4f}{mae:<12.4f}")
            else:
                typer.echo(f"{name:<25}{rmse:<12}{mae:<12}")


# ─── EDA ───────────────────────────────────────────────────────
@app.command()
def eda():
    """Generate EDA visualizations and save to data/plots/."""
    auditlog, content, R = _load_all()

    from eda import run_full_eda
    run_full_eda(auditlog, content, R)
    typer.echo("✅ EDA plots generated in data/plots/")


# ─── MODELS ────────────────────────────────────────────────────
@app.command()
def models():
    """List all saved models."""
    from models.persistence import list_saved_models

    saved = list_saved_models()
    if not saved:
        typer.echo("No saved models found.")
        return

    typer.echo(f"\n{'Filename':<45}{'Model':<20}{'Saved At':<20}")
    typer.echo("-" * 85)
    for m in saved:
        typer.echo(f"{m['filename']:<45}{m['model_name']:<20}{m.get('saved_at', '?'):<20}")


# ─── ENTRY POINT ───────────────────────────────────────────────
if __name__ == "__main__":
    app()
