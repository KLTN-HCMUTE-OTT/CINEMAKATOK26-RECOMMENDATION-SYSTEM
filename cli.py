"""
CLI Interface for the Recommendation System.

Commands:
    train       Train the Multi-Stage Machine Learning (LightGBM) model
    recommend   Get top-N recommendations for a user
    eda         Generate EDA visualizations
    models      List saved models

Usage:
    python cli.py train
    python cli.py recommend --user-id <UUID> --top-n 10
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
def train(val_days: int = typer.Option(0, help="Days to hold out for validation split")):
    """Run the modern Multi-Stage ML training pipeline (Candidate Gen + Feature Eng + LightGBM)."""
    from pipeline.train import run_training_pipeline
    auditlog, content, _ = _load_all()
    run_training_pipeline(auditlog, content, val_days)


# ─── RECOMMEND ─────────────────────────────────────────────────
@app.command()
def recommend(
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
