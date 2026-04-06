# OTT Recommendation System

A multi-algorithm recommendation system for an OTT (Over-The-Top) content platform. It reads user interaction data from a PostgreSQL database and generates personalized movie/series recommendations.

## Features

| Feature | Description |
|---|---|
| **ALS Collaborative Filtering** | Matrix factorization via `implicit` |
| **SVD** | `surprise.SVD` with cross-validation |
| **FP-Growth** | Association-rule mining via `mlxtend` |
| **Item-Item Similarity** | Cosine / Pearson similarity |
| **Hybrid Model** | Weighted blend of CF + Content-Based + Popularity |
| **Content-Based** | TF-IDF on genres, cast, director, keywords |
| **EDA Visualizations** | Rating distribution, user activity, sparsity heatmap |
| **Model Persistence** | Save/load trained models with `joblib` |
| **CLI** | Full command-line interface via `typer` |
| **REST API** | FastAPI with endpoints for all models |

## Project Structure

```
recommendation_sytem/
├── api.py                  # FastAPI REST API
├── cli.py                  # Typer CLI interface
├── eda.py                  # EDA + visualizations
├── main.py                 # Original main script
├── config/
│   └── setting.py          # All configuration constants
├── data/
│   ├── db.py               # SQLAlchemy engine
│   ├── load_auditlog.py    # Load from audit_logs table
│   ├── load_content.py     # Load from content table
│   ├── preprocessing.py    # Cleaning, rating matrix, quality report
│   ├── plots/              # Generated EDA plots
│   └── saved_models/       # Persisted model files
├── features/
│   ├── extract_ids.py      # UUID extraction (legacy)
│   └── text_features.py    # TF-IDF feature builder
├── models/
│   ├── cf_model.py         # ALS collaborative filtering
│   ├── cb_model.py         # Content-based model
│   ├── hybrid.py           # Hybrid recommender
│   ├── evaluation.py       # RMSE, MAE, Precision@K, Recall@K
│   ├── svd_recommender.py  # SVD (surprise)
│   ├── fpgrowth_recommender.py  # FP-Growth (mlxtend)
│   ├── similarity_recommender.py  # Cosine/Pearson similarity
│   └── persistence.py      # Save/load models
└── utils/
    ├── logger.py
    └── timer.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure database

Copy `.env.example` to `.env` and set your PostgreSQL connection string:

```env
POSTGRES_URL=postgresql://user:password@localhost:5432/dbname
```

### 3. Verify database tables

The system reads from these tables:
- **`audit_logs`** — user interactions (`userId`, `resourceId`, `signalWeight`, `resourceType`, `action`)
- **`content`** + related tables (movies, tvseries, category, tag, actor, director)

## CLI Usage

```bash
# Train a model
python cli.py train --model svd
python cli.py train --model fpgrowth
python cli.py train --model similarity
python cli.py train --model als

# Get recommendations
python cli.py recommend --user-id <UUID> --model svd --top-n 10
python cli.py recommend --user-id <UUID> --model fpgrowth
python cli.py recommend --user-id <UUID> --model hybrid

# Evaluate models
python cli.py evaluate --model all
python cli.py evaluate --model svd

# Generate EDA plots
python cli.py eda

# List saved models
python cli.py models
```

## API Usage

```bash
# Start the server
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Or
python api.py
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/recommend/{user_id}` | Hybrid recommendations |
| `GET` | `/recommend/{user_id}/svd` | SVD recommendations |
| `GET` | `/recommend/{user_id}/fpgrowth` | FP-Growth recommendations |
| `GET` | `/recommend/{user_id}/similarity?metric=cosine` | Similarity recommendations |
| `GET` | `/models` | List saved models |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## How signalWeight Maps to Ratings

| signalWeight | Meaning | Rating |
|---|---|---|
| `2` | Strong signal (PLAY, LIKE) | `5` |
| `1` | Medium signal (WATCHLIST) | `3` |
| `-1` | Negative signal (UNLIKE, REMOVE) | `1` (filtered by default) |
| `0` | Ignored | Excluded |

## Evaluation Metrics

- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE** — Mean Absolute Error
- **Precision@K** — Fraction of recommended items that are relevant
- **Recall@K** — Fraction of relevant items that are recommended
- **Hit Rate** — Fraction of users with at least one hit
- **Coverage** — Percentage of items that can be recommended