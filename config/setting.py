import os
from dotenv import load_dotenv

load_dotenv()

# ─── Database ──────────────────────────────────────────
POSTGRES_URL_AUDIT = os.getenv("POSTGRES_URL_AUDIT")
POSTGRES_URL_CONTENT = os.getenv("POSTGRES_URL_CONTENT")

# ─── SVD ─────────────────────────────────────────────
SVD_N_FACTORS = 50

# ─── FP-Growth ───────────────────────────────────────
FP_MIN_SUPPORT = 0.01
FP_MIN_CONFIDENCE = 0.1
FP_MIN_LIFT = 1.0

# ─── Similarity ──────────────────────────────────────
TOP_K_NEIGHBORS = 20