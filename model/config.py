"""
Single source of truth for tunable constants used across the pipeline.
Per PROJECT_PLAN.md #10: keep all tunable numbers here, not scattered.
"""

# --- Data sources -----------------------------------------------------------
REFERENCE_REPO = "https://github.com/rezarahiminia/worldcup2026"
RAW_DATA_DIR = "data/raw"
MANUAL_DATA_DIR = "data/manual"

# --- Feature weights (fallback / hand-set, used if regression fit is unstable)
# These get overwritten by train.py's fitted logistic regression coefficients
# when the fit is stable; kept here as a sane, documented default. Keys must
# match model/features.py's FEATURE_COLUMNS exactly (see build_team_table()).
FALLBACK_WEIGHTS = {
    "fifa_points": 0.25,
    "wc_titles": 0.10,
    "major_trophies": 0.05,
    "narrative": 0.05,
    "group_pts": 0.15,
    "group_gd": 0.10,
    "rank_vs_expectation": 0.10,
    "ko_wins": 0.08,
    "ko_losses": 0.08,
    "ko_gd": 0.09,
    "ko_fragile": -0.05,  # more ET/pens escapes so far = slightly less reliable
    "is_host": 0.05,
}

# --- Logistic regression (calibration step) ---------------------------------
LOGREG_C = 0.1          # small C = heavy regularization, given ~24-game sample
LOGREG_MAX_ITER = 1000

# --- Strength -> probability conversion --------------------------------------
LOGISTIC_K = 1.0        # slope on the strength gap; tuned during calibration

# --- Monte Carlo bracket simulation ------------------------------------------
MONTE_CARLO_RUNS = 10_000
RANDOM_SEED = 2026

# --- Host nations (small contextual flag) ------------------------------------
HOST_NATIONS = {"USA", "Canada", "Mexico"}

# --- Output -------------------------------------------------------------------
PREDICTIONS_JSON_PATH = "site/predictions.json"
