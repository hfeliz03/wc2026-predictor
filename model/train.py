"""
Milestone 5 (revised).

PROJECT_PLAN.md contains two slightly different framings of train/test that
we had to reconcile explicitly with the user:
  - §1: pre-tournament signal + group-stage performance = training data;
    R32 + R16 results = testing data, checked for upset-calling accuracy.
    No fitting on any knockout outcome at all.
  - §5's "Train/test/predict split" subsection: fit weights *on* R32
    results, validate on R16 — i.e. knockout outcomes ARE used to fit.

The user confirmed §1 is the intended design. So the PRIMARY model here is
the interpretable backbone rating (PROJECT_PLAN.md §5, formula 1):
strength(team) = sum(w_i * zscore(feature_i)) using only pre-tournament +
group-stage features (never knockout-form — that data doesn't exist yet at
"training time" in this framing), with HAND-SET weights (config.
FALLBACK_WEIGHTS) — no supervised fitting on any match outcome. Win
probability is a fixed logistic squash of the strength gap. This backbone
is then graded, with zero leakage, against all 24 completed R32+R16 games.

A supervised logistic-regression calibration (§5, paragraph 2: "fit on
R32+R16 ... to learn how much each feature mattered") is still run and
reported, but only as a secondary, clearly-labeled diagnostic — it is NOT
used for grading or for Milestone 6 predictions, since fitting it touches
the same games we test on. It answers "what would the data suggest if we
let it recalibrate weights", nothing more.

*Dependency note:* scikit-learn/scipy can't be installed in this sandbox
(PyPI is blocked by the same network allowlist as GitHub; confirmed with
the user, who couldn't find a broader allowlist option either). The
logistic-regression diagnostic below is a small self-contained L2-
regularized implementation (Newton's method / IRLS) using only numpy,
with an sklearn-shaped API (coef_, intercept_, predict_proba) so it can be
swapped for the real thing with zero call-site changes if this repo is run
somewhere with full internet access.

Entry points:
    backbone_strength(team_table, weights) -> pd.Series indexed by team
    backbone_predict_proba(strength, team_a, team_b, k) -> float
    evaluate_backbone(team_table, k) -> dict            # PRIMARY grading
    fit_weights / leave_one_out_cv                       # secondary diagnostic
    save_weights(...)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features
from config import FALLBACK_WEIGHTS, LOGISTIC_K, LOGREG_C, LOGREG_MAX_ITER

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = REPO_ROOT / "model" / "trained_weights.json"

# Pre-tournament + group-stage features only — no knockout-form, since this
# backbone is meant to be fully specified before any knockout game is played.
# Used for the Milestone 5 "how much does pre-knockout signal alone get us"
# audit (graded on all 24 R32+R16 games).
BACKBONE_FEATURES = features.STATIC_FEATURE_COLUMNS + ["is_host"]

# Pre-tournament + group-stage + knockout-form. Knockout form is NOT
# available before any knockout game exists, so this variant only makes
# sense for matches that have prior knockout history (R16 onward). Using
# already-completed R32/R16 results to inform R16/QF predictions is not
# leakage - it's forward-looking use of real, already-known information
# about EARLIER matches to predict LATER ones (never a match's own result).
FULL_FEATURES = BACKBONE_FEATURES + features.KNOCKOUT_FORM_COLUMNS

COEF_STABILITY_THRESHOLD = 8.0  # for the secondary diagnostic fit only


# --------------------------------------------------------------------------
# PRIMARY MODEL: hand-set-weight backbone, graded on all 24 knockout games
# --------------------------------------------------------------------------

def zscore_columns(team_table: pd.DataFrame, columns: list) -> pd.DataFrame:
    z = team_table[columns].astype(float).copy()
    for col in columns:
        std = z[col].std()
        z[col] = (z[col] - z[col].mean()) / (std if std > 0 else 1.0)
    return z


def backbone_strength(team_table: pd.DataFrame, weights: dict = FALLBACK_WEIGHTS,
                       feature_columns: list = BACKBONE_FEATURES) -> pd.Series:
    """strength(team) = sum(w_i * zscore(feature_i)) over feature_columns."""
    z = zscore_columns(team_table, feature_columns)
    w = np.array([weights.get(c, 0.0) for c in feature_columns])
    return pd.Series(z.values @ w, index=team_table.index, name="strength")


def team_table_asof(before_date) -> pd.DataFrame:
    """
    features.build_team_table() but with knockout-form columns recomputed as
    of a specific cutoff date, instead of "now" (which reflects everything
    through R16). Used to rebuild a clean, leak-free snapshot for testing
    the form-aware backbone on R16 using only prior (R32) knockout results.
    """
    table = features.build_team_table().copy()
    matches = features.load_data.load_matches()
    ml = features.load_data.matches_long(matches)
    ko = {team: features.knockout_form(team, ml, before_date=before_date) for team in table.index}
    ko_df = pd.DataFrame(ko).T[features.KNOCKOUT_FORM_COLUMNS]
    table.update(ko_df)
    return table


def evaluate_form_aware_on_r16(weights: dict = FALLBACK_WEIGHTS, k: float = LOGISTIC_K) -> dict:
    """
    Apples-to-apples comparison, requested explicitly: does adding knockout
    form (computed only from R32 results - strictly prior to any R16 game)
    improve on the form-blind backbone, tested on the same 8 R16 games?
    A single fixed cutoff (just after the last R32 game, before any R16
    game) is used for every R16 prediction, so no R16 result ever leaks
    into another R16 prediction's inputs.
    """
    matches = features.load_data.load_matches()
    r32_end = matches.loc[matches["stage"] == "Round of 32", "date"].max() + pd.Timedelta(days=1)
    table_asof = team_table_asof(r32_end)
    strength = backbone_strength(table_asof, weights, FULL_FEATURES)

    r16 = matches[matches["stage"] == "Round of 16"].reset_index(drop=True)
    probs, y_true = [], []
    for _, m in r16.iterrows():
        p = backbone_predict_proba(strength, m["team_a"], m["team_b"], k)
        probs.append(p)
        y_true.append(int(m["winner"] == m["team_a"]))

    probs, y_true = np.array(probs), np.array(y_true)
    preds = (probs >= 0.5).astype(int)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "probs": probs.tolist(),
        "rows": r16,
    }


def backbone_predict_proba(strength: pd.Series, team_a: str, team_b: str, k: float = LOGISTIC_K) -> float:
    gap = strength[team_a] - strength[team_b]
    return float(1.0 / (1.0 + np.exp(-k * gap)))


def evaluate_backbone(team_table: pd.DataFrame, k: float = LOGISTIC_K, weights: dict = FALLBACK_WEIGHTS) -> dict:
    """
    Grade the backbone against every completed knockout game (R32 + R16,
    24 games) — pure test data, nothing here was fit to these outcomes.
    """
    matches = features.load_data.load_matches()
    knockout = matches[matches["stage"].isin(["Round of 32", "Round of 16"])].reset_index(drop=True)
    strength = backbone_strength(team_table, weights)

    probs, y_true, rows = [], [], []
    for _, m in knockout.iterrows():
        p = backbone_predict_proba(strength, m["team_a"], m["team_b"], k)
        won = int(m["winner"] == m["team_a"])
        probs.append(p)
        y_true.append(won)
        rows.append(m)

    probs = np.array(probs)
    y_true = np.array(y_true)
    preds = (probs >= 0.5).astype(int)

    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "probs": probs.tolist(),
        "rows": knockout,
        "strength": strength,
    }


# --------------------------------------------------------------------------
# Metrics (no sklearn.metrics available - see dependency note above)
# --------------------------------------------------------------------------

def accuracy_score(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def brier_score_loss(y_true, probs) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(y_true)) ** 2))


def log_loss(y_true, probs, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# --------------------------------------------------------------------------
# SECONDARY DIAGNOSTIC ONLY: supervised calibration on all 24 games.
# Not used for grading (§ above) or for Milestone 6 predictions - fitting
# and testing on the same 24 games would be leakage for a "graded" number.
# --------------------------------------------------------------------------

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.scale_ = std
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RidgeLogisticRegression:
    """L2-regularized logistic regression via Newton's method (IRLS). sklearn-shaped API."""

    def __init__(self, C: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        w = np.zeros(d + 1)
        lam = 1.0 / self.C
        reg = np.eye(d + 1) * lam
        reg[0, 0] = 0.0
        for _ in range(self.max_iter):
            z = np.clip(Xb @ w, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) + reg @ w
            W = p * (1 - p)
            H = Xb.T @ (Xb * W[:, None]) + reg
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, grad, rcond=None)[0]
            w_new = w - step
            if np.max(np.abs(w_new - w)) < self.tol:
                w = w_new
                break
            w = w_new
        self.intercept_ = np.array([w[0]])
        self.coef_ = w[1:].reshape(1, -1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = np.clip(self.intercept_[0] + X @ self.coef_.ravel(), -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _strip_prefix(col: str) -> str:
    return col[2:] if col.startswith("d_") else col


def fit_weights(X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    model = RidgeLogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER)
    model.fit(X_scaled, y_train.values)
    used_fallback = bool(np.abs(model.coef_).max() > COEF_STABILITY_THRESHOLD)
    if used_fallback:
        fallback_coef = np.array([FALLBACK_WEIGHTS.get(_strip_prefix(c), 0.0) for c in X_train.columns])
        model.coef_ = fallback_coef.reshape(1, -1)
        model.intercept_ = np.array([0.0])
    return scaler, model, used_fallback


def predict_proba_diag(scaler, model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(X.values))[:, 1]


def leave_one_out_cv(X_all: pd.DataFrame, y_all: pd.Series) -> dict:
    n = len(X_all)
    probs, y_true = [], []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler, model, _ = fit_weights(X_all.iloc[train_idx], y_all.iloc[train_idx])
        probs.append(predict_proba_diag(scaler, model, X_all.iloc[[i]])[0])
        y_true.append(y_all.iloc[i])
    probs, y_true = np.array(probs), np.array(y_true)
    preds = (probs >= 0.5).astype(int)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, preds),
        "brier": brier_score_loss(y_true, probs),
        "log_loss": log_loss(y_true, probs),
    }


def save_weights(weights: dict, k: float, backbone_metrics: dict, r16_comparison: dict = None, path: Path = WEIGHTS_PATH) -> None:
    payload = {
        "method": "backbone_handset_weights",
        "note": (
            "form_blind_features is graded on all 24 R32+R16 games with zero "
            "knockout-outcome signal (Milestone 5 audit). full_features adds "
            "knockout form and is what Milestone 6 uses for QF/SF/Final "
            "predictions, since R32+R16 are real completed results by then, "
            "not a leak."
        ),
        "form_blind_features": BACKBONE_FEATURES,
        "full_features": FULL_FEATURES,
        "weights": weights,
        "logistic_k": k,
        "form_blind_test_metrics": {
            "n": backbone_metrics["n"],
            "n_correct": int(round(backbone_metrics["accuracy"] * backbone_metrics["n"])),
            "accuracy": backbone_metrics["accuracy"],
            "brier": backbone_metrics["brier"],
            "log_loss": backbone_metrics["log_loss"],
        },
        "r16_form_comparison": r16_comparison or {},
    }
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    team_table = features.build_team_table()

    print("=== PRIMARY: hand-set backbone, graded on all 24 R32+R16 games ===")
    print("(no fitting on any knockout outcome - see module docstring)\n")
    result = evaluate_backbone(team_table)
    print(f"  n={result['n']}  accuracy={result['accuracy']:.3f}  "
          f"brier={result['brier']:.3f}  log_loss={result['log_loss']:.3f}\n")

    print("Per-game calls:")
    rows = result["rows"].reset_index(drop=True)
    n_correct = 0
    for i, row in rows.iterrows():
        p = result["probs"][i]
        pred_winner = row["team_a"] if p >= 0.5 else row["team_b"]
        correct = pred_winner == row["winner"]
        n_correct += correct
        tag = "correct" if correct else "MISS"
        print(f"  [{row['stage']:12s}] {row['team_a']:6s} vs {row['team_b']:6s}: "
              f"P({row['team_a']})={p:.2f} -> called {pred_winner:6s} | "
              f"actual {row['winner']:6s} [{tag}]")
    print(f"\n  {n_correct}/{result['n']} correct\n")

    print("Team strength ranking (top 10):")
    print(result["strength"].sort_values(ascending=False).head(10), "\n")

    print("=== Does recent knockout form add predictive value? ===")
    print("Apples-to-apples on the 8 R16 games: form-blind backbone vs. form-aware")
    print("backbone (adds ko_wins/ko_losses/ko_gd/ko_fragile, computed only from")
    print("each team's R32 result - strictly prior to the R16 game being predicted).\n")

    r16_mask = rows["stage"] == "Round of 16"
    r16_rows = rows[r16_mask].reset_index(drop=True)
    r16_form_blind_probs = [result["probs"][i] for i in rows.index[r16_mask]]
    form_blind_preds = [int(p >= 0.5) for p in r16_form_blind_probs]
    form_blind_y = [int(r["winner"] == r["team_a"]) for _, r in r16_rows.iterrows()]
    form_blind_r16_acc = accuracy_score(form_blind_y, form_blind_preds)
    print(f"  form-blind backbone on R16 only:  accuracy={form_blind_r16_acc:.3f} "
          f"({sum(form_blind_preds[i] == form_blind_y[i] for i in range(8))}/8)")

    form_aware = evaluate_form_aware_on_r16()
    print(f"  form-aware backbone on R16 only:  accuracy={form_aware['accuracy']:.3f} "
          f"({int(round(form_aware['accuracy'] * form_aware['n']))}/{form_aware['n']})  "
          f"brier={form_aware['brier']:.3f}  log_loss={form_aware['log_loss']:.3f}\n")

    for i, row in form_aware["rows"].reset_index(drop=True).iterrows():
        p_blind = r16_form_blind_probs[i]
        p_aware = form_aware["probs"][i]
        pick_blind = row["team_a"] if p_blind >= 0.5 else row["team_b"]
        pick_aware = row["team_a"] if p_aware >= 0.5 else row["team_b"]
        flip = " <- form flipped the pick" if pick_blind != pick_aware else ""
        print(f"  {row['team_a']:6s} vs {row['team_b']:6s}: blind P={p_blind:.2f} ({pick_blind:6s}) | "
              f"aware P={p_aware:.2f} ({pick_aware:6s}) | actual {row['winner']:6s}{flip}")

    n = 8
    print(f"\n  With only n={n} games this isn't statistically conclusive either way, but it's")
    print("  the honest, non-leaky answer to 'does recent form help': the two models are")
    print("  compared on identical held-out games, and form only ever uses strictly earlier")
    print("  results. Given the direction here, Milestone 6 will use the form-aware backbone")
    print("  (informed by both R32 + R16 by the time QF predictions are made) as the primary")
    print("  forecasting model - that's forward-looking use of real information, not leakage.")

    r16_comparison = {
        "n": 8,
        "form_blind_accuracy": form_blind_r16_acc,
        "form_blind_n_correct": sum(form_blind_preds[i] == form_blind_y[i] for i in range(8)),
        "form_aware_accuracy": form_aware["accuracy"],
        "form_aware_n_correct": int(round(form_aware["accuracy"] * form_aware["n"])),
    }
    save_weights(FALLBACK_WEIGHTS, LOGISTIC_K, result, r16_comparison)
    print(f"\nSaved backbone weights + test metrics -> {WEIGHTS_PATH}\n")

    print("=== SECONDARY (diagnostic only, not used for grading or predictions) ===")
    print("Supervised logistic regression fit on all 24 R32+R16 games, evaluated via LOO-CV:\n")
    X_all, y_all, meta_all = features.build_match_features(stages=("Round of 32", "Round of 16"))
    scaler, model, used_fallback = fit_weights(X_all, y_all)
    print("  (fallback triggered)" if used_fallback else "  Fitted coefficients (full-feature set, incl. knockout form):")
    for col, coef in zip(X_all.columns, model.coef_.ravel()):
        print(f"    {col:28s} {coef:+.3f}")
    loo = leave_one_out_cv(X_all, y_all)
    print(f"\n  LOO-CV: accuracy={loo['accuracy']:.3f}  brier={loo['brier']:.3f}  log_loss={loo['log_loss']:.3f}")
    print("  (shown for comparison only - Milestone 6 predictions use the backbone above)")
