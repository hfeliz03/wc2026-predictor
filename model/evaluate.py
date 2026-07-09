"""
Milestone 9.

Live grading loop. As QF/SF/Final games are actually played, their results
get appended to data/raw/matches.csv (see record_result()), and this module
grades each round's LOCKED pre-round predictions against the real outcome.

Why "locked"? A round's predictions must be frozen and archived to
model/predictions_history/ the moment they're first generated, BEFORE that
round is played - otherwise grading would compare a game's outcome against
a probability recomputed with knowledge of that same outcome (real
leakage, unlike the R32/R16-informs-QF forecasting in train.py, which never
touches a game's own result). lock_round() refuses to (re-)lock a round
that already has real results recorded, precisely to prevent this.

Entry points:
    record_result(date, stage, team_a, team_b, score_a, score_b,
                   went_to_et=False, went_to_pens=False, winner=None)
    lock_round(round_key, payload)         -> Path | None
    grade_round(round_key, stage_label)    -> dict | None
    cumulative_validation()                -> dict
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import load_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHES_CSV = REPO_ROOT / "data" / "raw" / "matches.csv"
HISTORY_DIR = REPO_ROOT / "model" / "predictions_history"
TRAINED_WEIGHTS_PATH = REPO_ROOT / "model" / "trained_weights.json"

ROUND_STAGE_LABELS = {
    "quarterfinals": "Quarterfinal",
    "semifinals": "Semifinal",
    "final": "Final",
}


# --------------------------------------------------------------------------
# Metrics (matches train.py's - no sklearn.metrics available, see there)
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
# Recording real results as they happen
# --------------------------------------------------------------------------

def record_result(date: str, stage: str, team_a: str, team_b: str, score_a: int, score_b: int,
                   went_to_et: bool = False, went_to_pens: bool = False, winner: str = None) -> None:
    """
    Append one real, played match to data/raw/matches.csv. `stage` must be
    one of "Quarterfinal", "Semifinal", "Final". If the match was decided
    on penalties (score_a == score_b), pass `winner` explicitly - it can't
    be inferred from the score alone (see the R32/R16 penalty-shootout bug
    this exact ambiguity caused in load_data.matches_long(), now fixed).
    """
    if winner is None:
        if score_a == score_b:
            raise ValueError("Scores are level - this must have been decided on penalties or is a data error. Pass winner= explicitly.")
        winner = team_a if score_a > score_b else team_b

    with open(MATCHES_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([date, stage, team_a, team_b, score_a, score_b, went_to_et, went_to_pens, winner])


def played_result(stage_label: str, team_a: str, team_b: str) -> dict | None:
    """The real result for this pairing at this stage, if it's been played, else None."""
    matches = load_data.load_matches()
    rows = matches[
        (matches["stage"] == stage_label)
        & (
            ((matches["team_a"] == team_a) & (matches["team_b"] == team_b))
            | ((matches["team_a"] == team_b) & (matches["team_b"] == team_a))
        )
    ]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {
        "winner": r["winner"],
        "score_a": int(r["score_a"]),
        "score_b": int(r["score_b"]),
        "date": str(r["date"].date()),
    }


def stage_has_any_results(stage_label: str) -> bool:
    matches = load_data.load_matches()
    return bool((matches["stage"] == stage_label).any())


def locked_game(round_key: str, team_a: str, team_b: str) -> dict | None:
    """
    The locked (pre-round, never-recomputed) prediction for this exact
    pairing, if that round has been locked. Once a game is played, this is
    what should be DISPLAYED for it too - not a fresh recompute, which for
    the aware model would now have that game's own result baked into the
    knockout-form features used to predict it (real leakage, just in the
    display path rather than the official grading, which already used this
    correctly - see grade_round()).
    """
    lock_path = HISTORY_DIR / f"{round_key}_locked.json"
    if not lock_path.exists():
        return None
    games = json.loads(lock_path.read_text())
    games = [games] if isinstance(games, dict) else games
    for g in games:
        if {g["team_a"], g["team_b"]} == {team_a, team_b}:
            return g
    return None


def all_played_results(stage_label: str) -> dict:
    """
    {frozenset({team_a, team_b}): result_dict} for every match already
    played at this stage. Load once and reuse for O(1) lookups instead of
    re-reading + re-parsing matches.csv on every call - matters inside a
    10k-iteration Monte Carlo loop (see predict.py's monte_carlo_bracket).
    """
    matches = load_data.load_matches()
    rows = matches[matches["stage"] == stage_label]
    out = {}
    for _, r in rows.iterrows():
        out[frozenset({r["team_a"], r["team_b"]})] = {
            "winner": r["winner"],
            "score_a": int(r["score_a"]),
            "score_b": int(r["score_b"]),
            "date": str(r["date"].date()),
        }
    return out


# --------------------------------------------------------------------------
# Locking (archiving) a round's predictions BEFORE it's played
# --------------------------------------------------------------------------

def lock_round(round_key: str, payload: dict) -> Path | None:
    """
    Archive payload[round_key] to model/predictions_history/{round_key}_locked.json,
    but only if: (a) not already locked, and (b) that stage has zero real
    results recorded yet (i.e. we're not locking a prediction that was
    computed with knowledge of its own outcome).
    """
    stage_label = ROUND_STAGE_LABELS[round_key]
    HISTORY_DIR.mkdir(exist_ok=True)
    out_path = HISTORY_DIR / f"{round_key}_locked.json"

    if out_path.exists():
        return None  # already locked - never overwrite

    if stage_has_any_results(stage_label):
        print(f"WARNING: refusing to lock '{round_key}' - {stage_label} already has "
              f"real results recorded, so this prediction can't be trusted as pre-round. "
              f"(Should have been locked in an earlier run, before kickoff.)")
        return None

    games = payload[round_key]
    games = [games] if isinstance(games, dict) else games  # "final" is a single dict
    out_path.write_text(json.dumps(games, indent=2))
    return out_path


# --------------------------------------------------------------------------
# Grading a round against its locked predictions
# --------------------------------------------------------------------------

def grade_round(round_key: str) -> dict | None:
    """
    Compare the locked pre-round predictions for `round_key` against real
    results now in data/raw/matches.csv. Returns None if not locked yet, or
    if none of that round's games have been played yet.
    """
    stage_label = ROUND_STAGE_LABELS[round_key]
    lock_path = HISTORY_DIR / f"{round_key}_locked.json"
    if not lock_path.exists():
        return None

    locked_games = json.loads(lock_path.read_text())
    graded = []
    for g in locked_games:
        real = played_result(stage_label, g["team_a"], g["team_b"])
        if real is None:
            continue
        graded.append({
            "team_a": g["team_a"],
            "team_b": g["team_b"],
            "actual_winner": real["winner"],
            "actual_score": f"{real['score_a']}-{real['score_b']}",
            "pA_blind": g["pA_blind"],
            "pA_aware": g["pA_aware"],
            "correct_blind": g["predicted_blind"] == real["winner"],
            "correct_aware": g["predicted_aware"] == real["winner"],
        })

    if not graded:
        return None

    y_true = [int(g["actual_winner"] == g["team_a"]) for g in graded]
    p_blind = [g["pA_blind"] for g in graded]
    p_aware = [g["pA_aware"] for g in graded]

    def block(preds_correct, probs):
        preds = [int(p >= 0.5) for p in probs]
        return {
            "n_correct": sum(preds_correct),
            "accuracy": accuracy_score(y_true, preds),
            "brier": brier_score_loss(y_true, probs),
            "log_loss": log_loss(y_true, probs),
        }

    return {
        "stage": stage_label,
        "n": len(graded),
        "n_total_in_round": len(locked_games),
        "games": graded,
        "blind": block([g["correct_blind"] for g in graded], p_blind),
        "aware": block([g["correct_aware"] for g in graded], p_aware),
    }


def cumulative_validation() -> dict:
    """
    Everything the site's scorecard needs: the original R32+R16 audit
    (frozen at Milestone 5, never changes) plus whichever of QF/SF/Final
    have live-graded results so far.
    """
    base = json.loads(TRAINED_WEIGHTS_PATH.read_text())["form_blind_test_metrics"]
    live_rounds = {}
    for round_key in ("quarterfinals", "semifinals", "final"):
        g = grade_round(round_key)
        if g:
            live_rounds[round_key] = g
    return {"r32_r16": base, "live_rounds": live_rounds}


if __name__ == "__main__":
    v = cumulative_validation()
    print("=== R32 + R16 (frozen Milestone 5 audit) ===")
    print(f"  accuracy={v['r32_r16']['accuracy']:.3f}  n_correct={v['r32_r16']['n_correct']}/{v['r32_r16']['n']}")

    if not v["live_rounds"]:
        print("\nNo live rounds graded yet (no QF/SF/Final results recorded, or not locked in).")
    for round_key, g in v["live_rounds"].items():
        print(f"\n=== {g['stage']} ({g['n']}/{g['n_total_in_round']} games played) ===")
        print(f"  blind: accuracy={g['blind']['accuracy']:.3f}  n_correct={g['blind']['n_correct']}/{g['n']}  brier={g['blind']['brier']:.3f}")
        print(f"  aware: accuracy={g['aware']['accuracy']:.3f}  n_correct={g['aware']['n_correct']}/{g['n']}  brier={g['aware']['brier']:.3f}")
        for game in g["games"]:
            print(f"    {game['team_a']:10s} vs {game['team_b']:10s}  actual={game['actual_winner']:10s} ({game['actual_score']})  "
                  f"blind {'✓' if game['correct_blind'] else '✗'}  aware {'✓' if game['correct_aware'] else '✗'}")
