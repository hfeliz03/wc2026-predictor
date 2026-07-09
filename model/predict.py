"""
Milestone 6.

Generates QF -> SF -> Final predictions and writes predictions.json per the
schema in PROJECT_PLAN.md #6 (extended slightly - see deviations below).

*Deviation from the original plan, agreed with the user after Milestone 5's
form-blind-vs-form-aware test came back inconclusive (n=8, no picks
flipped):* every prediction here is reported TWICE, not once - "blind"
(pre-tournament + group-stage signal only, the same model graded at 0.750
accuracy on all 24 R32+R16 games) and "aware" (adds each team's knockout
form from the already-completed R32 + R16 - not leakage, since those games
are in the past relative to QF/SF/Final; see train.py's module docstring
for the leakage-vs-forecasting distinction). Showing both lets the site
(and the user) see exactly where recent form changes a call and where it
doesn't, rather than us silently picking one.

Simplification, flagged: knockout-form features are frozen at their current
(post-R16) values for every round - QF, SF, and Final all use the same
snapshot. We do not attempt to simulate a team "gaining form" from a
hypothetical win deeper in the same simulated bracket path; Milestone 5's
test found the form signal barely moves predictions with just one prior
knockout game, so modeling recursive in-simulation form updates isn't
justified by the evidence yet.

Two views, each computed for both blind and aware:
    1. Deterministic bracket - advance the favorite each round.
    2. Monte Carlo (config.MONTE_CARLO_RUNS trials) - sample each game from
       its win probability; report each team's % chance to reach the SF,
       final, and to win the cup.

Entry points:
    quarterfinal_fixtures() -> list[dict]
    deterministic_bracket(strength, k) -> dict
    monte_carlo_bracket(strength, k, n_runs, seed) -> dict
    build_predictions() -> dict                      # full predictions.json payload
    write_predictions_json(payload, path)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import features
import train
from config import HOST_NATIONS, LOGISTIC_K, MONTE_CARLO_RUNS, PREDICTIONS_JSON_PATH, RANDOM_SEED

REPO_ROOT = Path(__file__).resolve().parent.parent

# Confirmed via research (FIFA's official R32/R16 bracket slotting, cross-
# checked across multiple sources): the fixed QF fixtures and which QF
# winners meet in which semifinal. QFs are July 9-11, 2026; none played yet
# as of this run.
QF_FIXTURES = [
    {"slot": "QF1", "date": "2026-07-09", "team_a": "France", "team_b": "Morocco"},
    {"slot": "QF2", "date": "2026-07-10", "team_a": "Spain", "team_b": "Belgium"},
    {"slot": "QF3", "date": "2026-07-10", "team_a": "Norway", "team_b": "England"},
    {"slot": "QF4", "date": "2026-07-11", "team_a": "Argentina", "team_b": "Switzerland"},
]
SF_PAIRING = [("QF1", "QF2"), ("QF3", "QF4")]  # SF1 = winner(QF1) v winner(QF2); SF2 similarly

FEATURE_LABELS = {
    "fifa_points": "FIFA ranking points",
    "wc_titles": "World Cup titles",
    "major_trophies": "continental titles",
    "narrative": "storyline",
    "group_pts": "group-stage points",
    "group_gd": "group-stage goal difference",
    "rank_vs_expectation": "over/underperforming their seed",
    "ko_wins": "knockout wins so far",
    "ko_losses": "knockout losses so far",
    "ko_gd": "knockout-round goal difference",
    "ko_fragile": "extra-time/penalty escapes (fragility)",
    "is_host": "host-nation status",
}


def top_contributors(team_table: pd.DataFrame, team_a: str, team_b: str, weights: dict, feature_columns: list, n: int = 2) -> list:
    """The n features whose weighted z-score gap most favors/hurts team_a, for rationale text."""
    z = train.zscore_columns(team_table, feature_columns)
    contributions = []
    for col in feature_columns:
        w = weights.get(col, 0.0)
        gap = z.loc[team_a, col] - z.loc[team_b, col]
        contributions.append((col, w * gap))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:n]


def rationale(team_table: pd.DataFrame, team_a: str, team_b: str, weights: dict, feature_columns: list) -> str:
    top = top_contributors(team_table, team_a, team_b, weights, feature_columns)
    parts = []
    for col, contribution in top:
        who = team_a if contribution > 0 else team_b
        label = FEATURE_LABELS.get(col, col)
        parts.append(f"{who} favored on {label}")
    return "; ".join(parts) if parts else "Evenly matched on the features considered"


def quarterfinal_fixtures() -> list:
    return QF_FIXTURES


def _game_probs(strength_blind: pd.Series, strength_aware: pd.Series, team_a: str, team_b: str, k: float = LOGISTIC_K) -> dict:
    p_blind = train.backbone_predict_proba(strength_blind, team_a, team_b, k)
    p_aware = train.backbone_predict_proba(strength_aware, team_a, team_b, k)
    return {
        "team_a": team_a,
        "team_b": team_b,
        "pA_blind": round(p_blind, 4),
        "pB_blind": round(1 - p_blind, 4),
        "pA_aware": round(p_aware, 4),
        "pB_aware": round(1 - p_aware, 4),
        "predicted_blind": team_a if p_blind >= 0.5 else team_b,
        "predicted_aware": team_a if p_aware >= 0.5 else team_b,
    }


def deterministic_bracket(team_table: pd.DataFrame, strength_blind: pd.Series, strength_aware: pd.Series,
                           weights: dict, k: float = LOGISTIC_K) -> dict:
    """
    Advance the favorite each round, per the AWARE model (informed by all
    completed results) - the blind model's picks are still reported
    per-game for comparison, but the aware model is canonical for which
    single bracket path gets advanced here.
    """
    # Advance each model along its OWN independent path first - this is the
    # only correct way to get champion_blind/champion_aware right in
    # general. (Earlier version always advanced via the aware model's pick
    # and reported the blind probability for whatever pairing that
    # produced - harmless only because the two models happened to agree at
    # every QF/SF slot in this run. Fixed so it can't silently mislabel a
    # future run where they diverge earlier in the bracket.)
    blind_path = _advance_bracket(strength_blind, k)
    aware_path = _advance_bracket(strength_aware, k)
    bracket_agrees = (
        blind_path["qf_winners"] == aware_path["qf_winners"]
        and blind_path["final_matchup"] == aware_path["final_matchup"]
    )

    # Display pairing: aware model's path (informed by more completed
    # results). If the two models' paths diverge, note it explicitly rather
    # than paper over it - see bracket_agrees above.
    qf_results = {}
    for game in QF_FIXTURES:
        probs = _game_probs(strength_blind, strength_aware, game["team_a"], game["team_b"], k)
        probs["slot"] = game["slot"]
        probs["date"] = game["date"]
        probs["rationale_blind"] = rationale(team_table, game["team_a"], game["team_b"], weights, train.BACKBONE_FEATURES)
        probs["rationale_aware"] = rationale(team_table, game["team_a"], game["team_b"], weights, train.FULL_FEATURES)
        qf_results[game["slot"]] = probs

    sf_results = {}
    sf_slots = ["SF1", "SF2"]
    for sf_slot, (qf_a_slot, qf_b_slot) in zip(sf_slots, SF_PAIRING):
        team_a = qf_results[qf_a_slot]["predicted_aware"]
        team_b = qf_results[qf_b_slot]["predicted_aware"]
        probs = _game_probs(strength_blind, strength_aware, team_a, team_b, k)
        probs["slot"] = sf_slot
        probs["rationale_blind"] = rationale(team_table, team_a, team_b, weights, train.BACKBONE_FEATURES)
        probs["rationale_aware"] = rationale(team_table, team_a, team_b, weights, train.FULL_FEATURES)
        sf_results[sf_slot] = probs

    final_team_a, final_team_b = aware_path["final_matchup"]
    final_probs = _game_probs(strength_blind, strength_aware, final_team_a, final_team_b, k)
    final_probs["rationale_blind"] = rationale(team_table, final_team_a, final_team_b, weights, train.BACKBONE_FEATURES)
    final_probs["rationale_aware"] = rationale(team_table, final_team_a, final_team_b, weights, train.FULL_FEATURES)

    return {
        "quarterfinals": list(qf_results.values()),
        "semifinals": list(sf_results.values()),
        "final": final_probs,
        "champion_blind": blind_path["champion"],
        "champion_aware": aware_path["champion"],
        "bracket_agrees": bracket_agrees,
    }


def _advance_bracket(strength: pd.Series, k: float = LOGISTIC_K) -> dict:
    """Advance QF -> SF -> Final using ONE model's own predictions at every round."""
    qf_winners = {}
    for game in QF_FIXTURES:
        p = train.backbone_predict_proba(strength, game["team_a"], game["team_b"], k)
        qf_winners[game["slot"]] = game["team_a"] if p >= 0.5 else game["team_b"]

    sf_winners = []
    for qf_a_slot, qf_b_slot in SF_PAIRING:
        team_a, team_b = qf_winners[qf_a_slot], qf_winners[qf_b_slot]
        p = train.backbone_predict_proba(strength, team_a, team_b, k)
        sf_winners.append(team_a if p >= 0.5 else team_b)

    final_matchup = (sf_winners[0], sf_winners[1])
    p_final = train.backbone_predict_proba(strength, final_matchup[0], final_matchup[1], k)
    champion = final_matchup[0] if p_final >= 0.5 else final_matchup[1]

    return {"qf_winners": qf_winners, "sf_winners": sf_winners, "final_matchup": final_matchup, "champion": champion}


def monte_carlo_bracket(strength: pd.Series, k: float = LOGISTIC_K, n_runs: int = MONTE_CARLO_RUNS, seed: int = RANDOM_SEED) -> dict:
    """
    Sample each game from its win probability, n_runs times. Knockout form
    is frozen (see module docstring), so a team's strength doesn't change
    mid-simulation - only who they happen to face does.
    """
    rng = np.random.default_rng(seed)
    all_qf_teams = sorted({t for g in QF_FIXTURES for t in (g["team_a"], g["team_b"])})
    reach_sf = {t: 0 for t in all_qf_teams}
    reach_final = {t: 0 for t in all_qf_teams}
    champion = {t: 0 for t in all_qf_teams}

    for _ in range(n_runs):
        qf_winners = {}
        for game in QF_FIXTURES:
            p = train.backbone_predict_proba(strength, game["team_a"], game["team_b"], k)
            winner = game["team_a"] if rng.random() < p else game["team_b"]
            qf_winners[game["slot"]] = winner
            reach_sf[winner] += 1

        sf_winners = []
        for qf_a_slot, qf_b_slot in SF_PAIRING:
            team_a, team_b = qf_winners[qf_a_slot], qf_winners[qf_b_slot]
            p = train.backbone_predict_proba(strength, team_a, team_b, k)
            winner = team_a if rng.random() < p else team_b
            sf_winners.append(winner)
            reach_final[winner] += 1

        p_final = train.backbone_predict_proba(strength, sf_winners[0], sf_winners[1], k)
        champ = sf_winners[0] if rng.random() < p_final else sf_winners[1]
        champion[champ] += 1

    return {
        team: {
            "sf_pct": round(100 * reach_sf[team] / n_runs, 2),
            "final_pct": round(100 * reach_final[team] / n_runs, 2),
            "champion_pct": round(100 * champion[team] / n_runs, 2),
        }
        for team in all_qf_teams
    }


def build_predictions() -> dict:
    team_table = features.build_team_table()
    weights_payload = json.loads((REPO_ROOT / "model" / "trained_weights.json").read_text())
    weights = weights_payload["weights"]

    strength_blind = train.backbone_strength(team_table, weights, train.BACKBONE_FEATURES)
    strength_aware = train.backbone_strength(team_table, weights, train.FULL_FEATURES)

    qf_teams = [t for g in QF_FIXTURES for t in (g["team_a"], g["team_b"])]
    teams_payload = []
    for t in sorted(set(qf_teams)):
        row = team_table.loc[t]
        teams_payload.append({
            "name": t,
            "strength_blind": round(float(strength_blind[t]), 4),
            "strength_aware": round(float(strength_aware[t]), 4),
            "fifa_rank": int(row["fifa_rank"]),
            "group_pts": int(row["group_pts"]),
            "narrative": int(row["narrative"]),
            "is_host": bool(row["is_host"]),
        })

    bracket = deterministic_bracket(team_table, strength_blind, strength_aware, weights)
    mc_blind = monte_carlo_bracket(strength_blind)
    mc_aware = monte_carlo_bracket(strength_aware)

    return {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "method_note": (
            "Every game is scored by two models: 'blind' uses only "
            "pre-tournament + group-stage signal (graded 0.750 accuracy on "
            "all 24 completed R32+R16 games, zero knockout-outcome fitting); "
            "'aware' adds each team's knockout form from the already-played "
            "R32+R16 (a legitimate forward-looking use of completed results, "
            "not leakage - see model/train.py). An apples-to-apples test on "
            "the 8 R16 games found 'aware' didn't change any pick and was "
            "not distinguishably better with n=8, so both are shown rather "
            "than picking one as definitive."
        ),
        "teams": teams_payload,
        "quarterfinals": bracket["quarterfinals"],
        "semifinals": bracket["semifinals"],
        "final": bracket["final"],
        "champion_blind": bracket["champion_blind"],
        "champion_aware": bracket["champion_aware"],
        "bracket_agrees": bracket["bracket_agrees"],
        "monte_carlo": {"blind": mc_blind, "aware": mc_aware, "runs": MONTE_CARLO_RUNS},
        "validation": {
            "note": "Backbone (blind) graded on all 24 completed R32+R16 games; zero knockout-outcome fitting.",
            **weights_payload["form_blind_test_metrics"],
            "r16_form_comparison": weights_payload.get("r16_form_comparison", {}),
        },
    }


def write_predictions_json(payload: dict, path: Path = None) -> Path:
    out_path = Path(path) if path else REPO_ROOT / PREDICTIONS_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


if __name__ == "__main__":
    payload = build_predictions()

    print("=== Quarterfinals ===")
    for g in payload["quarterfinals"]:
        print(f"  {g['team_a']:10s} vs {g['team_b']:10s}  "
              f"blind: {g['predicted_blind']:10s} ({g['pA_blind']:.2f}/{g['pB_blind']:.2f})  "
              f"aware: {g['predicted_aware']:10s} ({g['pA_aware']:.2f}/{g['pB_aware']:.2f})")
        print(f"      blind rationale: {g['rationale_blind']}")
        print(f"      aware rationale: {g['rationale_aware']}")

    print("\n=== Semifinals (bracket advanced via aware model) ===")
    for g in payload["semifinals"]:
        print(f"  {g['team_a']:10s} vs {g['team_b']:10s}  "
              f"blind: {g['predicted_blind']:10s} ({g['pA_blind']:.2f}/{g['pB_blind']:.2f})  "
              f"aware: {g['predicted_aware']:10s} ({g['pA_aware']:.2f}/{g['pB_aware']:.2f})")

    f = payload["final"]
    print("\n=== Final ===")
    print(f"  {f['team_a']:10s} vs {f['team_b']:10s}  "
          f"blind: {f['predicted_blind']:10s} ({f['pA_blind']:.2f}/{f['pB_blind']:.2f})  "
          f"aware: {f['predicted_aware']:10s} ({f['pA_aware']:.2f}/{f['pB_aware']:.2f})")

    print(f"\nDeterministic champion (blind):  {payload['champion_blind']}")
    print(f"Deterministic champion (aware):  {payload['champion_aware']}")

    print(f"\n=== Monte Carlo ({MONTE_CARLO_RUNS:,} runs) - champion % ===")
    print(f"{'team':12s} {'blind':>8s} {'aware':>8s}")
    for team in sorted(payload["monte_carlo"]["blind"], key=lambda t: -payload["monte_carlo"]["aware"][t]["champion_pct"]):
        b = payload["monte_carlo"]["blind"][team]["champion_pct"]
        a = payload["monte_carlo"]["aware"][team]["champion_pct"]
        print(f"{team:12s} {b:7.2f}% {a:7.2f}%")

    out_path = write_predictions_json(payload)
    print(f"\nWrote {out_path}")
