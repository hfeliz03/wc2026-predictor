"""
Stretch goal (PROJECT_PLAN.md #7/#9): Golden Boot, top-assists, and Golden
Glove (clean sheets) leaderboards, plus a lightweight projection of each
eventual winner.

Current standings (through the Round of 16) are already-known facts, read
from data/manual/golden_boot.csv, top_assists.csv, and golden_glove.csv
(compiled via WebSearch, cross-checked against our own matches.csv
scorelines where possible - see each CSV's provenance), joined against
teams.csv to flag whether each player's team is still alive (their tally
can still grow) or eliminated (frozen).

*Projection, added on request - kept deliberately simple to avoid scope
creep*: rather than building a second predictive model (e.g. a per-opponent
expected-goals model), this reuses the Monte Carlo team-survival
probabilities already computed in predict.py. For a still-alive player:

    per_game_rate      = current_stat / matches_played_so_far
    expected_remaining = 1 (the QF, already scheduled)
                         + P(team wins QF)      [from monte_carlo sf_pct]
                         + P(team reaches Final) [from monte_carlo final_pct]
    projected_stat      = current_stat + per_game_rate * expected_remaining

Eliminated players get expected_remaining = 0, so projected_stat equals
their current (final) tally. This assumes a constant per-game scoring rate
for however many more games the team is expected to play - it does not
model tougher opposition, fatigue, rotation, or a player's own hot/cold
streak changing. That's a real simplification, flagged here rather than
hidden; a fuller model would forecast expected goals per specific remaining
opponent, which is a materially bigger undertaking than what was asked for.

Entry points:
    build_awards(mc_aware: dict) -> dict
        {"golden_boot": [...], "top_assists": [...]}, each row carrying
        both the current standing and a "projected_<stat>" field.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import load_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MANUAL_DIR = REPO_ROOT / "data" / "manual"


def _expected_remaining_games(team: str, still_alive: bool, mc_aware: dict) -> float:
    if not still_alive or team not in mc_aware:
        return 0.0
    team_mc = mc_aware[team]
    p_win_qf = team_mc["sf_pct"] / 100.0     # reaching SF requires winning the QF
    p_reach_final = team_mc["final_pct"] / 100.0
    return 1.0 + p_win_qf + p_reach_final    # QF is certain (already scheduled) + SF/Final in expectation


def _with_projection(df: pd.DataFrame, stat_col: str, mc_aware: dict) -> list:
    teams = load_data.load_teams().set_index("team")
    out = []
    for _, row in df.iterrows():
        team = row["team"]
        still_alive = bool(teams.loc[team, "advanced_to_qf"]) if team in teams.index else False
        rate = row[stat_col] / row["matches_played"] if row["matches_played"] else 0.0
        remaining = _expected_remaining_games(team, still_alive, mc_aware)
        projected = round(row[stat_col] + rate * remaining, 1)
        out.append({
            **row.to_dict(),
            "still_alive": still_alive,
            "expected_remaining_games": round(remaining, 2),
            f"projected_{stat_col}": projected,
        })
    return out


def _rerank_by_projection(rows: list, stat_col: str) -> list:
    ranked = sorted(rows, key=lambda r: -r[f"projected_{stat_col}"])
    for i, r in enumerate(ranked, start=1):
        r["projected_rank"] = i
    return ranked


def build_awards(mc_aware: dict) -> dict:
    golden_boot = pd.read_csv(MANUAL_DIR / "golden_boot.csv")
    top_assists = pd.read_csv(MANUAL_DIR / "top_assists.csv")
    golden_glove = pd.read_csv(MANUAL_DIR / "golden_glove.csv")

    gb_rows = _with_projection(golden_boot, "goals", mc_aware)
    ta_rows = _with_projection(top_assists, "assists", mc_aware)
    gg_rows = _with_projection(golden_glove, "clean_sheets", mc_aware)

    return {
        "note": (
            "Standings through the Round of 16 (not a prediction - these are "
            "already-known facts). 'Still alive' means that player's team "
            "reached the quarterfinals, so their tally can still grow. "
            "'Projected' extrapolates each still-alive player's current "
            "per-game rate over their team's expected remaining games (from "
            "the aware model's Monte Carlo) - a simple estimate, not a full "
            "per-opponent scoring model; see model/awards.py."
        ),
        "golden_boot": gb_rows,
        "golden_boot_projected": _rerank_by_projection(gb_rows, "goals"),
        "top_assists": ta_rows,
        "top_assists_projected": _rerank_by_projection(ta_rows, "assists"),
        "golden_glove": gg_rows,
        "golden_glove_projected": _rerank_by_projection(gg_rows, "clean_sheets"),
    }


if __name__ == "__main__":
    # Standalone sanity check with made-up survival probabilities (real ones
    # come from predict.py's Monte Carlo at pipeline run time).
    demo_mc = {
        "Argentina": {"sf_pct": 62.0, "final_pct": 38.0, "champion_pct": 24.5},
        "France": {"sf_pct": 66.0, "final_pct": 41.0, "champion_pct": 25.2},
        "Norway": {"sf_pct": 41.0, "final_pct": 12.0, "champion_pct": 5.0},
        "England": {"sf_pct": 59.0, "final_pct": 27.0, "champion_pct": 11.3},
        "Spain": {"sf_pct": 69.0, "final_pct": 36.0, "champion_pct": 19.3},
        "Belgium": {"sf_pct": 31.0, "final_pct": 12.0, "champion_pct": 3.7},
        "Morocco": {"sf_pct": 33.0, "final_pct": 10.0, "champion_pct": 7.0},
        "Switzerland": {"sf_pct": 30.0, "final_pct": 10.0, "champion_pct": 4.1},
    }
    a = build_awards(demo_mc)
    print("=== Golden Boot (current) ===")
    for p in a["golden_boot"]:
        alive = "still alive" if p["still_alive"] else "eliminated"
        print(f"  {p['rank']:>2}. {p['player']:22s} ({p['team']:10s}) {p['goals']} goals -> "
              f"projected {p['projected_goals']:.1f} [{alive}, ~{p['expected_remaining_games']:.1f} games left]")

    print("\n=== Golden Boot (re-ranked by projection) ===")
    for p in a["golden_boot_projected"]:
        print(f"  {p['projected_rank']:>2}. {p['player']:22s} projected {p['projected_goals']:.1f} "
              f"(currently {p['goals']}, rank {p['rank']})")

    print("\n=== Top Assists (re-ranked by projection) ===")
    for p in a["top_assists_projected"]:
        print(f"  {p['projected_rank']:>2}. {p['player']:22s} projected {p['projected_assists']:.1f} "
              f"(currently {p['assists']}, rank {p['rank']})")

    print("\n=== Golden Glove (re-ranked by projection) ===")
    for p in a["golden_glove_projected"]:
        print(f"  {p['projected_rank']:>2}. {p['player']:22s} projected {p['projected_clean_sheets']:.1f} "
              f"(currently {p['clean_sheets']}, rank {p['rank']})")
