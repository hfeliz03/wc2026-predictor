"""
Stretch goal (PROJECT_PLAN.md #7/#9): Golden Boot + top-assists leaderboard.

This is NOT a predictive model - group-stage-through-R16 goals/assists are
already-known facts, not something to forecast with the backbone/logistic
machinery used elsewhere in this project. It's a straight leaderboard, read
from data/manual/golden_boot.csv and data/manual/top_assists.csv (compiled
via WebSearch - see those files' provenance note below), joined against
teams.csv to flag whether each player's team is still alive (their tally
can still grow) or eliminated (their tally is final).

Entry points:
    build_awards() -> dict  # {"golden_boot": [...], "top_assists": [...]}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import load_data

REPO_ROOT = Path(__file__).resolve().parent.parent
MANUAL_DIR = REPO_ROOT / "data" / "manual"


def _with_alive_flag(df: pd.DataFrame) -> list:
    teams = load_data.load_teams().set_index("team")
    out = []
    for _, row in df.iterrows():
        team = row["team"]
        still_alive = bool(teams.loc[team, "advanced_to_qf"]) if team in teams.index else False
        out.append({**row.to_dict(), "still_alive": still_alive})
    return out


def build_awards() -> dict:
    golden_boot = pd.read_csv(MANUAL_DIR / "golden_boot.csv")
    top_assists = pd.read_csv(MANUAL_DIR / "top_assists.csv")
    return {
        "note": (
            "Standings through the Round of 16 (not a prediction - these are "
            "already-known facts). 'Still alive' means that player's team "
            "reached the quarterfinals, so their tally can still grow."
        ),
        "golden_boot": _with_alive_flag(golden_boot),
        "top_assists": _with_alive_flag(top_assists),
    }


if __name__ == "__main__":
    a = build_awards()
    print("=== Golden Boot ===")
    for p in a["golden_boot"]:
        alive = "still alive" if p["still_alive"] else "eliminated"
        print(f"  {p['rank']:>2}. {p['player']:22s} ({p['team']:10s}) {p['goals']} goals, {p['assists']} assists [{alive}]")
    print("\n=== Top Assists ===")
    for p in a["top_assists"]:
        alive = "still alive" if p["still_alive"] else "eliminated"
        print(f"  {p['rank']:>2}. {p['player']:22s} ({p['team']:10s}) {p['assists']} assists, {p['goals']} goals [{alive}]")
