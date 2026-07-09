"""
Milestone 2.

Reads raw match/team files from data/raw/ (compiled via WebSearch — GitHub
and general web fetches are not reachable from this environment, so the
originally-planned direct pull from the rezarahiminia/worldcup2026 repo was
replaced with a manually-verified reconstruction; see data/raw/SOURCE.md)
plus the manual CSVs in data/manual/ (FIFA ranking, titles, storylines,
added in Milestone 3), and produces tidy pandas DataFrames.

Entry points:
    load_matches() -> pd.DataFrame        # one row per match, wide format
    matches_long(matches) -> pd.DataFrame  # one row per team per match
    load_teams() -> pd.DataFrame
    load_manual_tables() -> dict[str, pd.DataFrame]
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
MANUAL_DIR = REPO_ROOT / "data" / "manual"


def load_matches() -> pd.DataFrame:
    """Load data/raw/matches.csv as-is: one row per match (team_a vs team_b)."""
    path = RAW_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    bool_cols = ["went_to_et", "went_to_pens"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def matches_long(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Reshape matches.csv into one row per team per match (the "tidy" form
    features.py will aggregate over): team, opponent, stage, gf, ga, gd,
    result (win/draw/loss), went_to_et, went_to_pens.
    """
    if matches is None:
        matches = load_matches()

    a = matches.rename(
        columns={
            "team_a": "team",
            "team_b": "opponent",
            "score_a": "gf",
            "score_b": "ga",
        }
    ).copy()
    b = matches.rename(
        columns={
            "team_b": "team",
            "team_a": "opponent",
            "score_b": "gf",
            "score_a": "ga",
        }
    ).copy()

    long_df = pd.concat([a, b], ignore_index=True)
    long_df["gd"] = long_df["gf"] - long_df["ga"]

    def result(row):
        # Use the authoritative `winner` column, not gf/ga - a penalty-
        # shootout win still shows as a 1-1 (or 0-0) scoreline, which would
        # otherwise be misread as a draw for a knockout match that actually
        # had a decisive winner.
        if row["winner"] == "draw":
            return "draw"
        return "win" if row["team"] == row["winner"] else "loss"

    long_df["result"] = long_df.apply(result, axis=1)
    cols = [
        "date", "stage", "team", "opponent", "gf", "ga", "gd", "result",
        "went_to_et", "went_to_pens",
    ]
    return long_df[cols].sort_values(["team", "date"]).reset_index(drop=True)


def load_teams() -> pd.DataFrame:
    """Load data/raw/teams.csv: one row per team, group + how far they went."""
    path = RAW_DIR / "teams.csv"
    df = pd.read_csv(path)
    bool_cols = ["advanced_to_r32", "advanced_to_r16", "advanced_to_qf"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def load_manual_tables() -> dict[str, pd.DataFrame]:
    """
    Load every CSV in data/manual/ (fifa_ranking.csv, titles.csv,
    storylines.csv — added in Milestone 3) keyed by filename stem.
    Returns {} gracefully if the folder has no CSVs yet.
    """
    tables = {}
    if not MANUAL_DIR.exists():
        return tables
    for csv_path in sorted(MANUAL_DIR.glob("*.csv")):
        tables[csv_path.stem] = pd.read_csv(csv_path)
    return tables


if __name__ == "__main__":
    # Milestone sanity check (PROJECT_PLAN.md #10): print shapes and heads.
    m = load_matches()
    t = load_teams()
    ml = matches_long(m)
    manual = load_manual_tables()

    print(f"matches.csv: {len(m)} rows")
    print(m.head(), "\n")

    print(f"teams.csv: {len(t)} rows")
    print(t.head(), "\n")

    print(f"matches_long: {len(ml)} rows (should be 2x matches = {2 * len(m)})")
    assert len(ml) == 2 * len(m), "matches_long row count mismatch"
    print(ml.head(), "\n")

    qf_teams = t.loc[t["advanced_to_qf"], "team"].tolist()
    print(f"QF teams ({len(qf_teams)}):", qf_teams)
    assert len(qf_teams) == 8, "expected exactly 8 QF teams"

    print(f"\nmanual tables loaded: {list(manual.keys()) or '(none yet — Milestone 3)'}")
