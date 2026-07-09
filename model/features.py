"""
Milestone 4.

Single source of truth for feature engineering (PROJECT_PLAN.md #4).

Per-team features (see FEATURE_COLUMNS):
    - fifa_points                pre-tournament FIFA ranking points
    - wc_titles, major_trophies  historical pedigree
    - narrative                  manual storyline score (-2..+2)
    - group_pts, group_gd        group-stage form
    - rank_vs_expectation        did they over/underperform their seed within
                                  their own group? (+ve = overperformed)
    - ko_wins, ko_losses, ko_gd  knockout-round form SO FAR (computed as of
                                  a cutoff date, to avoid leaking the result
                                  of the match being predicted)
    - ko_fragile                 count of knockout games so far that needed
                                  extra time or penalties (fragility proxy)
    - is_host                    1 if USA / Canada / Mexico, else 0

Per-match features: teamA_features - teamB_features (prefixed d_), built by
differenced_features() for a single hypothetical matchup, or in bulk by
build_match_features() for a set of already-played matches (used to build
the train/validate frames in train.py).

IMPORTANT — no leakage: ko_wins/ko_losses/ko_gd/ko_fragile are computed only
from matches strictly *before* the match being featurized, via the
`before_date` cutoff. Group-stage and pre-tournament features are static
(all group games finish before any knockout game), so they carry no leakage
risk.
"""

from __future__ import annotations

import pandas as pd

import load_data
from config import HOST_NATIONS

STATIC_FEATURE_COLUMNS = [
    "fifa_points",
    "wc_titles",
    "major_trophies",
    "narrative",
    "group_pts",
    "group_gd",
    "rank_vs_expectation",
]
KNOCKOUT_FORM_COLUMNS = ["ko_wins", "ko_losses", "ko_gd", "ko_fragile"]
FEATURE_COLUMNS = STATIC_FEATURE_COLUMNS + KNOCKOUT_FORM_COLUMNS + ["is_host"]


def _group_stage_agg(matches_long: pd.DataFrame) -> pd.DataFrame:
    """Per-team group-stage points, GF/GA, goal difference."""
    g = matches_long[matches_long["stage"] == "Group"].copy()
    pts_map = {"win": 3, "draw": 1, "loss": 0}
    g["pts"] = g["result"].map(pts_map)
    agg = g.groupby("team").agg(
        group_pts=("pts", "sum"),
        group_gf=("gf", "sum"),
        group_ga=("ga", "sum"),
    )
    agg["group_gd"] = agg["group_gf"] - agg["group_ga"]
    return agg[["group_pts", "group_gf", "group_ga", "group_gd"]]


def _rank_vs_expectation(teams: pd.DataFrame, fifa: pd.DataFrame) -> pd.Series:
    """
    Within each 4-team group, rank teams by fifa_rank (1 = strongest seed) to
    get an "expected" finish order, then compare to actual group_position.
    rank_vs_expectation = expected_position - actual_position:
        +ve => finished better than their seed predicted (overperformed)
        -ve => finished worse than their seed predicted (underperformed)
    """
    t = teams.merge(fifa[["team", "fifa_rank"]], on="team", how="left")
    t["expected_position"] = t.groupby("group")["fifa_rank"].rank(method="first").astype(int)
    position_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    t["actual_position"] = t["group_position"].map(position_map)
    t["rank_vs_expectation"] = t["expected_position"] - t["actual_position"]
    return t.set_index("team")["rank_vs_expectation"]


def build_team_table() -> pd.DataFrame:
    """
    One row per team, indexed by team name, with every static feature in
    STATIC_FEATURE_COLUMNS plus current (as-of-today) knockout form and
    is_host. This is the "current snapshot" predict.py uses for QF onward —
    for historical training/validation rows, use differenced_features()
    with an explicit before_date instead, so knockout form doesn't leak.
    """
    teams = load_data.load_teams()
    matches = load_data.load_matches()
    ml = load_data.matches_long(matches)
    manual = load_data.load_manual_tables()

    fifa = manual["fifa_ranking"]
    titles = manual["titles"]
    storylines = manual["storylines"]

    table = teams.set_index("team")[["group", "group_position", "eliminated_in"]]
    table = table.join(fifa.set_index("team")[["fifa_rank", "fifa_points", "major_trophies"]])
    table = table.join(titles.set_index("team")[["wc_titles"]])
    table = table.join(storylines.set_index("team")[["narrative"]])
    table = table.join(_group_stage_agg(ml))
    table["rank_vs_expectation"] = _rank_vs_expectation(teams, fifa)

    # Knockout form as of "now" (all played knockout matches so far).
    now_cutoff = matches["date"].max() + pd.Timedelta(days=1)
    ko = {team: knockout_form(team, ml, before_date=now_cutoff) for team in table.index}
    ko_df = pd.DataFrame(ko).T[KNOCKOUT_FORM_COLUMNS]
    table = table.join(ko_df)

    table["is_host"] = table.index.isin(HOST_NATIONS).astype(int)

    return table


def knockout_form(team: str, matches_long: pd.DataFrame, before_date) -> dict:
    """
    Cumulative knockout-round (non-Group) form for `team`, using only
    matches strictly before `before_date`. Returns zeros if the team hasn't
    played a knockout match yet as of that date (e.g. any team, evaluated
    just before Round of 32 kicks off).
    """
    m = matches_long[
        (matches_long["team"] == team)
        & (matches_long["stage"] != "Group")
        & (matches_long["date"] < before_date)
    ]
    return {
        "ko_wins": int((m["result"] == "win").sum()),
        "ko_losses": int((m["result"] == "loss").sum()),
        "ko_gd": int(m["gd"].sum()),
        "ko_fragile": int((m["went_to_et"] | m["went_to_pens"]).sum()),
    }


def differenced_features(
    team_a: str,
    team_b: str,
    before_date,
    static_table: pd.DataFrame,
    matches_long: pd.DataFrame,
) -> dict:
    """
    Build the differenced (team_a - team_b) feature vector for a single
    matchup, as of `before_date`. static_table must be indexed by team and
    contain STATIC_FEATURE_COLUMNS (use build_team_table(), or a variant
    with knockout-form columns overwritten — static columns don't change
    over time so build_team_table()'s static half is always valid).
    """
    row = {}
    a_static = static_table.loc[team_a, STATIC_FEATURE_COLUMNS]
    b_static = static_table.loc[team_b, STATIC_FEATURE_COLUMNS]
    for col in STATIC_FEATURE_COLUMNS:
        row[f"d_{col}"] = a_static[col] - b_static[col]

    a_ko = knockout_form(team_a, matches_long, before_date)
    b_ko = knockout_form(team_b, matches_long, before_date)
    for col in KNOCKOUT_FORM_COLUMNS:
        row[f"d_{col}"] = a_ko[col] - b_ko[col]

    row["d_is_host"] = int(team_a in HOST_NATIONS) - int(team_b in HOST_NATIONS)
    return row


def build_match_features(
    stages: tuple[str, ...] = ("Round of 32", "Round of 16"),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build the training-ready (X, y, meta) frames for every already-played
    match in `stages` (default: R32 + R16, the ~24 knockout games used to
    calibrate and validate the model per PROJECT_PLAN.md #5).

    X: one row per match, differenced features (d_-prefixed), teamA - teamB.
    y: 1 if team_a won, 0 if team_b won. Draws are dropped (knockout matches
       are always decisive; a stray Group-stage draw sneaking into `stages`
       would otherwise poison the label).
    meta: date, stage, team_a, team_b, winner — for readable reporting.
    """
    matches = load_data.load_matches()
    ml = load_data.matches_long(matches)
    static_table = build_team_table()  # static columns only; ko columns unused here

    subset = matches[matches["stage"].isin(stages)].copy()
    subset = subset[subset["winner"] != "draw"].reset_index(drop=True)

    feature_rows = []
    for _, m in subset.iterrows():
        feats = differenced_features(m["team_a"], m["team_b"], m["date"], static_table, ml)
        feature_rows.append(feats)

    X = pd.DataFrame(feature_rows)
    y = (subset["winner"] == subset["team_a"]).astype(int).rename("team_a_won")
    meta = subset[["date", "stage", "team_a", "team_b", "winner"]].reset_index(drop=True)
    return X, y, meta


if __name__ == "__main__":
    # Milestone sanity check (PROJECT_PLAN.md #10).
    team_table = build_team_table()
    print(f"team_table: {team_table.shape}")
    print(team_table.loc[["Spain", "Argentina", "Norway", "Portugal"]], "\n")

    X, y, meta = build_match_features()
    print(f"match features: X={X.shape}, y={y.shape}")
    assert len(X) == len(y) == len(meta)
    print("feature columns:", list(X.columns), "\n")

    by_stage = meta["stage"].value_counts()
    print("rows by stage:\n", by_stage, "\n")
    assert by_stage.get("Round of 32", 0) == 16
    assert by_stage.get("Round of 16", 0) == 8

    # Eyeball one upset game: Norway beat Brazil in the R16.
    idx = meta[(meta["team_a"] == "Brazil") & (meta["team_b"] == "Norway")].index
    if len(idx):
        i = idx[0]
        print("Brazil vs Norway (R16, Norway won) feature row:")
        print(X.loc[i])
        print("label (team_a_won):", y.loc[i], "(0 = Brazil lost, correct)")
