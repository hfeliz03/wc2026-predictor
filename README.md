# WC2026 Knockout Predictor

Predicts World Cup 2026 quarterfinal, semifinal, and final outcomes using a
transparent rating model calibrated with a lightly-regularized logistic
regression, validated against the Round of 32 and Round of 16 results.
Presented as a static, GitHub-Pages-hosted bracket site.

See `PROJECT_PLAN.md` (repo root, one level up) for the full design spec.

## Structure

```
data/
  raw/           pulled from the reference repo (matches, teams, groups)
  manual/        fifa_ranking.csv, titles.csv, storylines.csv,
                 golden_boot.csv, top_assists.csv
model/
  load_data.py   read raw + manual, build team table
  features.py    feature engineering (single source of truth)
  train.py       backbone weights, graded on all completed R32+R16 games
  predict.py     QF -> SF -> Final + Monte Carlo -> predictions.json
  evaluate.py    accuracy / Brier / log-loss, updated as games finish
  awards.py      Golden Boot + top-assists leaderboard (not predictive -
                 straight standings through the Round of 16)
  config.py      all tunable constants in one place
site/            index.html, styles.css, app.js, predictions.json
docs/            published build for GitHub Pages
```

## Awards (Golden Boot / top assists)

`data/manual/golden_boot.csv` and `top_assists.csv` are standings through
the Round of 16, compiled via WebSearch — this is factual reporting, not a
prediction (forecasting future goals would need a different kind of model
than the win-probability backbone used elsewhere here). Rerun the same
WebSearch-based research after each round to update these two files, then
rerun `predict.py`.

**No real player photos**: this sandbox can't fetch images from outside
`*.anthropic.com`/`claude.com` (same restriction that blocked GitHub
earlier), and the Chrome browser bridge wasn't connected either. Each
player gets a deterministic-colored initials avatar instead. To swap in
real headshots: drop image files in `site/images/players/` (e.g.
`lionel-messi.jpg`) and update `app.js`'s `podiumCard()`/`awardRow()` to
render an `<img>` when a matching file exists, falling back to the avatar
otherwise.

**Projected winner**: each still-alive player's current per-game rate is
extrapolated over their team's expected remaining games (from the aware
model's Monte Carlo `sf_pct`/`final_pct`) — see `model/awards.py`'s module
docstring for the exact formula and its limits (constant-rate assumption,
no per-opponent difficulty modeling). Eliminated players' projections equal
their current tally, since it's frozen. The podium and table both rank by
this projection; the table still shows the current (non-projected) number
alongside it.

## Cache-busting

`index.html` loads `app.js`/`styles.css` with a `?v=N` query param, and
`app.js` fetches `predictions.json` with `?_=<timestamp>` and `cache:
"no-store"`. Browsers (and GitHub Pages' CDN) can otherwise serve a stale
`app.js` alongside fresh HTML/data, which is confusing in a very specific
way: old rendering code silently misaligns with new data/markup rather
than erroring, so a table column can end up showing the wrong stat
entirely. **Bump the `?v=` number in `index.html` any time `app.js` or
`styles.css` changes.** If something looks visually stale despite a fresh
push, hard-refresh (Cmd+Shift+R / Ctrl+Shift+R) before assuming it's a code
bug.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline order

1. `model/load_data.py` — build tidy match/team tables from `data/raw` + `data/manual`
2. `model/features.py` — engineer per-team and per-match features
3. `model/train.py` — calibrate the backbone (hand-set weights, graded on all
   24 completed R32+R16 games), print the blind-vs-aware R16 comparison,
   save `model/trained_weights.json`
4. `model/predict.py` — generate `site/predictions.json`
5. View the site (see below)

## Viewing the site locally

`app.js` fetches `predictions.json` over `fetch()`, which most browsers
block for pages opened directly from disk (`file://...`) due to CORS. Serve
the folder instead:

```bash
cd site && python3 -m http.server 8000
```

then open `http://localhost:8000`. This isn't an issue once deployed to
GitHub Pages — pages served over `https://` don't hit this restriction.

## The two models

Every prediction is shown for two models, side by side, with a toggle on
the site to switch which one drives the bracket's winner highlighting:

- **Blind** — pre-tournament + group-stage signal only. Never touches any
  knockout-round outcome. Graded at 75% accuracy (18/24) on every completed
  R32+R16 game, with zero fitting to those outcomes — a true held-out test.
- **Aware** — adds each team's already-completed knockout-round form
  (wins/losses/goal difference/penalty-shootout fragility). This uses real,
  already-known results to inform later rounds, which is forward-looking
  forecasting, not leakage — a match's own outcome never informs its own
  prediction. An apples-to-apples check on the 8 R16 games (the only round
  where prior knockout form exists) found this barely moved any pick with
  n=8, so both models are shown rather than one being presented as
  definitive. See `model/train.py`'s module docstring for the full
  reasoning.

## Recording results as they happen (QF / SF / Final)

Each round's predictions are locked in `model/predictions_history/*_locked.json`
the first time `predict.py` generates them, before that round has any real
results — this is what grading compares against, so a game's own outcome
can never leak into its own prediction (see `model/evaluate.py`'s module
docstring). After each game finishes:

```bash
cd model
python3 -c "
import evaluate
evaluate.record_result('2026-07-09', 'Quarterfinal', 'France', 'Morocco', 2, 0)
"
```

`stage` must be exactly `"Quarterfinal"`, `"Semifinal"`, or `"Final"`. If a
game was decided on penalties (scores level), pass `winner=` explicitly —
it can't be inferred from the scoreline alone:

```bash
python3 -c "
import evaluate
evaluate.record_result('2026-07-11', 'Quarterfinal', 'Argentina', 'Switzerland', 0, 0, went_to_pens=True, winner='Argentina')
"
```

Then regenerate and republish:

```bash
python3 predict.py       # regenerates site/predictions.json; auto-locks the next round
                          # once its fixture is fully determined by real results
python3 evaluate.py      # prints the live grading report to the terminal
cd .. && ./scripts/sync_docs.sh
git add -A && git commit -m "QF: France beat Morocco 2-0" && git push
```

The site's scorecard and matchup cards update automatically from
`predictions.json` — no code changes needed each round.
