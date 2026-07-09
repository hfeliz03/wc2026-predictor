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
  manual/        fifa_ranking.csv, titles.csv, storylines.csv
model/
  load_data.py   read raw + manual, build team table
  features.py    feature engineering (single source of truth)
  train.py       backbone weights, graded on all completed R32+R16 games
  predict.py     QF -> SF -> Final + Monte Carlo -> predictions.json
  evaluate.py    accuracy / Brier / log-loss, updated as games finish
  config.py      all tunable constants in one place
site/            index.html, styles.css, app.js, predictions.json
docs/            published build for GitHub Pages
```

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
