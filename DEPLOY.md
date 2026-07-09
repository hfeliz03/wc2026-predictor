# Deploying to GitHub Pages

I couldn't run git inside this project from my sandbox (the folder is
synced/mounted, and git's atomic file operations aren't supported over
that mount - you may see a leftover, broken `.git/` folder here from a
failed attempt; delete it first if so: `rm -rf .git`). Everything else is
ready to go. Run these from your own Terminal:

## 1. Clean up and initialize

```bash
cd "/Users/vilma/Documents/Claude/Projects/WC2026 Final prediction/wc2026-predictor"
rm -rf .git   # only needed if a partial one exists from my attempt
git init
git branch -m main
git add -A
git commit -m "Initial commit: WC2026 knockout predictor pipeline + site"
```

## 2. Create the GitHub repo and push

Create an empty repo on GitHub (no README/license, so it stays empty) -
e.g. `wc2026-predictor` - then:

```bash
git remote add origin https://github.com/<your-username>/wc2026-predictor.git
git push -u origin main
```

## 3. Enable GitHub Pages

On GitHub: **Settings -> Pages -> Build and deployment -> Source: Deploy
from a branch -> Branch: `main`, folder: `/docs` -> Save**.

Wait a minute or two, then visit:

```
https://<your-username>.github.io/wc2026-predictor/
```

## 4. Publishing updates later

`site/` is the source you edit; `docs/` is what GitHub Pages actually
serves, and the two need to stay in sync manually (no build step). After
rerunning `model/predict.py` (e.g. once QF results are in, per Milestone
9), sync and push:

```bash
./scripts/sync_docs.sh
git add -A
git commit -m "Update predictions"
git push
```

## Verifying it worked

- The repo on GitHub should show the same file tree as this folder.
- `https://<your-username>.github.io/wc2026-predictor/` should load the
  bracket, matchup cards, scorecard, and Monte Carlo table - no local
  server needed once it's on Pages, since `https://` doesn't hit the
  `file://` CORS restriction we ran into locally.
- If the page loads but looks unstyled/broken, check the browser console
  for a 404 on `styles.css`/`app.js`/`predictions.json` - usually means
  `docs/` is missing a file or the Pages folder setting is wrong.
