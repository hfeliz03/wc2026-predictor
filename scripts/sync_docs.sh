#!/usr/bin/env bash
# Mirrors site/ (source) into docs/ (GitHub Pages serves from /docs).
# Run this after any predict.py rerun that updates site/predictions.json.
set -euo pipefail
cd "$(dirname "$0")/.."
cp site/index.html site/styles.css site/app.js site/predictions.json docs/
echo "Synced site/ -> docs/"
