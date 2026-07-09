# Data source note

**Plan deviation:** PROJECT_PLAN.md §3 called for pulling `matches`/`teams`/
`groups` directly from `github.com/rezarahiminia/worldcup2026`. This sandbox
has no network access to GitHub or general websites (allowlist blocks
everything except anthropic.com/claude.com domains), and the Chrome
extension bridge was unavailable too. `matches.csv` and `teams.csv` were
instead reconstructed from `WebSearch` result snippets (news coverage,
official group tables, score reports) as of 2026-07-08.

## Known gaps / uncertainties

- A few group-stage point/goal-difference totals from one aggregator
  disagreed by exactly 1 point/goal with the directly-sourced match scores
  (e.g. Canada, Belgium, Saudi Arabia). The match-level scores were treated
  as authoritative; any derived standings should be recomputed from
  `matches.csv`, not copied from a secondary table.
- Kickoff times and venues were not collected (not needed by the model).
- One Round of 32 game (USA v Bosnia) had ambiguous phrasing about extra
  time; recorded as `went_to_et=False` based on the goal-minute evidence
  (45', 82') not requiring it.
- No results are invented for the quarterfinals onward — those are exactly
  what the model is meant to predict.

## Coverage

- 48 teams, 96 matches (72 group + 16 Round of 32 + 8 Round of 16).
- All 8 confirmed quarterfinalists (France, Morocco, Spain, Belgium,
  Argentina, Switzerland, England, Norway) have complete, cross-checked
  group + R32 + R16 paths.

If GitHub access is enabled later (Settings → Capabilities) or the Chrome
extension reconnects, this data can be re-pulled from the reference repo
and diffed against these CSVs as a accuracy check.
