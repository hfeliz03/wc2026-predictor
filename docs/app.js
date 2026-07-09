// WC2026 Knockout Predictor - reads predictions.json, renders bracket,
// matchup cards, scorecard, and Monte Carlo table. No build step, no
// framework - vanilla JS against a frozen JSON schema (see PROJECT_PLAN.md
// #6). Two models are shown throughout: "blind" (pre-tournament +
// group-stage signal only) and "aware" (adds each team's already-completed
// knockout-round form). See model/train.py for why both are shown rather
// than one being picked as definitive.

// Real flag graphics via the flag-icons library (see index.html <link>) instead
// of emoji, which render inconsistently (or as raw country codes) across OSes.
// Codes are ISO 3166-1 alpha-2, except England which uses flag-icons' regional
// "gb-eng" code.
const FLAGS = {
  "France": "fr", "Morocco": "ma",
  "Spain": "es", "Belgium": "be",
  "Norway": "no", "England": "gb-eng",
  "Argentina": "ar", "Switzerland": "ch",
  "Brazil": "br", "Portugal": "pt",
  "Mexico": "mx", "USA": "us",
  "Canada": "ca", "Egypt": "eg",
  "Colombia": "co", "Germany": "de",
  "Netherlands": "nl", "Paraguay": "py",
  "Sweden": "se", "Austria": "at",
  "Algeria": "dz", "Cape Verde": "cv",
  "Senegal": "sn", "DR Congo": "cd",
  "Ivory Coast": "ci", "Croatia": "hr",
  "Ghana": "gh", "Panama": "pa",
  "Bosnia and Herzegovina": "ba", "Uzbekistan": "uz",
  "Australia": "au", "Japan": "jp",
};
function flag(team) {
  const code = FLAGS[team];
  return code
    ? `<span class="fi fi-${code} flag-icon" title="${team}"></span>`
    : `<span class="flag-fallback" title="${team}">⚽</span>`;
}

let DATA = null;
let MODEL = "aware";

async function main() {
  try {
    // Cache-bust: predictions.json changes every round, so it must never be
    // served stale from browser/CDN cache (this bit us once already - a
    // stale cached app.js earlier in the day caused genuinely confusing
    // mis-rendered output, e.g. an "assists" value appearing under a
    // "Projected" header). no-store + a timestamp param covers both
    // fetch-level and URL-level caching.
    const res = await fetch("./predictions.json?_=" + Date.now(), { cache: "no-store" });
    DATA = await res.json();
  } catch (e) {
    document.getElementById("hero-sub").textContent =
      "Couldn't load predictions.json (if you opened this file directly, try running a local server instead - see README).";
    return;
  }
  render();
  positionToggleIndicator();
  document.getElementById("btn-aware").addEventListener("click", () => setModel("aware"));
  document.getElementById("btn-blind").addEventListener("click", () => setModel("blind"));
  window.addEventListener("resize", positionToggleIndicator);
}

function setModel(model) {
  MODEL = model;
  document.getElementById("btn-aware").classList.toggle("active", model === "aware");
  document.getElementById("btn-blind").classList.toggle("active", model === "blind");
  positionToggleIndicator();
  render();
}

// Slides the little pill behind the active Aware/Blind button instead of
// just swapping a background color, so switching models feels like a tab
// transition rather than a hard cut.
function positionToggleIndicator() {
  const active = document.querySelector(".toggle-btn.active");
  const indicator = document.getElementById("toggle-indicator");
  if (!active || !indicator) return;
  indicator.style.width = active.offsetWidth + "px";
  indicator.style.left = active.offsetLeft + "px";
  indicator.classList.toggle("aware", active.dataset.model === "aware");
  indicator.classList.toggle("blind", active.dataset.model === "blind");
}

// Probability/mini bars are rendered at width:0 in the HTML string, then
// animated up to their real width on the next frame so the transition in
// CSS actually plays instead of snapping straight to the final value.
function animateBars(root = document) {
  const bars = root.querySelectorAll("[data-w]");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bars.forEach(el => { el.style.width = el.dataset.w + "%"; });
    });
  });
}

function render() {
  renderHero();
  renderChampions();
  renderBracket();
  renderMatchups();
  renderScorecard();
  renderMonteCarlo();
  renderAwards();
  renderFooter();
}

function renderHero() {
  document.getElementById("hero-sub").textContent =
    `Generated ${DATA.generated_at} · backbone rating + logistic calibration, graded on all 24 completed Round of 32 + Round of 16 games.`;
  document.getElementById("toggle-help").textContent = MODEL === "aware"
    ? "Aware: pre-tournament + group-stage signal, plus each team's already-completed knockout form (R32 + R16). Using real, already-known results to inform later rounds - not leakage."
    : "Blind: pre-tournament + group-stage signal only. Never touches any knockout-round outcome. This is the version graded at 75% accuracy on all 24 R32+R16 games.";
}

function renderChampions() {
  const el = document.getElementById("champion-row");
  const mc = DATA.monte_carlo;
  const blindPct = mc.blind[DATA.champion_blind]?.champion_pct ?? 0;
  const awarePct = mc.aware[DATA.champion_aware]?.champion_pct ?? 0;
  el.innerHTML = `
    <div class="champion-badge blind">
      <span class="label">Blind picks</span>
      <span class="team">${flag(DATA.champion_blind)} ${DATA.champion_blind}</span>
      <span class="pct">${blindPct}% (MC)</span>
    </div>
    <div class="champion-badge aware">
      <span class="label">Aware picks</span>
      <span class="team">${flag(DATA.champion_aware)} ${DATA.champion_aware}</span>
      <span class="pct">${awarePct}% (MC)</span>
    </div>
  `;
}

function pctFor(game, side) {
  // side: 'a' or 'b'. returns the % for the currently selected model.
  const key = (side === "a" ? "pA_" : "pB_") + MODEL;
  return Math.round(game[key] * 100);
}
function pickFor(game) {
  return game["predicted_" + MODEL];
}

function bracketTeamRow(team, pct, isWinner) {
  return `<div class="bracket-team ${isWinner ? "winner" : ""}">
    <div class="bracket-team-row">
      <span class="name"><span class="flag">${flag(team)}</span>${team}</span>
      <span class="pct">${pct}%</span>
    </div>
    <div class="mini-bar"><div class="mini-fill" data-w="${pct}" style="width:0%"></div></div>
  </div>`;
}

function renderBracket() {
  const qf = DATA.quarterfinals;
  const sf = DATA.semifinals;
  const final = DATA.final;

  const qfCol = qf.map((g, i) => {
    const winner = pickFor(g);
    return `<div class="bracket-card" style="--i:${i}">
      ${bracketTeamRow(g.team_a, pctFor(g, "a"), winner === g.team_a)}
      ${bracketTeamRow(g.team_b, pctFor(g, "b"), winner === g.team_b)}
    </div>`;
  }).join("");

  const sfCol = sf.map((g, i) => {
    const winner = pickFor(g);
    return `<div class="bracket-card" style="--i:${i + 1}">
      ${bracketTeamRow(g.team_a, pctFor(g, "a"), winner === g.team_a)}
      ${bracketTeamRow(g.team_b, pctFor(g, "b"), winner === g.team_b)}
    </div>`;
  }).join("");

  const finalWinner = pickFor(final);
  const finalCard = `<div class="bracket-card final-card" style="--i:2">
    ${bracketTeamRow(final.team_a, pctFor(final, "a"), finalWinner === final.team_a)}
    ${bracketTeamRow(final.team_b, pctFor(final, "b"), finalWinner === final.team_b)}
    <div style="text-align:center; margin-top:8px; font-size:22px;">🏆 ${flag(finalWinner)} ${finalWinner}</div>
  </div>`;

  document.getElementById("bracket").innerHTML = `
    <div class="bracket-col round-qf"><h3>Quarterfinals</h3>${qfCol}</div>
    <div class="bracket-col round-sf"><h3>Semifinals</h3>${sfCol}</div>
    <div class="bracket-col round-final"><h3>Final</h3>${finalCard}</div>
  `;
  animateBars(document.getElementById("bracket"));

  if (DATA.bracket_agrees === false) {
    document.getElementById("bracket").insertAdjacentHTML("beforebegin",
      `<p class="panel-sub" style="color:var(--loss)">Note: the blind and aware models advance different teams at some stage this round - the bracket above follows the aware model's path; champion badges above reflect each model's own independent path.</p>`);
  }
}

function resultBadge(g) {
  if (!g.played) return "";
  const blindTag = g.correct_blind
    ? `<span class="badge badge-blind ok">blind ✓</span>` : `<span class="badge badge-blind miss">blind ✗</span>`;
  const awareTag = g.correct_aware
    ? `<span class="badge badge-aware ok">aware ✓</span>` : `<span class="badge badge-aware miss">aware ✗</span>`;
  return `<div class="result-badge">
    <span class="ft-tag">FT</span> ${flag(g.actual_winner)} <b>${g.actual_winner}</b> won ${g.actual_score}
    ${blindTag}${awareTag}
  </div>`;
}

function matchupCard(g, stageLabel, i) {
  return `<div class="matchup-card${g.played ? " played" : ""}" style="--i:${i}">
    <span class="stage-tag">${stageLabel}${g.date ? " · " + g.date : ""}</span>
    <div class="matchup-teams">
      <span class="team-side side-a">${flag(g.team_a)}<span class="team-name">${g.team_a}</span></span>
      <span class="matchup-vs">VS</span>
      <span class="team-side side-b">${flag(g.team_b)}<span class="team-name">${g.team_b}</span></span>
    </div>

    ${resultBadge(g)}

    <div class="prob-row">
      <div class="prob-label"><span>Blind</span><span>${Math.round(g.pA_blind*100)}% / ${Math.round(g.pB_blind*100)}%</span></div>
      <div class="prob-bar blind"><div class="fill-a" data-w="${g.pA_blind*100}" style="width:0%"></div></div>
    </div>
    <div class="prob-row">
      <div class="prob-label"><span>Aware</span><span>${Math.round(g.pA_aware*100)}% / ${Math.round(g.pB_aware*100)}%</span></div>
      <div class="prob-bar aware"><div class="fill-a" data-w="${g.pA_aware*100}" style="width:0%"></div></div>
    </div>

    <div class="rationale"><b>Blind:</b> ${g.rationale_blind}</div>
    <div class="rationale"><b>Aware:</b> ${g.rationale_aware}</div>
  </div>`;
}

function renderMatchups() {
  let i = 0;
  const cards = [
    ...DATA.quarterfinals.map(g => matchupCard(g, "Quarterfinal", i++)),
    ...DATA.semifinals.map(g => matchupCard(g, "Semifinal", i++)),
    matchupCard(DATA.final, "Final", i++),
  ];
  const grid = document.getElementById("matchup-grid");
  grid.innerHTML = cards.join("");
  animateBars(grid);
}

function renderScorecard() {
  const v = DATA.validation;
  const r16 = v.r16_form_comparison || {};
  let i = 0;

  const liveCards = Object.entries(v.live_rounds || {}).map(([roundKey, g]) => `
    <div class="stat-card" style="--i:${i++}">
      <div class="stat-value">${g.aware.n_correct}/${g.n}</div>
      <div class="stat-label">${g.stage} so far, aware model (blind: ${g.blind.n_correct}/${g.n})${g.n < g.n_total_in_round ? ` — ${g.n}/${g.n_total_in_round} played` : ""}</div>
    </div>`).join("");

  document.getElementById("scorecard").innerHTML = `
    <div class="stat-card" style="--i:${i++}"><div class="stat-value">${Math.round(v.accuracy*100)}%</div><div class="stat-label">Accuracy on all ${v.n} completed R32+R16 games (blind model)</div></div>
    <div class="stat-card" style="--i:${i++}"><div class="stat-value">${v.n_correct}/${v.n}</div><div class="stat-label">Games called correctly</div></div>
    <div class="stat-card" style="--i:${i++}"><div class="stat-value">${r16.form_aware_n_correct ?? "?"}/${r16.n ?? 8}</div><div class="stat-label">R16-only, aware model (blind: ${r16.form_blind_n_correct ?? "?"}/${r16.n ?? 8}) - the round with the most upsets</div></div>
    ${liveCards}
    <div class="scorecard-note">${DATA.method_note}</div>
  `;
}

function renderMonteCarlo() {
  const mc = DATA.monte_carlo;
  const teams = Object.keys(mc.aware).sort((a,b) => mc.aware[b].champion_pct - mc.aware[a].champion_pct);
  const rows = teams.map((t, i) => {
    const b = mc.blind[t], a = mc.aware[t];
    return `<tr style="--i:${i}">
      <td>${flag(t)} ${t}</td>
      <td>${b.sf_pct}%</td><td>${a.sf_pct}%</td>
      <td>${b.final_pct}%</td><td>${a.final_pct}%</td>
      <td>${b.champion_pct}%</td><td>${a.champion_pct}%</td>
    </tr>`;
  }).join("");
  document.getElementById("mc-tbody").innerHTML = rows;
}

// No real player photos are available in this build (see README) - each
// player instead gets a deterministic-colored initials avatar so the same
// player always renders the same way across reloads.
const AVATAR_COLORS = ["#38bdf8", "#fb923c", "#34d399", "#f472b6", "#a78bfa", "#f4c542", "#fb7185", "#4ade80"];
function avatarColor(name) {
  let hash = 0;
  for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
}
function initials(name) {
  const parts = name.trim().split(/\s+/);
  return ((parts[0]?.[0] || "") + (parts[parts.length - 1]?.[0] || "")).toUpperCase();
}

const STAT_LABELS = { goals: "goals", assists: "assists", clean_sheets: "clean sheets" };

function podiumCard(p, statKey, i) {
  const projKey = "projected_" + statKey;
  const statLabel = STAT_LABELS[statKey] || statKey;
  const changedRank = p.projected_rank !== p.rank;
  const aliveTag = p.still_alive
    ? `<span class="alive-tag alive">still alive</span>` : `<span class="alive-tag out">eliminated</span>`;
  return `<div class="podium-card rank-${p.projected_rank}" style="--i:${i}">
    <div class="rank-badge">#${p.projected_rank}${changedRank ? ` <span class="rank-was">(now #${p.rank})</span>` : ""}</div>
    <div class="avatar-wrap">
      <div class="avatar" style="background:${avatarColor(p.player)}">${initials(p.player)}</div>
      <div class="avatar-flag">${flag(p.team)}</div>
    </div>
    <div class="podium-name">${p.player}</div>
    <div class="podium-team">${p.team}</div>
    <div class="podium-stat">${p[projKey].toFixed(1)} <span>projected ${statLabel}</span></div>
    <div class="podium-current">currently ${p[statKey]}</div>
    ${aliveTag}
  </div>`;
}

function awardRow(p, statKey, otherKey) {
  const projKey = "projected_" + statKey;
  return `<tr>
    <td><div class="avatar avatar-sm" style="background:${avatarColor(p.player)}">${initials(p.player)}</div></td>
    <td>${p.player}</td>
    <td>${flag(p.team)} ${p.team}</td>
    <td>${p[statKey]}</td>
    <td class="projected-cell">${p[projKey].toFixed(1)}</td>
    <td>${p[otherKey]}</td>
  </tr>`;
}

// Podium arrays arrive ranked 1st-2nd-3rd; re-ordered to 2nd-1st-3rd below so
// the CSS's taller/raised rank-1 card lands in the visual middle, like a
// medal ceremony podium.
function renderAwards() {
  const a = DATA.awards;
  if (!a) return;

  document.getElementById("awards-sub").textContent = a.note;

  const gb = a.golden_boot_projected;
  document.getElementById("golden-boot-podium").innerHTML =
    [gb[1], gb[0], gb[2]].map((p, i) => podiumCard(p, "goals", i)).join("");
  document.getElementById("golden-boot-tbody").innerHTML =
    gb.map(p => awardRow(p, "goals", "assists")).join("");

  const ta = a.top_assists_projected;
  document.getElementById("top-assists-podium").innerHTML =
    [ta[1], ta[0], ta[2]].map((p, i) => podiumCard(p, "assists", i)).join("");
  document.getElementById("top-assists-tbody").innerHTML =
    ta.map(p => awardRow(p, "assists", "goals")).join("");

  const gg = a.golden_glove_projected;
  document.getElementById("golden-glove-podium").innerHTML =
    [gg[1], gg[0], gg[2]].map((p, i) => podiumCard(p, "clean_sheets", i)).join("");
  document.getElementById("golden-glove-tbody").innerHTML =
    gg.map(p => awardRow(p, "clean_sheets", "goals_conceded")).join("");
}

function renderFooter() {
  document.getElementById("footer-note").textContent =
    `WC2026 Knockout Predictor · generated ${DATA.generated_at} · predictions update as each round is played.`;
}

main();
