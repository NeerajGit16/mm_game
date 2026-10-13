// @ts-check
// The end-of-session reveal, drawn from the coaching record alone -- in
// play from the reveal message, or later from a saved record with no game
// running. The hidden layer is shown here for the first time.
//
// Summary, in two columns: on the left, staying in view, the reveal chart,
// your position against the benchmarks and the clients revealed; on the
// right, cards at their natural height -- the score and capture ratio, the
// truth-based breakdown with its horizon switch, the three moments (each
// marks the charts when clicked), the unannounced moves with the lean
// before each, the bots' rules. The page scrolls as a whole. Tabs: Clients
// (full), Tickets (every RFQ, under a chart of your wins against the best
// other dealer and the screen's error), Track record. Every panel has an
// (i) with what its columns mean.

import { createRevealCharts, createWinsChart } from "./reveal_chart.js";
import { hideGlossary, showGlossary } from "./glossary.js";
import { escapeHtml, gap, longCategory, mmss, money, percent, price, shortCategory, signedBp2, signedInt, tone } from "./format.js";

/** @typedef {import("./protocol.js").CoachingRecord} CoachingRecord */
/** @typedef {import("./protocol.js").TrackMsg} TrackMsg */
/** @typedef {import("./protocol.js").ClientReveal} ClientReveal */
/** @typedef {import("./protocol.js").RfqLayer} RfqLayer */
/** @typedef {import("./protocol.js").BotRfq} BotRfq */
/** @typedef {import("./protocol.js").Pooled} Pooled */
/** @typedef {import("./protocol.js").PooledScore} PooledScore */

const HORIZON_LABEL = /** @type {Record<string, string>} */ ({ "60s": "60s", "300s": "300s", next_event: "next print/move" });
const BY_TYPE = [
  ["uninformed", "Adverse selection · uninformed"],
  ["informed_print", "Adverse selection · informed, prints"],
  ["informed_move", "Adverse selection · informed, moves"],
];
const SHOWN_CLIENTS = 5;
const TRACK_SCORES = /** @type {const} */ ([["pnl", "P&L"], ["risk_adjusted", "risk-adjusted"]]);
const TRACK_WHO = /** @type {const} */ ([["you_minus_disciplined", "You"], ["oracle_minus_disciplined", "Flow oracle"]]);

/** @param {string} id */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
/** @param {string} id */
const body = (id) => /** @type {HTMLElement} */ ($(id).querySelector("tbody"));
/** @param {number} n @param {string} word */
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
/**
 * A P&L contribution in a table cell, coloured by sign.
 * @param {number} x
 */
const cell = (x) => `<td class="r ${tone(Math.abs(x) < 0.5 ? 0 : x)}">${money(x)}</td>`;

/**
 * A pooled gap's mean per session, ± its standard error when there is one.
 * @param {Pooled} p
 */
const meanSe = (p) => p.mean_usd === null ? "n/a" : money(p.mean_usd) + (p.se_usd === null ? "" : ` ± ${money(p.se_usd, false)}`);

/**
 * The pooled capture ratio, or why it is not shown yet -- summary.md's words.
 * @param {PooledScore} s
 * @param {number} threshold
 */
function pooledRatio(s, threshold) {
  if (s.capture_ratio !== null) return percent(s.capture_ratio);
  return `n/a — not enough flow-reading edge yet (${money(s.oracle_minus_disciplined.total_usd, false)} of ${money(threshold, false)})`;
}

/** @param {ClientReveal} c */
export function knewAbout(c) {
  return c.informed_about
    .map((k) => `${k.kind} ${k.index + 1}: ${k.signal_direction > 0 ? "up" : "down"} (${k.right ? "right" : "wrong"})`)
    .join("; ") || "—";
}

// -- the summary --------------------------------------------------------------

/** @param {CoachingRecord} record */
function header(record) {
  const cfg = record.header.config;
  const minutes = cfg.session_length / 60;
  const prints = record.reveal.prints.length;
  const moves = record.reveal.moves.length;
  $("rv-meta").textContent = `${minutes} min · seed ${record.header.seed} · ${plural(prints, "print")} · ${plural(moves, "unannounced move")}`;
  $("rv-flag").hidden = !record.flags.replay_of_played_seed;
}

/**
 * @param {CoachingRecord} record
 * @param {TrackMsg | null} track
 */
function score(record, track) {
  const b = record.benchmarks;
  const rows = [
    ["You", record.scores],
    ["Naive", b.naive.scores],
    ["Disciplined bot", b.disciplined.scores],
    ["Flow oracle", b.oracle.scores],
  ];
  body("rv-score").innerHTML = rows.map(([name, s], i) => {
    const sc = /** @type {{pnl_usd: number, risk_adjusted_usd: number}} */ (s);
    return `<tr class="${i === 0 ? "current" : ""}"><td>${name}</td><td class="r">${money(sc.pnl_usd, false)}</td><td class="r">${money(sc.risk_adjusted_usd, false)}</td></tr>`;
  }).join("");
  const cr = record.capture_ratio;
  const you = record.scores.pnl_usd - b.disciplined.scores.pnl_usd;
  const oracle = cr.pnl.oracle_minus_disciplined_usd;
  let text;
  if (cr.pnl.ratio !== null) {
    text = `Flow-reading capture: <b>${percent(cr.pnl.ratio)}</b> of the oracle's edge this session (you ${money(you)} against the disciplined bot; the oracle ${money(oracle)}).`;
  } else if (oracle < 0) {
    text = `Flow-reading capture: <b>n/a</b> this session — the flow oracle finished ${money(-oracle, false)} behind the disciplined bot, so there was no flow-reading edge to measure (you ${money(you)} against the disciplined bot).`;
  } else {
    text = `Flow-reading capture: <b>n/a</b> this session — the oracle's edge over the disciplined bot, ${money(oracle, false)}, is under the ${money(cr.min_gap_usd, false)} it takes to measure (you ${money(you)} against the disciplined bot).`;
  }
  if (track) {
    const a = track.aggregates;
    const p = a.pnl;
    text += ` Track record, pooled over ${plural(a.sessions, "scored session")}: you ${money(p.you_minus_disciplined.total_usd)} against the disciplined bot`
      + ` (${meanSe(p.you_minus_disciplined)} a session), the flow oracle ${money(p.oracle_minus_disciplined.total_usd)}`
      + ` (${meanSe(p.oracle_minus_disciplined)}); pooled capture ratio <b id="rv-capture-pooled">${pooledRatio(p, a.pooled_min_gap_usd)}</b>`
      + (a.replays_excluded ? `; ${plural(a.replays_excluded, "practice replay")} not counted.` : ".");
  }
  $("rv-capture").innerHTML = text;
  /** @param {import("./protocol.js").RuleItem} r */
  const rule = (r) => {
    // A head that runs on into its sentence ("The flow oracle is ...") or stands alone ("Quotes.").
    const head = r.head ? `<b>${escapeHtml(r.head)}${/^[a-z]/.test(r.text) ? "" : "."}</b> ` : "";
    return `<li>${head}${escapeHtml(r.text)}</li>`;
  };
  $("rv-rules").innerHTML = `<li class="lbl">The disciplined bot</li>${record.reveal.rules.disciplined.map(rule).join("")}`
    + record.reveal.rules.oracle.map(rule).join("");
}

/**
 * @param {CoachingRecord} record
 * @param {string} horizon
 */
export function breakdown(record, horizon) {
  const d = record.decomposition;
  const h = d.by_horizon[horizon];
  const lines = [
    ["Spread captured", d.spread_captured_usd],
    ...BY_TYPE.map(([key, label]) => [label, h.adverse_selection_by_type_usd[key]]),
    ["Inventory beyond the mark-outs", h.inventory_beyond_usd],
    ["Hedging and closing", d.hedging_and_closing_usd],
  ];
  const rows = lines.map(([label, x]) => `<tr data-line="${label}"><td>${label}</td>${cell(/** @type {number} */ (x))}</tr>`);
  rows.push(`<tr class="total" data-line="Total"><td>Total</td>${cell(d.total_usd)}</tr>`);
  body("rv-breakdown").innerHTML = rows.join("");
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#rv-horizons [data-horizon]"))) {
    b.classList.toggle("active", b.dataset.horizon === horizon);
  }
}

/**
 * Informed RFQs and, in brackets, how many of them you filled: "4 (0)".
 * @param {ClientReveal} c
 */
const informedRfqs = (c) => `${c.informed_rfqs} (${c.informed_fills})`;

/**
 * The next-print/move mark-out of one kind of fill, or — when there were none.
 * @param {ClientReveal} c
 * @param {"informed" | "other"} kind
 */
function splitCell(c, kind) {
  const fills = kind === "informed" ? c.informed_fills : c.won - c.informed_fills;
  return fills ? cell(c.markouts_by_fill_usd[kind].next_event) : '<td class="r">—</td>';
}

/** @param {CoachingRecord} record */
function clientsTop(record) {
  const entries = Object.entries(record.diagnostics.clients);
  const traded = entries.filter(([, c]) => c.won > 0)
    .sort(([, a], [, b]) => Math.abs(b.markouts_usd.next_event) - Math.abs(a.markouts_usd.next_event));
  const top = traded.slice(0, SHOWN_CLIENTS);
  body("rv-clients-top").innerHTML = top.map(([id, c]) => `<tr data-client="${id}">
    <td>C${id} · ${shortCategory(c.category)}</td><td>${c.informed_type ? "informed" : "uninformed"}</td><td>${escapeHtml(knewAbout(c))}</td>
    <td class="r">${informedRfqs(c)}</td><td class="r">${c.lots}</td><td class="r">${signedInt(c.net_lots)}</td>
    ${cell(c.markouts_usd["300s"])}${splitCell(c, "informed")}${splitCell(c, "other")}</tr>`).join("");
  const rest = entries.length - top.length;
  $("rv-clients-more").textContent = rest > 0 ? `${rest} more in the Clients tab` : "all clients you traded with";
}

/**
 * Clicking a moment marks it on both charts, zooms to it and highlights
 * its card; clicking the same card again clears the mark and the zoom.
 * One moment at a time.
 * @param {CoachingRecord} record
 * @param {import("./reveal_chart.js").RevealCharts} charts
 */
function moments(record, charts) {
  const box = $("rv-moments");
  box.innerHTML = record.reveal.moments.map((m, i) => `<div class="moment" data-moment="${i}">
      <div class="moment-head"><span>${escapeHtml(m.title)}</span><span class="num ${tone(m.pnl_usd)}">${money(m.pnl_usd)}</span></div>
      <p>${escapeHtml(m.sentence)}</p></div>`).join("") || '<p class="muted">Nothing traded: no moments.</p>';
  for (const el of /** @type {NodeListOf<HTMLElement>} */ (box.querySelectorAll(".moment"))) {
    el.addEventListener("click", () => {
      const m = record.reveal.moments[Number(el.dataset.moment)];
      const on = !el.classList.contains("active");
      for (const other of box.querySelectorAll(".moment")) other.classList.toggle("active", on && other === el);
      if (on) {
        charts.mark(m);
        charts.jumpTo(m.from_step, m.to_step);
      } else {
        charts.mark(null);
        charts.reset();
      }
    });
  }
}

/**
 * A lean before a move: bp toward the move and, in brackets, the quotes it
 * covers; — with none.
 * @param {import("./protocol.js").Lean} l
 */
const leanCell = (l) => `<td class="r ${tone(l.mean_bp)}">${l.mean_bp === null ? "—" : `${signedBp2(l.mean_bp)}bp (${l.quotes})`}</td>`;

/**
 * Each unannounced move with your lean before it, and the flow oracle's and
 * the disciplined bot's beside it.
 * @param {CoachingRecord} record
 */
function moves(record) {
  const all = record.diagnostics.position.moves;
  const b = record.benchmarks;
  const dt = record.header.config.dt;
  body("rv-moves").innerHTML = all.map((m, i) => `<tr data-move="${m.move_index}">
    <td>${m.move_index + 1} · ${mmss(m.start_step * dt)}</td><td class="r">${signedBp2(m.size_bp)}bp</td><td class="r">${(m.lead_s / 60).toFixed(1)} min</td>
    ${leanCell(m.lean_before)}${leanCell(b.oracle.summary.lean_before_moves[i])}${leanCell(b.disciplined.summary.lean_before_moves[i])}</tr>`).join("");
  $("rv-moves").hidden = !all.length;
  $("rv-moves-none").hidden = all.length > 0;
}

// -- the tabs ------------------------------------------------------------------

/** @param {CoachingRecord} record */
function clientsTab(record) {
  const entries = Object.entries(record.diagnostics.clients).sort(([a], [b]) => Number(a) - Number(b));
  body("rv-clients-table").innerHTML = entries.map(([id, c]) => {
    const mk = c.markouts_usd;
    /** @param {string} k */
    const m = (k) => (c.won ? cell(mk[k]) : '<td class="r">—</td>');
    return `<tr data-client="${id}"><td>${id}</td><td>${c.category.replace("_", " ")}</td><td>${c.informed_type ? "informed" : "uninformed"}</td>
      <td class="r">${c.rfqs}</td><td class="r">${informedRfqs(c)}</td><td class="r">${c.quoted}</td><td class="r">${c.won}</td><td class="r">${c.lots}</td>
      <td class="r">${signedInt(c.net_lots)}</td>${m("60s")}${m("300s")}${m("next_event")}${splitCell(c, "informed")}${splitCell(c, "other")}
      <td>${escapeHtml(knewAbout(c))}</td></tr>`;
  }).join("");
}

/**
 * @param {CoachingRecord} record
 * @param {RfqLayer} r
 */
function whoItWas(record, r) {
  const h = r.hidden;
  if (h.informed && h.informed_about && h.signal_bp !== null) {
    const a = h.informed_about;
    const truth = a.kind === "print" ? record.reveal.prints[a.index].jump_bp : record.reveal.moves[a.index].size_bp;
    const right = (h.signal_bp > 0) === (truth > 0);
    return `<span class="sell">informed</span> · ${a.kind} ${a.index + 1}, ${h.signal_bp > 0 ? "up" : "down"} (${right ? "right" : "wrong"})`;
  }
  return "uninformed";
}

/** @param {BotRfq | undefined} o */
function oracleDid(o) {
  if (!o) return "—";
  if (o.outcome === "passed") return "passed";
  const quote = `${price(o.bid)} / ${price(o.ask)}`;
  if (o.outcome === "won") return `${quote} → traded ${signedInt(o.quantity)}`;
  if (o.outcome === "done_away") return `${quote} → done away`;
  return `${quote} → not done`;
}

/** @param {CoachingRecord} record */
function ticketsTab(record) {
  const cfg = record.header.config;
  const usd = cfg.tick_value / cfg.tick_size;
  const oracle = new Map(record.reveal.bots.oracle.rfqs.map((o) => [o.rfq_ordinal, o]));
  const clients = new Map(record.world.clients.map((c) => [c.client_id, c]));
  body("rv-tickets-table").innerHTML = record.rfqs.map((r) => {
    const rfq = r.shown ? /** @type {any} */ (r).observable.client : null;
    const client = rfq ? clients.get(rfq.client_id) : null;
    const side = r.hidden.side === 1 ? "buys" : "sells";
    const type = !rfq ? "—" : rfq.one_way ? `client ${side} (one-way)` : `two-way · client ${side}`;
    const a = r.action;
    const quote = !a ? "—" : a.bid === null && a.ask === null ? "passed" : `${price(a.bid)} / ${price(a.ask)}`;
    const o = r.outcome;
    const outcome = !o ? "not shown" : o.outcome === "won" ? `won · ${signedInt(o.quantity)} at ${price(o.price)}`
      : o.outcome === "done_away" ? `done away at ${price(o.cover)}` : o.outcome === "not_done" ? "not done" : "passed";
    const marks = r.hidden.markouts;
    /** @param {string} k */
    const mk = (k) => {
      const c = marks ? marks[k] : null;
      return !c || c.pnl_bp === null || c.truncated ? '<td class="r">—</td>' : cell(c.pnl_bp * usd);
    };
    return `<tr data-rfq="${r.ordinal}" class="${o && o.outcome === "won" ? "" : "muted"}"><td>${mmss(r.step * cfg.dt)}</td><td class="r">${r.ordinal}</td>
      <td>${client ? `C${client.client_id} · ${shortCategory(client.category)}` : "—"}</td><td>${whoItWas(record, r)}</td>
      <td class="r">${rfq ? rfq.size : "—"}</td><td>${type}</td><td class="r">${quote}</td><td>${outcome}</td>
      <td class="r">${gap(r.vs_best_dealer_bp)}</td><td class="r">${o ? signedInt(o.position_after) : "—"}</td>
      ${mk("60s")}${mk("300s")}${mk("next_event")}<td>${oracleDid(oracle.get(r.ordinal))}</td></tr>`;
  }).join("");
}

/**
 * @param {TrackMsg | null} track
 * @param {string | null} sessionId
 */
function trackTab(track, sessionId) {
  if (!track) {
    $("rv-track-pooled").textContent = "No track record found.";
    return;
  }
  const a = track.aggregates;
  $("rv-track-pooled").innerHTML = `Pooled over ${plural(a.sessions, "scored session")}; ${plural(a.replays_excluded, "practice replay")} shown, not counted. `
    + `Mean P&amp;L ${a.pnl_usd === null ? "n/a" : money(a.pnl_usd, false)}, risk-adjusted ${a.risk_adjusted_usd === null ? "n/a" : money(a.risk_adjusted_usd, false)}. `
    + `Pooled capture ratio: P&amp;L <b data-ratio="pnl">${pooledRatio(a.pnl, a.pooled_min_gap_usd)}</b>; `
    + `risk-adjusted <b data-ratio="risk_adjusted">${pooledRatio(a.risk_adjusted, a.pooled_min_gap_usd)}</b>.`;
  // The same rows as summary.md's track-record table.
  body("rv-track-gaps").innerHTML = TRACK_SCORES.flatMap(([key, label]) => TRACK_WHO.map(([who, name]) => {
    const p = a[key][who];
    const row = `${name}, ${label}`;
    return `<tr data-row="${escapeHtml(row)}"><td>${escapeHtml(row)}</td>${cell(p.total_usd)}`
      + `<td class="r">${p.mean_usd === null ? "n/a" : money(p.mean_usd)}</td><td class="r">${p.se_usd === null ? "n/a" : money(p.se_usd, false)}</td></tr>`;
  })).join("");
  body("rv-track-table").innerHTML = track.entries.map((e) => {
    const status = e.replay_of_played_seed ? "practice replay — not counted" : e.ended_early_at_step !== null ? "scored · ended early" : "scored";
    return `<tr data-session="${escapeHtml(e.session_id)}" class="${e.session_id === sessionId ? "current" : ""}${e.replay_of_played_seed ? " muted" : ""}">
      <td>${escapeHtml(e.session_id)}</td><td class="r">${e.preset ?? "—"}</td><td class="r">${e.seed}</td>
      <td class="r">${money(e.pnl_usd, false)}</td><td class="r">${money(e.disciplined_pnl_usd, false)}</td><td class="r">${money(e.oracle_pnl_usd, false)}</td>
      ${cell(e.you_minus_disciplined_pnl_usd)}${cell(e.oracle_minus_disciplined_pnl_usd)}<td class="r">${percent(e.capture_ratio_pnl)}</td>
      <td class="r">${money(e.risk_adjusted_usd, false)}</td>
      ${cell(e.you_minus_disciplined_risk_adjusted_usd)}${cell(e.oracle_minus_disciplined_risk_adjusted_usd)}<td class="r">${percent(e.capture_ratio_risk_adjusted)}</td>
      <td>${status}</td></tr>`;
  }).join("");
}

/**
 * Size the two charts so the left column -- charts and clients -- fits the
 * window below the tabs, the position chart half the reveal chart's height.
 */
function fitLeft() {
  const left = $("rv-left");
  if (!left.offsetHeight) return; // another tab is showing
  const chart = $("rv-chart");
  const pos = $("rv-pos");
  const top = left.getBoundingClientRect().top + window.scrollY;
  const room = window.innerHeight - top - 12;
  const fixed = left.offsetHeight - chart.offsetHeight - pos.offsetHeight;
  const h = Math.round(Math.max(220, Math.min(560, (room - fixed) / 1.5)));
  chart.style.height = `${h}px`;
  pos.style.height = `${Math.round(h / 2)}px`;
}

/** @param {string} name */
function showTab(name) {
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#rv-tabs [data-rtab]"))) {
    b.classList.toggle("active", b.dataset.rtab === name);
  }
  for (const t of ["summary", "clients", "tickets", "track"]) $(`rv-${t}`).hidden = t !== name;
  if (name === "summary") fitLeft();
}

/**
 * Hide the desk and show the reveal of ``record``.
 * @param {{record: CoachingRecord, track: TrackMsg | null, savedTo: string, sessionId: string | null}} args
 */
export function showReveal({ record, track, savedTo, sessionId }) {
  for (const id of ["app", "startscreen", "handbookscreen", "sizebox", "hint", "glossary"]) {
    const el = document.getElementById(id);
    if (el) el.hidden = true;
  }
  $("reveal").hidden = false;
  $("rv-saved").textContent = savedTo;
  header(record);
  score(record, track);
  breakdown(record, "next_event");
  clientsTop(record);
  clientsTab(record);
  ticketsTab(record);
  trackTab(track, sessionId);
  moves(record);
  fitLeft();
  const charts = createRevealCharts($("rv-chart"), $("rv-pos"), $("rv-tip"), record);
  const wins = createWinsChart($("rv-wins"), $("rv-tip"), record);
  const unplotted = record.rfqs.filter((r) => r.outcome && r.outcome.outcome === "won" && r.vs_best_dealer_bp === null).length;
  $("rv-wins-note").textContent = `${plural(wins.wins.length, "win")} with a cover`
    + (unplotted ? `; ${unplotted} with nobody else in, not drawn` : "");
  moments(record, charts);
  window.addEventListener("resize", fitLeft);
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#rv-horizons [data-horizon]"))) {
    b.addEventListener("click", () => breakdown(record, b.dataset.horizon ?? "next_event"));
  }
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#rv-tabs [data-rtab]"))) {
    b.addEventListener("click", () => showTab(b.dataset.rtab ?? "summary"));
  }
  // (i): what each panel's columns mean. No clock to pause here.
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#reveal [data-info]"))) {
    b.addEventListener("click", () => showGlossary(b.dataset.info ?? ""));
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideGlossary(); });
  const bots = /** @type {HTMLInputElement} */ ($("rv-bots"));
  bots.checked = false;
  bots.addEventListener("change", () => charts.showBots(bots.checked));
  // For the console and the browser tests.
  /** @type {any} */ (window).mmReveal = { record, track, charts, wins, horizons: HORIZON_LABEL, showTab };
  return charts;
}
