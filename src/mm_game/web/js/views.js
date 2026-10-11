// @ts-check
// Rendering the fixed frame: top bar, ticket, compact roster, blotter,
// risk strip and the overlays. Every number on screen comes from the
// store: the engine's snapshot, the desk, the handbook and the public
// events. Nothing is computed here that the desk or the engine has not
// already put on the wire, except the quote being built and where your
// position lands if it is hit.

import { ago, bp, clock, escapeHtml, longCategory, mmss, price, shortCategory, signedInt, tone, usd } from "./format.js";
import { quotePrices } from "./keys.js";
import { renderMarkdown } from "./markdown.js";

/** @typedef {import("./store.js").Store} Store */
/** @typedef {import("./store.js").LiveStore} LiveStore */
/** @typedef {import("./store.js").DeskStore} DeskStore */
/** @typedef {import("./keys.js").Controls} Controls */
/** @typedef {import("./protocol.js").BlotterRow} BlotterRow */
/** @typedef {import("./protocol.js").Interaction} Interaction */
/** @typedef {import("./protocol.js").HandbookMsg} HandbookMsg */

/** @param {string} id */
export const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/**
 * @param {string} id
 * @param {string} text
 */
export const set = (id, text) => {
  const el = $(id);
  if (el.textContent !== text) el.textContent = text;
};

/**
 * The body of a table, by the table's id.
 * @param {string} id
 */
export const tbody = (id) => /** @type {HTMLElement} */ ($(id).querySelector("tbody"));

/**
 * A data attribute that the markup always sets.
 * @param {HTMLElement} el
 * @param {string} name
 */
export const data = (el, name) => el.dataset[name] ?? "";

/**
 * @param {HTMLElement} el
 * @param {string} cls
 */
export const setClass = (el, cls) => {
  if (el.className !== cls) el.className = cls;
};

/**
 * @param {string} id
 * @param {string} html
 */
export const setHtml = (id, html) => {
  const el = $(id);
  if (el.innerHTML !== html) el.innerHTML = html;
};

let rosterSort = { key: "client_id", dir: 1 };
/** @type {object | null} */
let lastDesk = null;
/** @type {string | null} */
let lastTicketKey = null;
/** @type {number | null} */
let lastRosterHighlight = null;

// -- top bar --------------------------------------------------------------------

/** @param {LiveStore} store */
function renderTopbar(store) {
  const st = store.state;
  set("clock", clock(st.time_s));
  set("clock-total", clock(store.nSteps * store.dt));
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#speeds [data-speed]"))) {
    b.classList.toggle("active", Number(b.dataset.speed) === st.clock.speed);
  }
  set("pause-btn", st.clock.paused ? "▶ Resume" : "❚❚ Pause");
  const next = store.printSteps.findIndex((s) => s * store.dt > st.time_s);
  const chips = store.printSteps.map((step, i) => {
    const t = step * store.dt;
    if (t <= st.time_s) return `<span class="chip done num">#${i + 1} ${mmss(t)} · done</span>`;
    const isNext = i === next;
    return `<span class="chip num${isNext ? " next" : ""}">#${i + 1} ${mmss(t)}${isNext ? ` · in ${mmss(t - st.time_s)}` : ""}</span>`;
  });
  setHtml("prints", chips.join("") || '<span class="muted">none</span>');

  const b = st.book;
  set("tb-position", signedInt(b.position));
  set("tb-pnl", usd(b.total_usd));
  setClass($("tb-pnl"), `num big ${tone(b.total_usd)}`);
  set("tb-unreal", `(unreal. ${usd(b.unrealised_usd)})`);
  const penalty = -st.risk.penalty_so_far_usd;
  set("tb-penalty", usd(penalty));
  setClass($("tb-penalty"), `num ${tone(penalty)}`);
  const adj = b.total_usd - st.risk.penalty_so_far_usd;
  set("tb-riskadj", usd(adj));
  setClass($("tb-riskadj"), `num big ${tone(adj)}`);
  $("tb-flag").hidden = !store.replayOfPlayedSeed;
}

// -- ticket ---------------------------------------------------------------------

/**
 * @param {Store} store
 * @param {number} id
 */
function clientDesk(store, id) {
  return store.desk ? store.desk.clients.find((c) => c.client_id === id) : null;
}

/**
 * @param {LiveStore} store
 * @param {Interaction} r
 */
function recentLine(store, r) {
  const t = mmss(r.step * store.dt);
  let what;
  if (r.outcome === "won") {
    const m = r.markout_bp_per_lot;
    const mark = m["300s"] !== null ? `300s ${bp(m["300s"])}/lot` : m["60s"] !== null ? `60s ${bp(m["60s"])}/lot` : "marking out";
    what = `you ${r.quantity > 0 ? "bought" : "sold"} · ${mark}`;
  } else if (r.outcome === "done_away") {
    what = `done away · client ${r.client_side === 1 ? "bought" : "sold"} at ${price(r.level)}`;
  } else if (r.outcome === "not_done") {
    what = "not done";
  } else {
    what = "you passed";
  }
  return `<div><span class="muted">${t}</span>  ${String(r.size).padStart(3, " ")}  ${escapeHtml(what)}</div>`;
}

/**
 * @param {LiveStore} store
 * @param {import("./protocol.js").Ticket} ticket
 */
function renderHistory(store, ticket) {
  const c = clientDesk(store, ticket.client_id);
  if (!c) return "";
  const m = c.markout_bp_per_lot;
  const last = c.last_trade_step === null ? "—" : ago((store.state.step - c.last_trade_step) * store.dt);
  const lines = c.recent.map((r) => recentLine(store, r)).join("") || '<div class="muted">no history yet</div>';
  return `
    <div class="lbl" style="margin-bottom:6px">Your history with client ${c.client_id} (marked against the screen)</div>
    <div class="hist-grid">
      <div><div class="lbl">RFQs</div><div class="v">${c.rfqs}</div></div>
      <div><div class="lbl">Traded</div><div class="v">${c.traded}</div></div>
      <div><div class="lbl">Your net</div><div class="v">${signedInt(c.net)}</div></div>
      <div><div class="lbl">Last trade</div><div class="v">${last}</div></div>
      <div style="grid-column: span 2"><div class="lbl">60s mark-out / lot</div><div class="v ${tone(m["60s"])}">${bp(m["60s"])}</div></div>
      <div style="grid-column: span 2"><div class="lbl">300s mark-out / lot</div><div class="v ${tone(m["300s"])}">${bp(m["300s"])}</div></div>
    </div>
    <div class="hist-lines">${lines}</div>`;
}

/**
 * @param {LiveStore} store
 * @param {Controls} controls
 */
function renderTicket(store, controls) {
  const st = store.state;
  const t = st.ticket;
  const k = controls.state;
  $("ticket").classList.toggle("live", !!t);
  $("ticket-empty").hidden = !!t;
  $("ticket-body").hidden = !t;
  set("ticket-title", t ? `RFQ ticket · #${t.ordinal}` : "RFQ ticket");
  if (t) {
    set("t-client", `Client ${t.client_id}`);
    set("t-category", `· ${longCategory(t.category)}`);
    set("t-size", `${t.size} lot${t.size === 1 ? "" : "s"}`);
    set("t-type", t.one_way ? (t.side === 1 ? "client buys" : "client sells") : "two-way");
    set("t-comp", t.in_comp === 0 ? "in comp: none" : `in comp: ${t.in_comp} dealer${t.in_comp > 1 ? "s" : ""}`);
    const key = `${t.ordinal}:${store.desk ? store.desk.step : 0}`;
    if (lastTicketKey !== key) {
      $("t-history").innerHTML = renderHistory(store, t);
      lastTicketKey = key;
    }
  }
  const mid = st.screen.mid;
  set("q-screen", price(mid));
  set("q-skew", `${k.skew > 0 ? "+" : k.skew < 0 ? "−" : ""}${Math.abs(k.skew)} tick${Math.abs(k.skew) === 1 ? "" : "s"}`);
  set("q-width", `±${k.halfWidth} tick${k.halfWidth === 1 ? "" : "s"}`);
  const two = quotePrices(mid, k.skew, k.halfWidth, store.tick, null);
  const sent = quotePrices(mid, k.skew, k.halfWidth, store.tick, t);
  const pos = st.book.position;
  const blocked = t ? t.blocked_sides : [];
  /** @type {["bid" | "ask", string, number | null, number | null][]} */
  const sides = [
    ["bid", "q-bid", two.bid, t ? pos + t.size : null],
    ["ask", "q-ask", two.ask, t ? pos - t.size : null],
  ];
  for (const [side, id, p, landing] of sides) {
    set(id, price(p));
    const off = !!t && sent[side] === null;
    const isBlocked = blocked.includes(side);
    set(`${id}-pos`, landing === null ? "" : off ? "not asked" : isBlocked ? "blocked: past the limit" : `position → ${signedInt(landing)}`);
    setClass($(`${id}-box`), `side ${side === "bid" ? "buy" : "sell"}${off ? " off" : ""}${isBlocked && !off ? " blocked" : ""}`);
  }
  const block = store.lastBlock;
  const note = block && t && block.rfq_ordinal === t.ordinal ? `Limit blocked your ${block.sides.join(" and ")}.` : "";
  set("q-notice", note);
}

// -- blotter --------------------------------------------------------------------

/** @param {BlotterRow} r */
function outcomeText(r) {
  if (r.outcome === null) return "on the ticket";
  if (r.outcome === "won") return `WON · you ${r.quantity > 0 ? "bought" : "sold"}`;
  return { done_away: "DONE AWAY", not_done: "NOT DONE", passed: "PASSED" }[r.outcome];
}

/** @param {BlotterRow} r */
function learnedText(r) {
  const l = r.learned;
  if (r.outcome === null) return "";
  if (!l) return "nothing";
  if (r.outcome === "won") return l.cover === null || l.cover === undefined ? "no cover: nobody else in" : `cover ${price(l.cover)}`;
  return `client ${l.side === 1 ? "bought" : "sold"} at ${price(l.level)}`;
}

/** @param {DeskStore} store */
function renderBlotter(store) {
  const rows = store.desk.blotter.slice(0, 300).map((r) => {
    const type = r.one_way ? (r.side === 1 ? "client buys" : "client sells") : "two-way";
    const quoted = r.outcome === "passed" || (r.bid === null && r.ask === null) ? "—" : `${price(r.bid)} / ${price(r.ask)}`;
    return `<tr class="${r.outcome === null ? "pending" : ""}">
      <td>${mmss(r.step * store.dt)}</td><td class="r">${r.ordinal}</td><td>C${r.client_id} · ${shortCategory(r.category)}</td>
      <td class="r">${r.size}</td><td>${type}</td><td class="r">${r.outcome === null ? "" : quoted}</td>
      <td>${outcomeText(r)}</td><td>${learnedText(r)}</td>
      <td class="r">${r.position_after === null ? "" : signedInt(r.position_after)}</td></tr>`;
  });
  tbody("blotter").innerHTML = rows.join("");
}

// -- compact roster ---------------------------------------------------------------

/**
 * @param {import("./protocol.js").ClientDesk} c
 * @param {string} key
 * @returns {number}
 */
export function clientSortValue(c, key) {
  if (key === "60s" || key === "300s") return c.markout_bp_per_lot[key] ?? -Infinity;
  if (key === "lots60") return c.markout_lots["60s"];
  if (key === "lots300") return c.markout_lots["300s"];
  if (key === "category") return ["hedge_fund", "real_money", "corporate"].indexOf(c.category);
  const v = /** @type {Record<string, any>} */ (c)[key];
  return v === null || v === undefined ? -Infinity : v;
}

/** @param {DeskStore} store */
function renderRoster(store) {
  const highlight = store.state.ticket ? store.state.ticket.client_id : null;
  const clients = [...store.desk.clients].sort(
    (a, b) => rosterSort.dir * (clientSortValue(a, rosterSort.key) - clientSortValue(b, rosterSort.key)) || a.client_id - b.client_id,
  );
  tbody("roster-table").innerHTML = clients.map((c) => {
    const m = c.markout_bp_per_lot;
    return `<tr class="${c.client_id === highlight ? "highlight" : ""}">
      <td>C${c.client_id} · ${shortCategory(c.category)}</td><td class="r">${c.rfqs}</td><td class="r">${c.traded}</td>
      <td class="r">${signedInt(c.net)}</td>
      <td class="r ${tone(m["60s"])}">${bp(m["60s"]).replace("bp", "")}</td>
      <td class="r ${tone(m["300s"])}">${bp(m["300s"]).replace("bp", "")}</td></tr>`;
  }).join("");
  for (const th of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#roster-table th"))) {
    th.classList.toggle("sorted", th.dataset.sort === rosterSort.key);
  }
  lastRosterHighlight = highlight;
}

// -- risk strip -----------------------------------------------------------------

/** @param {LiveStore} store */
function renderRisk(store) {
  const st = store.state;
  const pos = st.book.position;
  const limit = st.book.limit;
  set("rs-pos", `${signedInt(pos)} lots`);
  set("lim-lo", `−${limit}`);
  set("lim-hi", `+${limit}`);
  const fill = $("pos-fill");
  const frac = Math.min(Math.abs(pos) / limit, 1) * 50;
  fill.style.left = pos >= 0 ? "50%" : `${50 - frac}%`;
  fill.style.width = `${frac}%`;
  setClass(fill, `bar-fill${pos < 0 ? " short" : ""}${Math.abs(pos) >= 0.8 * limit ? " near" : ""}`);

  const np = st.risk.next_print;
  if (np) {
    set("pr-title", `Print risk · #${np.print_index + 1} in ${mmss(np.in_s)}`);
    set("pr-text", `carry ${signedInt(pos)} → ≈ ${usd(np.penalty_if_held_usd, false)} penalty`);
  } else {
    set("pr-title", "Print risk");
    set("pr-text", "no more prints");
  }
  $("pr-marker").style.left = `calc(${Math.min(Math.abs(pos) / 100, 1) * 100}% - 1px)`;

  const costs = st.venue.quotes.map((q) => `${q.size}: ${q.cost_bp_per_lot.toFixed(2)}`).join(" · ");
  set("venue-cost", `now ${costs} bp/lot${st.venue.near_print ? " · wider: near a print" : " · wider near prints"}`);

  const h = store.lastHedge;
  if (h) {
    let text;
    if (h.quantity === 0) text = `Hedge refused: at the ±${limit} limit (asked ${signedInt(h.requested)}).`;
    else {
      const did = `${h.quantity > 0 ? "Bought" : "Sold"} ${Math.abs(h.quantity)} at ${price(h.price)} (${h.cost_bp_per_lot.toFixed(2)}bp/lot), position ${signedInt(h.position_after)}`;
      text = h.clipped ? `Hedge clipped by the ±${limit} limit: asked ${signedInt(h.requested)}, filled ${signedInt(h.quantity)}. ${did}.` : `${did}.`;
    }
    set("hedge-notice", `${mmss(h.step * store.dt)} · ${text}`);
    setClass($("hedge-notice"), `notice${h.clipped ? " warn" : ""}`);
  }
}

// -- the handbook ---------------------------------------------------------------

/** @param {HandbookMsg} handbook */
export function handbookHtml(handbook) {
  return handbook.parts.map((part) => {
    const items = part.items.map((i) => {
      if (!i.head) return `<li>${escapeHtml(i.text)}</li>`;
      // A head that runs on into its sentence ("Clients send RFQs") or stands alone ("The product.").
      const runsOn = /^[a-z±]/.test(i.text);
      return `<li><strong>${escapeHtml(i.head)}${runsOn ? "" : "."}</strong> ${escapeHtml(i.text)}</li>`;
    }).join("");
    const note = part.note ? `<p class="locked">${escapeHtml(part.note)}</p>` : "";
    return `<h2>${escapeHtml(part.title)}<span class="muted">${escapeHtml(part.subtitle)}</span></h2>${note}${items ? `<ul>${items}</ul>` : ""}`;
  }).join("");
}

/** @param {boolean} open */
export function setHandbook(open) {
  $("handbookscreen").hidden = !open;
}

// -- overlays -------------------------------------------------------------------

/** @type {number | undefined} */
let hintTimer;
/**
 * @param {string | null} text
 * @param {number} [ms]
 */
export function setHint(text, ms = 1500) {
  const el = $("hint");
  clearTimeout(hintTimer);
  if (!text) { el.hidden = true; return; }
  el.textContent = text;
  el.hidden = false;
  hintTimer = window.setTimeout(() => { el.hidden = true; }, ms);
}

/** @param {{text: string} | null} box */
export function setBox(box) {
  $("sizebox").hidden = !box;
  if (box) set("sizebox-value", box.text || "_");
}

let shownErrors = 0;
/** @param {Store} store */
function renderToasts(store) {
  while (shownErrors < store.errors.length) {
    const e = store.errors[shownErrors++];
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = e.detail;
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }
}

/** @type {HandbookMsg | null} */
let renderedHandbook = null;
/** @param {Store} store */
function renderScreens(store) {
  set("start-title", store.preset ? `${store.preset}-minute session` : "Session");
  $("start-flag").hidden = !store.replayOfPlayedSeed;
  $("startscreen").hidden = store.begun || !!store.summary;
  if (store.handbook && store.handbook !== renderedHandbook) {
    const html = handbookHtml(store.handbook);
    $("start-handbook").innerHTML = html;
    $("handbook-body").innerHTML = html;
    renderedHandbook = store.handbook;
  }
  if (store.summary && $("endscreen").hidden) {
    $("end-summary").innerHTML = renderMarkdown(store.summary.markdown);
    set("end-saved", store.summary.saved_to);
    $("endscreen").hidden = false;
    $("sizebox").hidden = true;
    $("handbookscreen").hidden = true;
  }
}

export function disconnected() {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = "Disconnected from the server. Reload the page once it is running again.";
  $("toasts").appendChild(el);
}

// -- wiring ---------------------------------------------------------------------

/**
 * @param {string} selector
 * @param {(el: HTMLElement) => void} fn
 */
export function onClick(selector, fn) {
  for (const el of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll(selector))) {
    el.addEventListener("click", () => { fn(el); el.blur(); });
  }
}

/**
 * @param {Controls} controls
 * @param {() => void} rerender
 */
export function bindButtons(controls, rerender) {
  const a = controls.actions;
  onClick("#speeds [data-speed]", (el) => a.speed(Number(el.dataset.speed)));
  onClick("#pause-btn", () => a.togglePause());
  onClick("#start-btn", () => a.togglePause());
  onClick("#handbook-btn", () => a.toggleHandbook());
  onClick("#quit-btn", () => { if (confirm("Stop trading? Your position rides to the close.")) a.quit(); });
  onClick("[data-hedge]", (el) => a.hedge(Number(el.dataset.hedge)));
  onClick("[data-digit]", (el) => a.boxDigit(data(el, "digit")));
  /** @type {Record<string, () => void>} */
  const acts = {
    "skew-down": () => a.skew(-1), "skew-up": () => a.skew(1), wider: () => a.width(1), tighter: () => a.width(-1),
    reset: () => a.resetSkew(), send: () => a.send(), pass: () => a.pass(), sizebox: () => a.openBox(),
    flatten: () => a.flattenPress(), "box-back": () => a.boxBack(), "box-cancel": () => a.boxCancel(),
    "box-buy": () => a.boxTrade(1), "box-sell": () => a.boxTrade(-1),
  };
  onClick("[data-act]", (el) => acts[data(el, "act")]());
  onClick("#roster-table th[data-sort]", (el) => {
    const key = data(el, "sort");
    rosterSort = { key, dir: rosterSort.key === key ? -rosterSort.dir : key === "client_id" ? 1 : -1 };
    lastDesk = null;
    rerender();
  });
}

/**
 * @param {Store} store
 * @param {Controls} controls
 */
export function render(store, controls) {
  renderScreens(store);
  renderToasts(store);
  if (!store.state) return;
  const live = /** @type {LiveStore} */ (store);
  renderTopbar(live);
  renderTicket(live, controls);
  renderRisk(live);
  const highlight = live.state.ticket ? live.state.ticket.client_id : null;
  if (store.desk && (store.desk !== lastDesk || highlight !== lastRosterHighlight)) {
    renderBlotter(/** @type {DeskStore} */ (store));
    renderRoster(/** @type {DeskStore} */ (store));
    lastDesk = store.desk;
  }
}
