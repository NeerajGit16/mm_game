// @ts-check
// The left-hand tabs: Market (in views.js and chart.js), P&L, Clients, and
// Calendar & log. Tabs swap the left side only; the ticket, the roster, the
// top bar and the risk strip never leave the screen. Everything here is
// screen-based: the desk's numbers and the public events.

import { ago, bp, escapeHtml, longCategory, mmss, price, shortCategory, signedInt, tone, usd } from "./format.js";
import { $, clientSortValue, data, onClick, set, setClass, tbody } from "./views.js";

/** @typedef {import("./store.js").Store} Store */
/** @typedef {import("./store.js").LiveStore} LiveStore */
/** @typedef {import("./store.js").DeskStore} DeskStore */
/** @typedef {import("./protocol.js").PnlRow} PnlRow */
/** @typedef {import("./protocol.js").EventMsg} EventMsg */

export const TABS = ["market", "pnl", "clients", "log"];
let activeTab = "market";

/** @param {string} name */
export function showTab(name) {
  activeTab = name;
  for (const b of /** @type {NodeListOf<HTMLElement>} */ (document.querySelectorAll("#tabs .tab"))) {
    b.classList.toggle("active", data(b, "tab") === name);
  }
  for (const t of TABS) $(`tab-${t}`).hidden = t !== name;
}

export function cycleTab() {
  showTab(TABS[(TABS.indexOf(activeTab) + 1) % TABS.length]);
}

/**
 * Click-to-sort state for one table: first click sorts descending (or
 * ascending for names and times), the next click reverses.
 * @param {string} key
 * @param {number} dir
 */
function sorter(key, dir) {
  const s = { key, dir };
  return {
    s,
    /** @param {string} k */
    click(k) {
      s.dir = s.key === k ? -s.dir : k === "client_id" || k === "category" ? 1 : -1;
      s.key = k;
    },
  };
}

/**
 * @param {string} tableId
 * @param {string} key
 */
function markSorted(tableId, key) {
  for (const th of /** @type {NodeListOf<HTMLElement>} */ ($(tableId).querySelectorAll("th[data-sort]"))) {
    th.classList.toggle("sorted", th.dataset.sort === key);
  }
}

// -- P&L --------------------------------------------------------------------------

const pnlSort = sorter("step", -1);

/**
 * @param {PnlRow} r
 * @param {string} key
 * @returns {number}
 */
function pnlValue(r, key) {
  if (key === "ticket") return r.kind === "client" ? r.ordinal ?? 0 : r.kind === "hedge" ? 1e6 + r.seq : 2e6;
  if (key === "with") return r.client_id === null ? 1e6 : r.client_id;
  return /** @type {Record<string, any>} */ (r)[key];
}

/** @param {DeskStore} store */
function renderPnl(store) {
  const { rows, tiles } = store.desk.pnl;
  const clients = rows.filter((r) => r.kind === "client");
  const broker = rows.filter((r) => r.kind !== "client");
  /** @param {string} id @param {number} v */
  const tile = (id, v) => { set(id, usd(v)); setClass($(id), `num tile-v ${tone(v)}`); };
  tile("tile-edge", tiles.edge_client_usd);
  tile("tile-broker", tiles.venue_charge_usd);
  tile("tile-move", tiles.move_usd);
  tile("tile-total", tiles.total_usd);
  set("tile-edge-sub", `price vs screen at the time · ${clients.length} fill${clients.length === 1 ? "" : "s"}`);
  const brokerLots = broker.reduce((n, r) => n + Math.abs(r.quantity), 0);
  set("tile-broker-sub", `${broker.length} hedge${broker.length === 1 ? "" : "s"}, ${brokerLots} lots`);
  set("tile-total-sub", `realised ${usd(tiles.realised_usd)} · unrealised ${usd(tiles.unrealised_usd)}`);

  const { key, dir } = pnlSort.s;
  const sorted = [...rows].sort((a, b) => dir * (pnlValue(a, key) - pnlValue(b, key)) || b.seq - a.seq);
  tbody("pnl-table").innerHTML = sorted.map((r) => {
    const ticket = r.kind === "client" ? `RFQ ${r.ordinal}` : r.kind === "hedge" ? "Hedge" : "Close";
    const withWhom = r.kind === "client" ? `C${r.client_id} · ${shortCategory(r.category ?? "")}` : r.kind === "hedge" ? "broker" : "close";
    return `<tr data-seq="${r.seq}"><td>${mmss(r.step * store.dt)}</td><td>${ticket}</td><td>${withWhom}</td>
      <td>${r.quantity > 0 ? "bought" : "sold"} ${Math.abs(r.quantity)}</td>
      <td class="r">${price(r.price)}</td><td class="r">${price(r.screen_then)}</td>
      <td class="r ${tone(r.edge_usd)}">${usd(r.edge_usd)}</td><td class="r ${tone(r.move_usd)}">${usd(r.move_usd)}</td>
      <td class="r ${tone(r.total_usd)}">${usd(r.total_usd)}</td></tr>`;
  }).join("");
  /** @param {(r: PnlRow) => number} f */
  const sum = (f) => rows.reduce((s, r) => s + f(r), 0);
  set("pnl-foot-label", `All ${rows.length}`);
  for (const [id, v] of /** @type {[string, number][]} */ ([
    ["pnl-foot-edge", sum((r) => r.edge_usd)],
    ["pnl-foot-move", sum((r) => r.move_usd)],
    ["pnl-foot-total", sum((r) => r.total_usd)],
  ])) {
    set(id, usd(v));
    setClass($(id), `r ${tone(v)}`);
  }
  markSorted("pnl-table", key);
}

// -- Clients ------------------------------------------------------------------------

const clientSort = sorter("client_id", 1);

/** @param {DeskStore} store */
function renderClients(store) {
  const highlight = store.state.ticket ? store.state.ticket.client_id : null;
  const { key, dir } = clientSort.s;
  const clients = [...store.desk.clients].sort(
    (a, b) => dir * (clientSortValue(a, key) - clientSortValue(b, key)) || a.client_id - b.client_id,
  );
  tbody("clients-table").innerHTML = clients.map((c) => {
    const m = c.markout_bp_per_lot;
    const last = c.last_trade_step === null ? "—" : ago((store.state.step - c.last_trade_step) * store.dt);
    return `<tr data-client="${c.client_id}" class="${c.client_id === highlight ? "highlight" : ""}">
      <td>C${c.client_id}</td><td>${longCategory(c.category)}</td><td class="r">${c.rfqs}</td><td class="r">${c.quoted}</td>
      <td class="r">${c.traded}</td><td class="r">${c.hit_rate === null ? "—" : `${Math.round(c.hit_rate * 100)}%`}</td>
      <td class="r">${c.lots}</td><td class="r">${signedInt(c.net)}</td><td class="r">${last}</td>
      <td class="r ${tone(m["60s"])}">${bp(m["60s"])}</td><td class="r">${c.markout_lots["60s"]}</td>
      <td class="r ${tone(m["300s"])}">${bp(m["300s"])}</td><td class="r">${c.markout_lots["300s"]}</td></tr>`;
  }).join("");
  markSorted("clients-table", key);
}

// -- Calendar & log -------------------------------------------------------------------

export const EVENT_KINDS = [
  "session_start", "screen_update", "print_released", "rfq_arrived", "quote_sent", "limit_blocked",
  "rfq_resolved", "hedge_executed", "speed_changed", "paused", "resumed", "timer_expired",
  "session_quit", "close_liquidation", "session_end",
];
/** @type {Set<string>} */
const shownKinds = new Set(EVENT_KINDS);
let logRendered = 0;
let logFilterKey = "";

/**
 * One line saying what an event was, from its public payload.
 * @param {Store} store
 * @param {EventMsg} e
 */
export function describe(store, e) {
  const d = e.data;
  /** @param {number} q */
  const did = (q) => `${q > 0 ? "bought" : "sold"} ${Math.abs(q)}`;
  switch (e.kind) {
    case "session_start":
      return `${d.config.session_length / 60}-minute session · ${d.print_steps.length} print${d.print_steps.length === 1 ? "" : "s"} on the calendar · ${d.roster.length} clients`;
    case "screen_update":
      return `screen ${price(d.mid)}`;
    case "print_released":
      return `print #${d.print_index + 1} released`;
    case "rfq_arrived": {
      const r = d.rfq;
      const type = r.one_way ? (r.side === 1 ? "client buys" : "client sells") : "two-way";
      return `RFQ ${r.ordinal} · C${r.client_id} ${shortCategory(r.category)} · ${r.size} lots · ${type} · in comp ${r.in_comp}`;
    }
    case "quote_sent":
      return d.bid === null && d.ask === null
        ? `RFQ ${d.rfq_ordinal} passed${d.by === "player" ? "" : ` (${d.by})`}`
        : `RFQ ${d.rfq_ordinal} quoted ${price(d.bid)} / ${price(d.ask)}`;
    case "limit_blocked":
      return `RFQ ${d.rfq_ordinal}: ${d.sides.join(" and ")} blocked by the limit at ${signedInt(d.position)}`;
    case "rfq_resolved": {
      const what = d.outcome === "won"
        ? `WON · you ${did(d.quantity)} at ${price(d.price)}${d.cover === null ? " · no cover" : ` · cover ${price(d.cover)}`}`
        : d.outcome === "done_away" ? `DONE AWAY · client ${d.side === 1 ? "bought" : "sold"} at ${price(d.cover)}`
        : d.outcome === "not_done" ? "NOT DONE" : "PASSED";
      return `RFQ ${d.rfq_ordinal} ${what} · position ${signedInt(d.position_after)}`;
    }
    case "hedge_executed":
      return d.quantity === 0
        ? `hedge refused at the limit (asked ${signedInt(d.requested)})`
        : `hedge ${did(d.quantity)} at ${price(d.price)} (${d.cost_bp_per_lot.toFixed(2)}bp/lot)${d.clipped ? ` · clipped from ${signedInt(d.requested)}` : ""} · position ${signedInt(d.position_after)}`;
    case "speed_changed":
      return `speed ${d.speed}×`;
    case "paused":
      return "paused";
    case "resumed":
      return "resumed";
    case "timer_expired":
      return `RFQ ${d.rfq_ordinal}: pricing timer ran out`;
    case "session_quit":
      return `quit with position ${signedInt(d.position)}: it rides to the close`;
    case "close_liquidation":
      return d.quantity === 0 ? "close: flat, nothing to liquidate" : `close: ${did(d.quantity)} at ${price(d.price)} (${d.cost_bp_per_lot.toFixed(2)}bp/lot)`;
    case "session_end":
      return `session end · P&L ${usd(d.pnl_usd)} · penalty ${usd(-d.risk_penalty_usd)} · risk-adjusted ${usd(d.risk_adjusted_usd)}`;
    default:
      return escapeHtml(JSON.stringify(d));
  }
}

/** @param {LiveStore} store */
function renderCalendar(store) {
  const now = store.state.time_s;
  const next = store.printSteps.findIndex((s) => s * store.dt > now);
  const rows = store.printSteps.map((step, i) => {
    const t = step * store.dt;
    const status = t <= now ? "done" : i === next ? `next · in ${mmss(t - now)}` : "later";
    return `<tr><td>#${i + 1}</td><td>${mmss(t)}</td><td>${status}</td></tr>`;
  });
  const html = rows.join("") || '<tr><td colspan="3" class="muted">No prints in this session.</td></tr>';
  const body = tbody("calendar-table");
  if (body.innerHTML !== html) body.innerHTML = html;
}

/** @param {LiveStore} store */
function renderFilters(store) {
  /** @type {Record<string, number>} */
  const counts = {};
  for (const e of store.events) counts[e.kind] = (counts[e.kind] || 0) + 1;
  const el = $("log-filters");
  if (!el.childElementCount) {
    el.innerHTML = EVENT_KINDS.map(
      (k) => `<label data-kind="${k}"><input type="checkbox" checked data-kind="${k}">${k.replace("_", " ")} <span class="muted"></span></label>`,
    ).join("");
    for (const box of /** @type {NodeListOf<HTMLInputElement>} */ (el.querySelectorAll("input"))) {
      box.addEventListener("change", () => {
        if (box.checked) shownKinds.add(data(box, "kind"));
        else shownKinds.delete(data(box, "kind"));
        logFilterKey = "";
        renderLog(store);
      });
    }
  }
  for (const label of /** @type {NodeListOf<HTMLElement>} */ (el.querySelectorAll("label"))) {
    const span = /** @type {HTMLElement} */ (label.querySelector("span"));
    const text = String(counts[data(label, "kind")] || 0);
    if (span.textContent !== text) span.textContent = text;
  }
}

/** @param {LiveStore} store */
function renderLog(store) {
  renderCalendar(store);
  renderFilters(store);
  const key = [...shownKinds].sort().join(",");
  const body = tbody("log-table");
  if (key !== logFilterKey) {
    body.innerHTML = "";
    logRendered = 0;
    logFilterKey = key;
  }
  if (logRendered >= store.events.length) return;
  const scroller = $("log-scroll");
  const atBottom = scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 30;
  const fresh = store.events.slice(logRendered).filter((e) => shownKinds.has(e.kind));
  logRendered = store.events.length;
  body.insertAdjacentHTML(
    "beforeend",
    fresh.map((e) => `<tr data-kind="${e.kind}"><td>${mmss(e.step * store.dt)}</td><td class="r">${e.seq}</td>
      <td>${e.kind.replace("_", " ")}</td><td>${describe(store, e)}</td></tr>`).join(""),
  );
  if (atBottom) scroller.scrollTop = scroller.scrollHeight;
}

// -- wiring -------------------------------------------------------------------------

/** @param {() => void} rerender */
export function bindTabs(rerender) {
  onClick("#tabs .tab", (el) => { showTab(data(el, "tab")); rerender(); });
  onClick("#pnl-table th[data-sort]", (el) => { pnlSort.click(data(el, "sort")); invalidate(); rerender(); });
  onClick("#clients-table th[data-sort]", (el) => { clientSort.click(data(el, "sort")); invalidate(); rerender(); });
}

/** @type {object | null} */
let lastDesk = null;
let lastTab = "";
/** @param {Store} store */
export function renderTabs(store) {
  if (!store.state) return;
  if (activeTab === "log") renderLog(/** @type {LiveStore} */ (store));
  if (!store.desk) return;
  const ready = /** @type {DeskStore} */ (store);
  const changed = store.desk !== lastDesk || activeTab !== lastTab;
  if (activeTab === "pnl" && changed) renderPnl(ready);
  if (activeTab === "clients" && changed) renderClients(ready);
  lastDesk = store.desk;
  lastTab = activeTab;
}

/** Force the next render to redraw the active tab (after a sort click). */
export function invalidate() {
  lastDesk = null;
}

export const isActive = (/** @type {string} */ name) => activeTab === name;
