// @ts-check
// The page: one websocket, one store, one render per animation frame.
// Each frame from the server is a JSON array of protocol messages.

import { createStore, apply } from "./store.js";
import { createControls, quotePrices } from "./keys.js";
import { createMarketChart, createSessionCharts } from "./chart.js";
import * as views from "./views.js";
import * as tabs from "./tabs.js";
import { showReveal } from "./reveal.js";

const store = createStore();
/** @type {WebSocket | null} */
let socket = null;

/** @param {import("./protocol.js").Action} message */
function send(message) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

let scheduled = false;
function schedule() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    render();
  });
}

const controls = createControls({
  store,
  send,
  ui: {
    changed: schedule,
    hint: views.setHint,
    box: views.setBox,
    handbook: views.setHandbook,
    cycleTab: () => { tabs.cycleTab(); schedule(); },
  },
});

function currentQuote() {
  const st = store.state;
  if (!st || !st.ticket) return null;
  return quotePrices(st.screen.mid, controls.state.skew, controls.state.halfWidth, store.tick, st.ticket);
}

const market = createMarketChart(views.$("chart"), store, currentQuote);
const session = createSessionCharts(views.$("pnl-chart"), views.$("pos-chart"), store);

let revealed = false;
function render() {
  if (revealed) return;
  // The session is over once the coaching record, its summary and the
  // track record have arrived: the reveal takes the page.
  if (store.record && store.summary && store.track) {
    revealed = true;
    showReveal({ record: store.record, track: store.track, savedTo: store.summary.saved_to, sessionId: store.summary.session_id });
    return;
  }
  views.render(store, controls);
  tabs.renderTabs(store);
  if (tabs.isActive("market")) market.update();
  if (tabs.isActive("pnl")) session.update();
}

/**
 * A saved session's reveal, with no game running: the record and the
 * track record come from the server's records, not a websocket.
 * @param {string} sessionId
 */
async function openSaved(sessionId) {
  revealed = true;
  const record = await fetch(`/record/${encodeURIComponent(sessionId)}`).then((r) => {
    if (!r.ok) throw new Error(`no saved session ${sessionId}`);
    return r.json();
  });
  const track = await fetch("/track").then((r) => (r.ok ? r.json() : null));
  showReveal({ record, track, savedTo: `sessions/${sessionId}/`, sessionId });
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (event) => {
    for (const message of JSON.parse(event.data)) apply(store, message);
    schedule();
  };
  socket.onclose = () => views.disconnected();
}

const saved = new URLSearchParams(location.search).get("reveal");
if (saved) {
  openSaved(saved).catch((error) => {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = String(error.message);
    views.$("toasts").appendChild(toast);
  });
} else {
  document.addEventListener("keydown", (e) => {
    if (!revealed && controls.onKey(e)) e.preventDefault();
  });
  views.bindButtons(controls, schedule);
  tabs.bindTabs(schedule);
  connect();
}

// For the console and the browser tests.
/** @type {any} */ (window).mm = { store, controls };
