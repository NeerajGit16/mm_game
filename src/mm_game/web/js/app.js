// @ts-check
// The page: one websocket, one store, one render per animation frame.
// Each frame from the server is a JSON array of protocol messages.

import { createStore, apply } from "./store.js";
import { createControls, quotePrices } from "./keys.js";
import { createMarketChart, createSessionCharts } from "./chart.js";
import * as views from "./views.js";
import * as tabs from "./tabs.js";

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

function render() {
  views.render(store, controls);
  tabs.renderTabs(store);
  if (tabs.isActive("market")) market.update();
  if (tabs.isActive("pnl")) session.update();
}

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (event) => {
    for (const message of JSON.parse(event.data)) apply(store, message);
    schedule();
  };
  socket.onclose = () => views.disconnected();
}

document.addEventListener("keydown", (e) => {
  if (controls.onKey(e)) e.preventDefault();
});
views.bindButtons(controls, schedule);
tabs.bindTabs(schedule);
connect();

// For the console and the browser tests.
/** @type {any} */ (window).mm = { store, controls };
