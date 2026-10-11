// The page: one websocket, one store, one render per animation frame.
// Each frame from the server is a JSON array of protocol messages.

import { createStore, apply } from "./store.js";
import { createControls, quotePrices } from "./keys.js";
import { createMarketChart } from "./chart.js";
import * as views from "./views.js";

const store = createStore();
let socket = null;

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
  ui: { changed: schedule, hint: views.setHint, box: views.setBox, cycleTab: () => { views.cycleTab(); schedule(); } },
});

function currentQuote() {
  const st = store.state;
  if (!st || !st.ticket) return null;
  return quotePrices(st.screen.mid, controls.state.skew, controls.state.halfWidth, store.tick, st.ticket);
}

const chart = createMarketChart(document.getElementById("chart"), store, currentQuote);

function render() {
  views.render(store, controls);
  chart.update();
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
connect();

// For the console and the browser tests.
window.mm = { store, controls };
