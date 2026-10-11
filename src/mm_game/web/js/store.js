// Everything the page knows, built only from messages the server sent.
// The page never holds anything else: reload it and the server's history
// rebuilds the same store.

export function createStore() {
  return {
    config: null,
    dt: 0.1,
    nSteps: 0,
    tick: 0.25,
    startLevel: 0,
    limit: 200,
    preset: null,
    printSteps: [],
    roster: [],
    replayOfPlayedSeed: false,
    screen: { t: [], y: [] }, // seconds, bp: one point per screen reading
    fills: [], // your trades: {t, price, qty, src: client | hedge | close}
    doneAway: [], // RFQs that traded elsewhere: {t, level, side}
    state: null, // the latest snapshot
    desk: null, // the latest desk
    begun: false,
    lastHedge: null,
    lastBlock: null,
    errors: [],
    revealed: false,
    summary: null,
    version: 0,
  };
}

export function apply(store, msg) {
  switch (msg.type) {
    case "event":
      applyEvent(store, msg);
      break;
    case "state":
      store.state = msg;
      break;
    case "desk":
      store.desk = msg;
      break;
    case "host":
      store.begun = msg.begun;
      break;
    case "error":
      store.errors.push({ at: performance.now(), detail: msg.detail });
      break;
    case "reveal":
      store.revealed = true;
      break;
    case "summary":
      store.summary = msg;
      break;
  }
  store.version += 1;
}

function applyEvent(store, m) {
  const d = m.data;
  const t = m.step * store.dt;
  switch (m.kind) {
    case "session_start": {
      const c = d.config;
      store.config = c;
      store.dt = c.dt;
      store.nSteps = Math.round(c.session_length / c.dt);
      store.tick = c.tick_size;
      store.startLevel = c.start_level;
      store.limit = c.position_limit_lots;
      store.preset = d.settings.preset;
      store.printSteps = d.print_steps;
      store.roster = d.roster;
      store.replayOfPlayedSeed = d.replay_of_played_seed;
      break;
    }
    case "screen_update":
      store.screen.t.push(t);
      store.screen.y.push(d.mid);
      break;
    case "rfq_resolved":
      if (d.outcome === "won") store.fills.push({ t, price: d.price, qty: d.quantity, src: "client" });
      else if (d.outcome === "done_away") store.doneAway.push({ t, level: d.cover, side: d.side });
      break;
    case "hedge_executed":
      if (d.quantity) store.fills.push({ t, price: d.price, qty: d.quantity, src: "hedge" });
      store.lastHedge = { ...d, step: m.step };
      break;
    case "limit_blocked":
      store.lastBlock = { ...d, step: m.step };
      break;
    case "close_liquidation":
      if (d.quantity) store.fills.push({ t, price: d.price, qty: d.quantity, src: "close" });
      break;
  }
}
