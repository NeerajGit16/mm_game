// @ts-check
// Everything the page knows, built only from messages the server sent.
// The page never holds anything else: reload it and the server's history
// rebuilds the same store.

/** @typedef {import("./protocol.js").ServerMessage} ServerMessage */
/** @typedef {import("./protocol.js").EventMsg} EventMsg */

/**
 * @typedef {Object} Fill
 * @property {number} t  seconds into the session
 * @property {number | null} price
 * @property {number} qty
 * @property {"client" | "hedge" | "close"} src
 */

/**
 * @typedef {Object} Store
 * @property {Record<string, any> | null} config
 * @property {number} dt
 * @property {number} nSteps
 * @property {number} tick
 * @property {number} startLevel
 * @property {number} limit
 * @property {number | null} preset
 * @property {number[]} printSteps
 * @property {{client_id: number, category: string}[]} roster
 * @property {boolean} replayOfPlayedSeed
 * @property {{t: number[], y: number[]}} screen  one point per screen reading: seconds, bp
 * @property {Fill[]} fills
 * @property {{t: number, level: number, side: number}[]} doneAway
 * @property {EventMsg[]} events  every event, in order
 * @property {import("./protocol.js").StateMsg | null} state
 * @property {import("./protocol.js").DeskMsg | null} desk
 * @property {import("./protocol.js").HandbookMsg | null} handbook
 * @property {boolean} begun
 * @property {(Record<string, any> & {step: number}) | null} lastHedge
 * @property {(Record<string, any> & {step: number}) | null} lastBlock
 * @property {{at: number, detail: string}[]} errors
 * @property {boolean} revealed
 * @property {import("./protocol.js").CoachingRecord | null} record  the coaching record, after the end
 * @property {import("./protocol.js").TrackMsg | null} track  the track record, after the end
 * @property {import("./protocol.js").SummaryMsg | null} summary
 * @property {number} version
 */

/**
 * A store once the first snapshot has arrived, and once the first desk has.
 * Renderers take these: the one null check happens where rendering starts.
 * @typedef {Store & {state: import("./protocol.js").StateMsg}} LiveStore
 * @typedef {LiveStore & {desk: import("./protocol.js").DeskMsg}} DeskStore
 */

/** @returns {Store} */
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
    screen: { t: [], y: [] },
    fills: [],
    doneAway: [],
    events: [],
    state: null,
    desk: null,
    handbook: null,
    begun: false,
    lastHedge: null,
    lastBlock: null,
    errors: [],
    revealed: false,
    record: null,
    track: null,
    summary: null,
    version: 0,
  };
}

/**
 * @param {Store} store
 * @param {ServerMessage} msg
 */
export function apply(store, msg) {
  switch (msg.type) {
    case "event":
      store.events.push(msg);
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
    case "handbook":
      store.handbook = msg;
      break;
    case "error":
      store.errors.push({ at: performance.now(), detail: msg.detail });
      break;
    case "reveal":
      store.revealed = true;
      store.record = msg.record;
      break;
    case "track":
      store.track = msg;
      break;
    case "summary":
      store.summary = msg;
      break;
  }
  store.version += 1;
}

/**
 * @param {Store} store
 * @param {EventMsg} m
 */
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
