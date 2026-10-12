// @ts-check
// Quote and hedge entry. A quote is a skew and a half-width around the
// screen, in whole ticks; both are sticky, so each new ticket starts from
// the last market you made and Enter alone sends it again. Nothing here
// skews for you: that is the decision the game teaches.
//
// Every key has a matching action, and every action a button.

import * as P from "./protocol.js";

/** @typedef {import("./store.js").Store} Store */
/** @typedef {import("./protocol.js").Ticket} Ticket */

/**
 * What the page does when an action changes something on screen.
 * @typedef {Object} Ui
 * @property {() => void} changed
 * @property {(text: string | null, ms?: number) => void} hint
 * @property {(box: {text: string} | null) => void} box
 * @property {(open: boolean) => void} handbook
 * @property {(key: string | null) => void} glossary  open one table's (i) panel, or close it
 * @property {() => void} cycleTab
 */

/** @type {Record<string, number>} */
export const HEDGE_KEYS = { q: 10, w: 25, e: 50, a: -10, s: -25, d: -50 };
export const DEFAULT_HALF_WIDTH = 2;
const FLATTEN_WINDOW_MS = 1000;
const MAX_BOX_DIGITS = 3;
// Keys that act once per press: holding them must not repeat a trade.
const NO_REPEAT = new Set(["Enter", "p", "0", "h", "f", " ", "Tab", "?", "1", "2", "4", ...Object.keys(HEDGE_KEYS)]);

/**
 * The two prices a skew and half-width make around the screen. On a
 * one-way ticket only the side the client asked for is sent: a client who
 * buys lifts your ask, so the bid goes as null.
 * @param {number} mid
 * @param {number} skew
 * @param {number} halfWidth
 * @param {number} tick
 * @param {Ticket | null} ticket
 * @returns {{bid: number | null, ask: number | null}}
 */
export function quotePrices(mid, skew, halfWidth, tick, ticket) {
  const centre = mid + skew * tick;
  /** @type {number | null} */
  let bid = centre - halfWidth * tick;
  /** @type {number | null} */
  let ask = centre + halfWidth * tick;
  if (ticket && ticket.one_way) {
    if (ticket.side === 1) bid = null;
    else ask = null;
  }
  return { bid, ask };
}

/**
 * @param {{store: Store, send: (message: P.Action) => void, ui: Ui, now?: () => number}} deps
 */
export function createControls({ store, send, ui, now = () => performance.now() }) {
  /** @type {{skew: number, halfWidth: number, box: {text: string} | null, lastF: number, overlay: "handbook" | "glossary" | null}} */
  const c = { skew: 0, halfWidth: DEFAULT_HALF_WIDTH, box: null, lastF: -Infinity, overlay: null };
  const ticket = () => (store.state ? store.state.ticket : null);
  const live = () => !!store.state && store.state.status === "live";
  // Reading the handbook or an (i) panel pauses the clock, and it stays
  // paused when the window shuts: you resume with Space when you are ready.
  const pauseToRead = () => {
    const st = store.state;
    if (st && st.status === "live" && !st.clock.paused) send(P.pause());
  };

  const actions = {
    /** @param {number} d */
    skew(d) { c.skew += d; ui.changed(); },
    resetSkew() { c.skew = 0; ui.changed(); },
    /** @param {number} d */
    width(d) { c.halfWidth = Math.max(1, c.halfWidth + d); ui.changed(); },
    send() {
      const st = store.state;
      if (!st || !st.ticket) return;
      const { bid, ask } = quotePrices(st.screen.mid, c.skew, c.halfWidth, store.tick, st.ticket);
      send(P.quote(st.ticket.ordinal, bid, ask));
    },
    pass() {
      const t = ticket();
      if (t) send(P.pass(t.ordinal));
    },
    /** @param {number} quantity */
    hedge(quantity) {
      if (quantity && live()) send(P.hedge(quantity));
    },
    flattenPress() {
      const t = now();
      if (t - c.lastF <= FLATTEN_WINDOW_MS) {
        c.lastF = -Infinity;
        ui.hint(null);
        const position = store.state ? store.state.book.position : 0;
        if (position) actions.hedge(-position);
      } else {
        c.lastF = t;
        ui.hint("Press F again within a second to flatten", FLATTEN_WINDOW_MS);
      }
    },
    openBox() { c.box = { text: "" }; ui.box(c.box); },
    /** @param {string} digit */
    boxDigit(digit) {
      if (c.box && c.box.text.length < MAX_BOX_DIGITS) { c.box.text += digit; ui.box(c.box); }
    },
    boxBack() {
      if (c.box) { c.box.text = c.box.text.slice(0, -1); ui.box(c.box); }
    },
    /** @param {number} sign */
    boxTrade(sign) {
      if (!c.box) return;
      const size = parseInt(c.box.text || "0", 10);
      c.box = null;
      ui.box(null);
      if (size > 0) actions.hedge(sign * size);
    },
    boxCancel() { c.box = null; ui.box(null); },
    togglePause() {
      if (!store.begun) { send(P.begin()); return; }
      const st = store.state;
      if (!st || st.status !== "live") return;
      send(st.clock.paused ? P.resume() : P.pause());
    },
    toggleHandbook() {
      if (c.overlay === "handbook") { c.overlay = null; ui.handbook(false); return; }
      if (c.overlay || !store.begun || store.summary) return; // the start screen already shows it
      c.overlay = "handbook";
      ui.handbook(true);
      pauseToRead();
    },
    /** @param {string} key  the table whose columns to explain */
    openGlossary(key) {
      if (c.overlay === "handbook" || store.summary) return;
      c.overlay = "glossary";
      ui.glossary(key);
      if (store.begun) pauseToRead();
    },
    closeGlossary() {
      if (c.overlay === "glossary") c.overlay = null;
      ui.glossary(null);
    },
    /** @param {number} n */
    speed(n) { if (live()) send(P.setSpeed(n)); },
    cycleTab() { ui.cycleTab(); },
    quit() { if (live()) send(P.quit()); },
  };

  /** @param {KeyboardEvent} e */
  function onKey(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    if (store.summary) return false; // the session is over
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;

    // While the size box is open every key goes to it: S sells the typed
    // size, digits are the size, nothing reaches the session.
    if (c.box) {
      if (/^[0-9]$/.test(key)) actions.boxDigit(key);
      else if (key === "Backspace") actions.boxBack();
      else if (key === "b" && !e.repeat) actions.boxTrade(1);
      else if (key === "s" && !e.repeat) actions.boxTrade(-1);
      else if (key === "Escape") actions.boxCancel();
      return true;
    }
    // While the handbook or an (i) panel is open, nothing trades.
    if (c.overlay === "handbook") {
      if ((key === "?" || key === "Escape") && !e.repeat) actions.toggleHandbook();
      return true;
    }
    if (c.overlay === "glossary") {
      if (key === "Escape" && !e.repeat) actions.closeGlossary();
      return true;
    }
    if (e.repeat && NO_REPEAT.has(key)) return true;

    if (key === "ArrowLeft") actions.skew(-1);
    else if (key === "ArrowRight") actions.skew(1);
    else if (key === "ArrowUp") actions.width(1);
    else if (key === "ArrowDown") actions.width(-1);
    else if (key === "0") actions.resetSkew();
    else if (key === "Enter") actions.send();
    else if (key === "p") actions.pass();
    else if (key in HEDGE_KEYS) actions.hedge(HEDGE_KEYS[key]);
    else if (key === "h") actions.openBox();
    else if (key === "f") actions.flattenPress();
    else if (key === " ") actions.togglePause();
    else if (key === "?") actions.toggleHandbook();
    else if (key === "1" || key === "2" || key === "4") actions.speed(Number(key));
    else if (key === "Tab") actions.cycleTab();
    else return false;
    return true;
  }

  return { state: c, actions, onKey };
}

/** @typedef {ReturnType<typeof createControls>} Controls */
