// Quote and hedge entry. A quote is a skew and a half-width around the
// screen, in whole ticks; both are sticky, so each new ticket starts from
// the last market you made and Enter alone sends it again. Nothing here
// skews for you: that is the decision the game teaches.
//
// Every key has a matching action, and every action a button.

import * as P from "./protocol.js";

export const HEDGE_KEYS = { q: 10, w: 25, e: 50, a: -10, s: -25, d: -50 };
export const DEFAULT_HALF_WIDTH = 2;
const FLATTEN_WINDOW_MS = 1000;
const MAX_BOX_DIGITS = 3;
// Keys that act once per press: holding them must not repeat a trade.
const NO_REPEAT = new Set(["Enter", "p", "0", "h", "f", " ", "Tab", "1", "2", "4", ...Object.keys(HEDGE_KEYS)]);

// The two prices a skew and half-width make around the screen. On a
// one-way ticket only the side the client asked for is sent: a client who
// buys lifts your ask, so the bid goes as null.
export function quotePrices(mid, skew, halfWidth, tick, ticket) {
  const centre = mid + skew * tick;
  let bid = centre - halfWidth * tick;
  let ask = centre + halfWidth * tick;
  if (ticket && ticket.one_way) {
    if (ticket.side === 1) bid = null;
    else ask = null;
  }
  return { bid, ask };
}

export function createControls({ store, send, ui, now = () => performance.now() }) {
  const c = { skew: 0, halfWidth: DEFAULT_HALF_WIDTH, box: null, lastF: -Infinity };
  const ticket = () => (store.state ? store.state.ticket : null);
  const live = () => !!store.state && store.state.status === "live";

  const actions = {
    skew(d) { c.skew += d; ui.changed(); },
    resetSkew() { c.skew = 0; ui.changed(); },
    width(d) { c.halfWidth = Math.max(1, c.halfWidth + d); ui.changed(); },
    send() {
      const t = ticket();
      if (!t) return;
      const { bid, ask } = quotePrices(store.state.screen.mid, c.skew, c.halfWidth, store.tick, t);
      send(P.quote(t.ordinal, bid, ask));
    },
    pass() {
      const t = ticket();
      if (t) send(P.pass(t.ordinal));
    },
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
    boxDigit(digit) {
      if (c.box && c.box.text.length < MAX_BOX_DIGITS) { c.box.text += digit; ui.box(c.box); }
    },
    boxBack() {
      if (c.box) { c.box.text = c.box.text.slice(0, -1); ui.box(c.box); }
    },
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
      if (!live()) return;
      send(store.state.clock.paused ? P.resume() : P.pause());
    },
    speed(n) { if (live()) send(P.setSpeed(n)); },
    cycleTab() { ui.cycleTab(); },
    quit() { if (live()) send(P.quit()); },
  };

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
    else if (key === "1" || key === "2" || key === "4") actions.speed(Number(key));
    else if (key === "Tab") actions.cycleTab();
    else return false;
    return true;
  }

  return { state: c, actions, onKey };
}
