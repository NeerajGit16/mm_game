// @ts-check
// The reveal's charts, drawn from the coaching record alone.
//
// What really happened: true fair value against the screen you saw; print
// run-ups and unannounced moves shaded (each move from its leak); prints
// with their jumps; every fill and hedge, fills with informed clients
// ringed; the bots' fills on a toggle, off by default. Drag to zoom,
// double-click to reset, hover a fill for its ticket.
//
// Below it, your position against the disciplined bot and the flow oracle
// on the same time axis, zoomed with it.
//
// A moment, once clicked, is marked on both charts: a labelled line at its
// time -- at the print; at the ramp of a move, with its leak window
// shaded; at a ticket, with that fill's marker highlighted.

import { gap, longCategory, mmss, money, price, signedBp2 } from "./format.js";

/** @typedef {import("./protocol.js").CoachingRecord} CoachingRecord */
/** @typedef {import("./protocol.js").RevealFill} RevealFill */
/** @typedef {import("./protocol.js").Moment} Moment */
/** @typedef {uPlot} UPlot */
/** @typedef {uPlot.Options} Options */

/**
 * A moment marked on the charts.
 * @typedef {{kind: string, step: number, label: string, leak: [number, number] | null, rfq: number | null}} Marker
 */

const COLOR = {
  truth: "#f5f7fa", screen: "#6b737c", axis: "#8b95a1", grid: "#1f252c",
  buy: "#4ea1ff", sell: "#f39c3d", ring: "#f2cc4d", hedge: "#c9ced4", print: "#f2cc4d",
  runup: "rgba(170, 170, 70, 0.16)", move: "rgba(140, 100, 220, 0.24)", label: "#c9ced4",
  you: "#4ea1ff", disciplined: "#9aa3ad", oracle: "#c39bff",
  marker: "#7ef0c8", leak: "rgba(126, 240, 200, 0.16)",
};
const FONT = "11px Consolas, monospace";
const stepped = /** @type {NonNullable<typeof uPlot.paths.stepped>} */ (uPlot.paths.stepped);
const ratio = () => window.devicePixelRatio || 1;

/**
 * A marker drawn on the chart, in CSS pixels, for hovering.
 * @typedef {{x: number, y: number, html: string}} Hit
 */

/**
 * @param {HTMLElement} el
 * @param {number} minHeight
 */
function size(el, minHeight) {
  return { width: Math.max(el.clientWidth, 400), height: Math.max(el.clientHeight, minHeight) };
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} s
 * @param {boolean} up
 * @param {string} color
 * @param {boolean} filled
 */
function triangle(ctx, x, y, s, up, color, filled) {
  ctx.beginPath();
  if (up) { ctx.moveTo(x, y - s); ctx.lineTo(x + s, y + s * 0.8); ctx.lineTo(x - s, y + s * 0.8); }
  else { ctx.moveTo(x, y + s); ctx.lineTo(x + s, y - s * 0.8); ctx.lineTo(x - s, y - s * 0.8); }
  ctx.closePath();
  if (filled) { ctx.fillStyle = color; ctx.fill(); }
  else { ctx.strokeStyle = color; ctx.lineWidth = 1.5 * ratio(); ctx.stroke(); }
}

/**
 * The record's paths on one x axis for uPlot: the truth every second, the
 * screen as shown (carried forward between readings).
 * @param {CoachingRecord} record
 * @returns {[number[], (number | null)[], number[]]}
 */
function pricePaths(record) {
  const dt = record.header.config.dt;
  const rv = record.reveal;
  const tx = rv.truth.values.map((_, i) => i * rv.truth.every_steps * dt);
  const sx = rv.screen.steps.map((s) => s * dt);
  /** @type {number[]} */
  const xs = [];
  /** @type {(number | null)[]} */
  const truth = [];
  /** @type {number[]} */
  const screen = [];
  let i = 0;
  let j = 0;
  let mid = rv.screen.mids[0];
  while (i < tx.length || j < sx.length) {
    const takeTruth = j >= sx.length || (i < tx.length && tx[i] <= sx[j]);
    const x = takeTruth ? tx[i] : sx[j];
    while (j < sx.length && sx[j] <= x) mid = rv.screen.mids[j++];
    if (takeTruth) i += 1;
    if (xs.length && xs[xs.length - 1] === x) continue;
    xs.push(x);
    truth.push(takeTruth ? rv.truth.values[i - 1] : null);
    screen.push(mid);
  }
  return [xs, truth, screen];
}

/**
 * The three positions as step functions on one x axis.
 * @param {CoachingRecord} record
 * @returns {number[][]}
 */
function positionPaths(record) {
  const dt = record.header.config.dt;
  const end = record.header.config.session_length;
  const who = /** @type {const} */ (["you", "disciplined", "oracle"]);
  const paths = who.map((w) => record.reveal.positions[w]);
  const xs = [...new Set([0, end, ...paths.flatMap((p) => p.steps.map((s) => s * dt))])].sort((a, b) => a - b);
  const series = paths.map((p) => {
    let k = 0;
    return xs.map((x) => {
      while (k + 1 < p.steps.length && p.steps[k + 1] * dt <= x) k += 1;
      return p.positions[k];
    });
  });
  return [xs, ...series];
}

/**
 * @param {HTMLElement} chartEl
 * @param {HTMLElement} posEl
 * @param {HTMLElement} tipEl
 * @param {CoachingRecord} record
 */
export function createRevealCharts(chartEl, posEl, tipEl, record) {
  const dt = record.header.config.dt;
  const rv = record.reveal;
  const clients = new Map(record.world.clients.map((c) => [c.client_id, c]));
  const state = {
    showBots: false,
    drawn: { you: 0, bots: 0, rings: 0, highlighted: 0 },
    range: /** @type {[number, number] | null} */ (null),
    bands: /** @type {{kind: string, from_step: number, to_step: number}[]} */ ([]),
    marker: /** @type {Marker | null} */ (null),
    // Where the marker was last drawn, chart by chart, in steps.
    markerDrawn: /** @type {{price: number | null, position: number | null}} */ ({ price: null, position: null }),
  };
  /** @type {Hit[]} */
  let hits = [];

  /**
   * Shade run-ups and moves, under the series.
   * @param {UPlot} u
   * @param {boolean} labels
   */
  function bands(u, labels) {
    const ctx = u.ctx;
    const r = ratio();
    const { top, height } = u.bbox;
    /**
     * @param {number} a  start step
     * @param {number} b  end step
     * @param {string} color
     * @param {string} label
     */
    const band = (a, b, color, label) => {
      const x0 = u.valToPos(a * dt, "x", true);
      const x1 = u.valToPos(b * dt, "x", true);
      ctx.fillStyle = color;
      ctx.fillRect(x0, top, Math.max(x1 - x0, 1), height);
      if (labels) {
        ctx.fillStyle = COLOR.label;
        ctx.font = `${11 * r}px Consolas, monospace`;
        ctx.textAlign = "left";
        ctx.fillText(label, x0 + 4 * r, top + 14 * r);
      }
    };
    ctx.save();
    ctx.beginPath();
    ctx.rect(u.bbox.left, top, u.bbox.width, height);
    ctx.clip();
    /** @type {{kind: string, from_step: number, to_step: number}[]} */
    const drawn = [];
    for (const p of rv.prints) {
      if (!p.runup) continue;
      band(p.runup.start_step, p.step, COLOR.runup, "run-up");
      drawn.push({ kind: "run-up", from_step: p.runup.start_step, to_step: p.step });
    }
    for (const m of rv.moves) {
      band(m.inform_step, m.end_step, COLOR.move, "leak → move");
      drawn.push({ kind: "move", from_step: m.inform_step, to_step: m.end_step });
    }
    if (labels) state.bands = drawn;
    ctx.restore();
  }

  /**
   * Prints as dashed lines, with their jumps on the price chart.
   * @param {UPlot} u
   * @param {boolean} labels
   */
  function prints(u, labels) {
    const ctx = u.ctx;
    const r = ratio();
    const { top, height } = u.bbox;
    ctx.save();
    for (const p of rv.prints) {
      const x = u.valToPos(p.step * dt, "x", true);
      if (x < u.bbox.left || x > u.bbox.left + u.bbox.width) continue;
      ctx.strokeStyle = COLOR.print;
      ctx.lineWidth = r;
      ctx.setLineDash([5 * r, 5 * r]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
      ctx.setLineDash([]);
      if (labels) {
        ctx.fillStyle = COLOR.print;
        ctx.font = `${11 * r}px Consolas, monospace`;
        ctx.textAlign = "left";
        ctx.fillText(`print ${p.index + 1} · ${p.jump_bp > 0 ? "+" : "−"}${Math.abs(p.jump_bp).toFixed(2)}bp`, x + 4 * r, top + height - 6 * r);
      }
    }
    ctx.restore();
  }

  /**
   * The clicked moment: its leak window (a move), then a labelled line at
   * its time. Drawn on both charts.
   * @param {UPlot} u
   * @param {"price" | "position"} which
   */
  function markerLayer(u, which) {
    const m = state.marker;
    if (!m) { state.markerDrawn[which] = null; return; }
    const ctx = u.ctx;
    const r = ratio();
    const { left, top, width, height } = u.bbox;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.clip();
    if (m.leak) {
      const x0 = u.valToPos(m.leak[0] * dt, "x", true);
      const x1 = u.valToPos(m.leak[1] * dt, "x", true);
      ctx.fillStyle = COLOR.leak;
      ctx.fillRect(x0, top, Math.max(x1 - x0, 1), height);
      if (which === "price") {
        ctx.fillStyle = COLOR.marker;
        ctx.font = `${11 * r}px Consolas, monospace`;
        ctx.textAlign = "left";
        ctx.fillText("leak", x0 + 4 * r, top + 30 * r);
      }
    }
    const x = u.valToPos(m.step * dt, "x", true);
    ctx.strokeStyle = COLOR.marker;
    ctx.lineWidth = 2 * r;
    ctx.setLineDash([]); // a series' dash must not leak into the mark
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
    ctx.font = `bold ${12 * r}px Consolas, monospace`;
    const text = `${m.label} · ${mmss(m.step * dt)}`;
    const w = ctx.measureText(text).width + 10 * r;
    const lx = Math.min(x + 4 * r, left + width - w);
    ctx.fillStyle = "#0f1317";
    ctx.fillRect(lx, top + 2 * r, w, 17 * r);
    ctx.fillStyle = COLOR.marker;
    ctx.textAlign = "left";
    ctx.fillText(text, lx + 5 * r, top + 15 * r);
    ctx.restore();
    state.markerDrawn[which] = m.step;
  }

  /** @param {RevealFill} f */
  function fillTip(f) {
    const when = mmss(f.step * dt);
    if (f.source !== "client") {
      const what = f.source === "close" ? "closing liquidation" : "hedge";
      return `<b>${what}</b> · ${when}<br>you ${f.quantity > 0 ? "bought" : "sold"} ${Math.abs(f.quantity)} at ${price(f.price)}<br>against the truth ${money(f.edge_usd)}`;
    }
    const c = clients.get(/** @type {number} */ (f.client_id));
    const who = f.informed_about
      ? `informed about ${f.informed_about.kind} ${f.informed_about.index + 1}`
      : "uninformed";
    return `<b>RFQ ${f.rfq_ordinal}</b> · ${when} · client ${f.client_id} (${c ? longCategory(c.category) : "?"}, ${who})`
      + `<br>you ${f.quantity > 0 ? "bought" : "sold"} ${Math.abs(f.quantity)} at ${price(f.price)}`
      + `<br>edge against the truth ${money(f.edge_usd)} · through the next print/move ${money(/** @type {number} */ (f.ticket_usd))}`;
  }

  /** @param {UPlot} u */
  function marks(u) {
    const ctx = u.ctx;
    const r = ratio();
    const s = 6 * r;
    hits = [];
    state.drawn = { you: 0, bots: 0, rings: 0, highlighted: 0 };
    ctx.save();
    ctx.beginPath();
    ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
    ctx.clip();
    prints(u, true);
    if (state.showBots) {
      for (const [bot, color] of /** @type {["disciplined" | "oracle", string][]} */ ([["disciplined", COLOR.disciplined], ["oracle", COLOR.oracle]])) {
        for (const f of rv.bots[bot].fills) {
          const x = u.valToPos(f.step * dt, "x", true);
          const y = u.valToPos(f.price, "y", true);
          triangle(ctx, x, y, s * 0.8, f.quantity > 0, color, false);
          hits.push({ x: x / r, y: y / r, html: `<b>${bot === "oracle" ? "flow oracle" : "disciplined bot"}</b> · ${mmss(f.step * dt)}<br>${f.source === "client" ? `RFQ ${f.rfq_ordinal}` : f.source}: ${f.quantity > 0 ? "bought" : "sold"} ${Math.abs(f.quantity)} at ${price(f.price)}` });
          state.drawn.bots += 1;
        }
      }
    }
    for (const f of rv.fills) {
      const x = u.valToPos(f.step * dt, "x", true);
      const y = u.valToPos(f.price, "y", true);
      const color = f.quantity > 0 ? COLOR.buy : COLOR.sell;
      if (f.source === "client") triangle(ctx, x, y, s, f.quantity > 0, color, true);
      else { ctx.fillStyle = COLOR.hedge; ctx.fillRect(x - s * 0.7, y - s * 0.7, s * 1.4, s * 1.4); }
      if (f.fill_type && f.fill_type !== "uninformed") {
        ctx.strokeStyle = COLOR.ring;
        ctx.lineWidth = 2 * r;
        ctx.beginPath(); ctx.arc(x, y, s * 1.9, 0, 2 * Math.PI); ctx.stroke();
        state.drawn.rings += 1;
      }
      if (state.marker && state.marker.rfq !== null && f.source === "client" && f.rfq_ordinal === state.marker.rfq) {
        ctx.strokeStyle = COLOR.marker;
        ctx.lineWidth = 3 * r;
        ctx.beginPath(); ctx.arc(x, y, s * 2.8, 0, 2 * Math.PI); ctx.stroke();
        state.drawn.highlighted += 1;
      }
      hits.push({ x: x / r, y: y / r, html: fillTip(f) });
      state.drawn.you += 1;
    }
    ctx.restore();
    markerLayer(u, "price");
  }

  /** @param {UPlot} u */
  function hover(u) {
    const { left, top } = u.cursor;
    if (left === undefined || top === undefined || left < 0) { tipEl.hidden = true; return; }
    const ox = u.bbox.left / ratio();
    const oy = u.bbox.top / ratio();
    let best = null;
    let bestD = 10;
    for (const h of hits) {
      const d = Math.hypot(h.x - ox - left, h.y - oy - top);
      if (d < bestD) { best = h; bestD = d; }
    }
    if (!best) { tipEl.hidden = true; return; }
    const rect = u.over.getBoundingClientRect();
    tipEl.innerHTML = best.html;
    tipEl.style.left = `${rect.left + left + 14}px`;
    tipEl.style.top = `${rect.top + top + 14}px`;
    tipEl.hidden = false;
  }

  let syncing = false;
  /**
   * Keep both charts on the same time window.
   * @param {UPlot} from
   * @param {UPlot} to
   */
  function follow(from, to) {
    if (syncing) return;
    const min = from.scales.x.min;
    const max = from.scales.x.max;
    if (min === undefined || max === undefined) return;
    syncing = true;
    to.setScale("x", { min, max });
    syncing = false;
    state.range = [min, max];
  }

  /** @type {UPlot | null} */
  let pos = null; // the position chart, made after the price chart
  const [xs, truth, screen] = pricePaths(record);
  const lows = [...screen, ...rv.fills.map((f) => f.price)];
  /** @type {Options} */
  const chartOpts = {
    ...size(chartEl, 300),
    legend: { show: false },
    cursor: { drag: { x: true, y: false }, points: { show: false }, sync: { key: "reveal" } },
    scales: {
      x: { time: false },
      y: {
        range: () => {
          const lo = Math.min(...lows, ...truth.filter((v) => v !== null).map(Number));
          const hi = Math.max(...lows, ...truth.filter((v) => v !== null).map(Number));
          return [Math.floor(lo * 2) / 2 - 0.5, Math.ceil(hi * 2) / 2 + 0.5];
        },
      },
    },
    axes: [
      { stroke: COLOR.axis, font: FONT, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        values: (u, splits) => splits.map((v) => mmss(v)) },
      { stroke: COLOR.axis, font: FONT, size: 70, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        label: "price", labelFont: FONT, labelSize: 16, labelGap: 0,
        values: (u, splits) => splits.map((v) => (v / 100).toFixed(4)) },
    ],
    series: [
      {},
      { label: "screen", stroke: COLOR.screen, width: 1, paths: stepped({ align: 1 }), points: { show: false } },
      { label: "truth", stroke: COLOR.truth, width: 1.8, spanGaps: true, points: { show: false } },
    ],
    hooks: {
      drawClear: [(u) => bands(u, true)],
      draw: [marks],
      setCursor: [hover],
      setScale: [(u, key) => { if (key === "x" && pos) follow(u, pos); }],
    },
  };
  const chart = new uPlot(chartOpts, [xs, screen, truth], chartEl);

  const positions = positionPaths(record);
  // The y axis fits the three positions, zero included, with a little room:
  // a scale set by the largest of them would flatten the other two.
  const held = positions.slice(1).flat();
  const lo = Math.min(0, ...held);
  const hi = Math.max(0, ...held);
  const pad = Math.max(5, (hi - lo) * 0.08);
  /** @type {Options} */
  const posOpts = {
    ...size(posEl, 120),
    legend: { show: false },
    cursor: { drag: { x: true, y: false }, points: { show: false }, sync: { key: "reveal" } },
    scales: {
      x: { time: false },
      y: { range: () => [lo - pad, hi + pad] },
    },
    axes: [
      { stroke: COLOR.axis, font: FONT, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        values: (u, splits) => splits.map((v) => mmss(v)) },
      { stroke: COLOR.axis, font: FONT, size: 70, space: 18, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        label: "lots (+ long)", labelFont: FONT, labelSize: 16, labelGap: 0,
        values: (u, splits) => splits.map((v) => (v > 0 ? `+${Math.round(v)}` : String(Math.round(v)))) },
    ],
    series: [
      {},
      { label: "you", stroke: COLOR.you, width: 2.5, paths: stepped({ align: 1 }), points: { show: false } },
      { label: "disciplined bot", stroke: COLOR.disciplined, width: 1.75, dash: [8, 5], paths: stepped({ align: 1 }), points: { show: false } },
      { label: "flow oracle", stroke: COLOR.oracle, width: 2, dash: [2, 3], paths: stepped({ align: 1 }), points: { show: false } },
    ],
    hooks: {
      drawClear: [(u) => bands(u, false)],
      draw: [(u) => { prints(u, false); markerLayer(u, "position"); }],
      setScale: [(u, key) => { if (key === "x") follow(u, chart); }],
    },
  };
  pos = new uPlot(posOpts, /** @type {uPlot.AlignedData} */ (positions), posEl);
  chartEl.addEventListener("mouseleave", () => { tipEl.hidden = true; });

  for (const [u, el, h] of /** @type {[UPlot, HTMLElement, number][]} */ ([[chart, chartEl, 200], [pos, posEl, 100]])) {
    new ResizeObserver(() => u.setSize(size(el, h))).observe(el);
  }
  const full = /** @type {[number, number]} */ ([xs[0], xs[xs.length - 1]]);
  const posChart = pos;
  // For the tests: the position axis as drawn, and how each line is drawn.
  const posState = {
    axisLabel: String((posOpts.axes ?? [])[1]?.label ?? ""),
    get range() { return [posChart.scales.y.min, posChart.scales.y.max]; },
    lines: (posOpts.series ?? []).slice(1).map((s) => ({ stroke: String(s.stroke), dash: (s.dash ?? []).join(","), width: s.width })),
  };

  return {
    state,
    posState,
    /** The whole session's time range, as the charts open. */
    full,
    /** @param {boolean} on */
    showBots(on) { state.showBots = on; chart.redraw(false, false); },
    /**
     * Mark a moment on both charts, or clear the mark.
     * @param {Moment | null} m
     */
    mark(m) {
      if (!m) state.marker = null;
      else if (m.kind === "print") {
        const p = rv.prints.find((q) => q.index === m.index);
        state.marker = { kind: "print", step: p ? p.step : m.step, label: `print ${m.index + 1}`, leak: null, rfq: null };
      } else if (m.kind === "move") {
        const mv = rv.moves.find((q) => q.index === m.index);
        state.marker = mv
          ? { kind: "move", step: mv.start_step, label: `move ${m.index + 1} · ramp`, leak: [mv.inform_step, mv.start_step], rfq: null }
          : { kind: "move", step: m.step, label: `move ${m.index + 1}`, leak: null, rfq: null };
      } else {
        state.marker = { kind: "ticket", step: m.step, label: `RFQ ${m.index}`, leak: null, rfq: m.index };
      }
      chart.redraw(false, false);
      posChart.redraw(false, false);
    },
    /** Back to the whole session on both charts. */
    reset() { chart.setScale("x", { min: full[0], max: full[1] }); },
    /**
     * Zoom both charts to a window of the session, in steps.
     * @param {number} fromStep
     * @param {number} toStep
     */
    jumpTo(fromStep, toStep) { chart.setScale("x", { min: fromStep * dt, max: toStep * dt }); },
    /**
     * Where a point of the price chart is on the page, for pointing at it.
     * @param {number} step
     * @param {number} bp
     */
    pointOf(step, bp) {
      const rect = chart.over.getBoundingClientRect();
      return { x: rect.left + chart.valToPos(step * dt, "x"), y: rect.top + chart.valToPos(bp, "y") };
    },
  };
}

/** @typedef {ReturnType<typeof createRevealCharts>} RevealCharts */

/**
 * Wins vs best dealer, by side: every win that had a cover, at its time and
 * by how far it beat the best other dealer -- ▲ blue when the client sold
 * to you, ▼ orange when it bought from you -- over the screen's error
 * against the truth every second. One axis, in bp: a run of wide wins on
 * one side lines up with the screen off the other way.
 * @param {HTMLElement} el
 * @param {HTMLElement} tipEl
 * @param {CoachingRecord} record
 */
export function createWinsChart(el, tipEl, record) {
  const dt = record.header.config.dt;
  const err = record.reveal.screen_error;
  const xs = err.bp.map((_, i) => i * err.every_steps * dt);
  const clients = new Map(record.world.clients.map((c) => [c.client_id, c]));
  const wins = record.rfqs.flatMap((r) => {
    const o = r.outcome;
    if (!o || o.outcome !== "won" || r.vs_best_dealer_bp === null || !r.observable) return [];
    return [{ r, o, gap: r.vs_best_dealer_bp, bought: o.quantity > 0, screenError: r.observable.screen_mid - r.hidden.true_fair_value }];
  });
  const state = { drawn: { bought: 0, sold: 0 } };
  /** @type {Hit[]} */
  let hits = [];

  /** @param {(typeof wins)[number]} w */
  function tip(w) {
    const c = clients.get(w.r.observable ? w.r.observable.client.client_id : -1);
    const who = c ? `client ${c.client_id} (${longCategory(c.category)})` : "";
    return `<b>RFQ ${w.r.ordinal}</b> · ${mmss(w.r.step * dt)} · ${who}`
      + `<br>client ${w.bought ? "sold to you" : "bought from you"}: you ${w.bought ? "bought" : "sold"} ${Math.abs(w.o.quantity)} at ${price(w.o.price)}`
      + `<br>vs best dealer ${gap(w.gap)}bp · screen − truth then ${signedBp2(w.screenError)}bp`;
  }

  /** @param {UPlot} u */
  function draw(u) {
    const ctx = u.ctx;
    const r = ratio();
    const s = 6 * r;
    hits = [];
    state.drawn = { bought: 0, sold: 0 };
    ctx.save();
    ctx.beginPath();
    ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
    ctx.clip();
    const y0 = u.valToPos(0, "y", true);
    ctx.strokeStyle = COLOR.axis;
    ctx.lineWidth = r;
    ctx.beginPath(); ctx.moveTo(u.bbox.left, y0); ctx.lineTo(u.bbox.left + u.bbox.width, y0); ctx.stroke();
    for (const w of wins) {
      const x = u.valToPos(w.r.step * dt, "x", true);
      const y = u.valToPos(w.gap, "y", true);
      triangle(ctx, x, y, s, w.bought, w.bought ? COLOR.buy : COLOR.sell, true);
      hits.push({ x: x / r, y: y / r, html: tip(w) });
      state.drawn[w.bought ? "bought" : "sold"] += 1;
    }
    ctx.restore();
  }

  /** @param {UPlot} u */
  function hover(u) {
    const { left, top } = u.cursor;
    if (left === undefined || top === undefined || left < 0) { tipEl.hidden = true; return; }
    const ox = u.bbox.left / ratio();
    const oy = u.bbox.top / ratio();
    let best = null;
    let bestD = 10;
    for (const h of hits) {
      const d = Math.hypot(h.x - ox - left, h.y - oy - top);
      if (d < bestD) { best = h; bestD = d; }
    }
    if (!best) { tipEl.hidden = true; return; }
    const rect = u.over.getBoundingClientRect();
    tipEl.innerHTML = best.html;
    tipEl.style.left = `${rect.left + left + 14}px`;
    tipEl.style.top = `${rect.top + top + 14}px`;
    tipEl.hidden = false;
  }

  const ys = [0, ...err.bp, ...wins.map((w) => w.gap)];
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  /** @type {Options} */
  const opts = {
    ...size(el, 200),
    legend: { show: false },
    cursor: { drag: { x: true, y: false }, points: { show: false } },
    scales: { x: { time: false }, y: { range: () => [Math.floor(lo * 4) / 4 - 0.25, Math.ceil(hi * 4) / 4 + 0.25] } },
    axes: [
      { stroke: COLOR.axis, font: FONT, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        values: (u, splits) => splits.map((v) => mmss(v)) },
      { stroke: COLOR.axis, font: FONT, size: 60, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        label: "bp", labelFont: FONT, labelSize: 16, labelGap: 0,
        values: (u, splits) => splits.map((v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))) },
    ],
    series: [{}, { label: "screen − truth", stroke: COLOR.hedge, width: 1.25, points: { show: false } }],
    hooks: { draw: [draw], setCursor: [hover] },
  };
  const chart = new uPlot(opts, [xs, err.bp], el);
  el.addEventListener("mouseleave", () => { tipEl.hidden = true; });
  new ResizeObserver(() => chart.setSize(size(el, 200))).observe(el);
  return {
    state,
    /** Wins plotted, by side: the client sold to you (you bought), or bought from you. */
    wins: wins.map((w) => ({ rfq: w.r.ordinal, step: w.r.step, gap: w.gap, bought: w.bought })),
    /**
     * Where a point is on the page, for pointing at it.
     * @param {number} step
     * @param {number} bp
     */
    pointOf(step, bp) {
      const rect = chart.over.getBoundingClientRect();
      return { x: rect.left + chart.valToPos(step * dt, "x"), y: rect.top + chart.valToPos(bp, "y") };
    },
  };
}
