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

import { longCategory, mmss, money, price } from "./format.js";

/** @typedef {import("./protocol.js").CoachingRecord} CoachingRecord */
/** @typedef {import("./protocol.js").RevealFill} RevealFill */
/** @typedef {uPlot} UPlot */
/** @typedef {uPlot.Options} Options */

const COLOR = {
  truth: "#f5f7fa", screen: "#6b737c", axis: "#8b95a1", grid: "#1f252c",
  buy: "#4ea1ff", sell: "#f39c3d", ring: "#f2cc4d", hedge: "#c9ced4", print: "#f2cc4d",
  runup: "rgba(170, 170, 70, 0.16)", move: "rgba(140, 100, 220, 0.24)", label: "#c9ced4",
  you: "#4ea1ff", disciplined: "#9aa3ad", oracle: "#b48cff",
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
    drawn: { you: 0, bots: 0, rings: 0 },
    range: /** @type {[number, number] | null} */ (null),
    bands: /** @type {{kind: string, from_step: number, to_step: number}[]} */ ([]),
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
      + `<br>edge against the truth ${money(f.edge_usd)} · through the next event ${money(/** @type {number} */ (f.ticket_usd))}`;
  }

  /** @param {UPlot} u */
  function marks(u) {
    const ctx = u.ctx;
    const r = ratio();
    const s = 6 * r;
    hits = [];
    state.drawn = { you: 0, bots: 0, rings: 0 };
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
      hits.push({ x: x / r, y: y / r, html: fillTip(f) });
      state.drawn.you += 1;
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
  /** @type {Options} */
  const posOpts = {
    ...size(posEl, 120),
    legend: { show: false },
    cursor: { drag: { x: true, y: false }, points: { show: false }, sync: { key: "reveal" } },
    scales: {
      x: { time: false },
      y: {
        range: (u, min, max) => {
          const m = Math.max(Math.abs(min ?? 0), Math.abs(max ?? 0), 10);
          return [-m * 1.1, m * 1.1];
        },
      },
    },
    axes: [
      { stroke: COLOR.axis, font: FONT, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        values: (u, splits) => splits.map((v) => mmss(v)) },
      { stroke: COLOR.axis, font: FONT, size: 70, space: 16, grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        values: (u, splits) => splits.map((v) => String(Math.round(v))) },
    ],
    series: [
      {},
      { label: "you", stroke: COLOR.you, width: 2, paths: stepped({ align: 1 }), points: { show: false } },
      { label: "disciplined bot", stroke: COLOR.disciplined, width: 1.5, dash: [5, 4], paths: stepped({ align: 1 }), points: { show: false } },
      { label: "flow oracle", stroke: COLOR.oracle, width: 1.5, paths: stepped({ align: 1 }), points: { show: false } },
    ],
    hooks: {
      drawClear: [(u) => bands(u, false)],
      draw: [(u) => prints(u, false)],
      setScale: [(u, key) => { if (key === "x") follow(u, chart); }],
    },
  };
  pos = new uPlot(posOpts, /** @type {uPlot.AlignedData} */ (positions), posEl);
  chartEl.addEventListener("mouseleave", () => { tipEl.hidden = true; });

  for (const [u, el, h] of /** @type {[UPlot, HTMLElement, number][]} */ ([[chart, chartEl, 300], [pos, posEl, 120]])) {
    new ResizeObserver(() => u.setSize(size(el, h))).observe(el);
  }

  return {
    state,
    /** @param {boolean} on */
    showBots(on) { state.showBots = on; chart.redraw(false, false); },
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
