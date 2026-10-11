// @ts-check
// Charts, all of the screen and your own book -- never the truth.
//
// The Market tab's chart: the last 15 minutes plus 10 of future space; the
// price grid runs on through the future; a solid live-time line; prints as
// dashed lines, the next one with its countdown; your fills solid,
// done-away client sides hollow; your bid and ask drawn at the live line
// while a ticket is open.
//
// The P&L tab's charts: P&L and position through the whole session, as two
// charts on one time axis (never one chart with two axes), prints marked,
// the live line drawn.

import { mmss, usd } from "./format.js";

/** @typedef {import("./store.js").Store} Store */
/** @typedef {{bid: number | null, ask: number | null}} Quote */
/** @typedef {uPlot} UPlot */
/** @typedef {uPlot.Options} Options */

const PAST_S = 15 * 60;
const FUTURE_S = 10 * 60;
const COLOR = {
  ink: "#e6e9ec", axis: "#8b95a1", grid: "#1f252c", buy: "#4ea1ff", sell: "#f39c3d",
  live: "#c9ced4", print: "#6b737c", next: "#f2cc4d", zero: "#3a434e",
};
const FONT = "11px Consolas, monospace";
// Y grid steps, in bp: every one a whole number of quarter-bp ticks.
const Y_INCRS = [0.25, 0.5, 1, 1.25, 2.5, 5, 12.5, 25, 50, 125];

const ratio = () => window.devicePixelRatio || 1;
// uPlot declares its path builders optional; this one is always built in.
const stepped = /** @type {NonNullable<typeof uPlot.paths.stepped>} */ (uPlot.paths.stepped);

/**
 * @param {HTMLElement} el
 * @param {number} [minHeight]
 */
function size(el, minHeight = 200) {
  return { width: Math.max(el.clientWidth, 300), height: Math.max(el.clientHeight, minHeight) };
}

/**
 * Keep a chart the size of its box, including when a hidden tab is shown.
 * @param {UPlot} u
 * @param {HTMLElement} el
 * @param {number} [minHeight]
 */
function follow(u, el, minHeight) {
  new ResizeObserver(() => u.setSize(size(el, minHeight))).observe(el);
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} x
 * @param {number} y
 * @param {number} s
 * @param {boolean} up
 * @param {boolean} filled
 * @param {string} color
 */
function triangle(ctx, x, y, s, up, filled, color) {
  ctx.beginPath();
  if (up) { ctx.moveTo(x, y - s); ctx.lineTo(x + s, y + s * 0.8); ctx.lineTo(x - s, y + s * 0.8); }
  else { ctx.moveTo(x, y + s); ctx.lineTo(x + s, y - s * 0.8); ctx.lineTo(x - s, y - s * 0.8); }
  ctx.closePath();
  if (filled) { ctx.fillStyle = color; ctx.fill(); }
  else { ctx.strokeStyle = color; ctx.lineWidth = 1.5 * ratio(); ctx.stroke(); }
}

/**
 * A vertical line across the plot.
 * @param {UPlot} u
 * @param {number} t
 * @param {string} color
 * @param {boolean} dashed
 */
function vline(u, t, color, dashed) {
  const ctx = u.ctx;
  const r = ratio();
  const x = u.valToPos(t, "x", true);
  ctx.strokeStyle = color;
  ctx.lineWidth = (dashed ? 1 : 1.5) * r;
  ctx.setLineDash(dashed ? [5 * r, 5 * r] : []);
  ctx.beginPath();
  ctx.moveTo(x, u.bbox.top);
  ctx.lineTo(x, u.bbox.top + u.bbox.height);
  ctx.stroke();
  ctx.setLineDash([]);
  return x;
}

/** @param {UPlot} u */
function clip(u) {
  u.ctx.save();
  u.ctx.beginPath();
  u.ctx.rect(u.bbox.left, u.bbox.top, u.bbox.width, u.bbox.height);
  u.ctx.clip();
}

/**
 * @param {HTMLElement} el
 * @param {Store} store
 * @param {() => Quote | null} currentQuote
 */
export function createMarketChart(el, store, currentQuote) {
  let now = 0;
  /** @type {[number, number]} */
  let yLimits = [store.startLevel - 2, store.startLevel + 2];

  /** @type {Options} */
  const opts = {
    ...size(el),
    legend: { show: false },
    cursor: { show: false },
    select: { show: false, left: 0, top: 0, width: 0, height: 0 },
    scales: {
      x: { time: false, range: () => [now - PAST_S, now + FUTURE_S] },
      y: { range: () => yLimits },
    },
    axes: [
      {
        stroke: COLOR.axis, font: FONT,
        grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        splits: () => [-15, -10, -5, 0, 5, 10].map((m) => now + m * 60),
        values: (u, splits) => splits.map((v) => {
          const m = Math.round((v - now) / 60);
          return m === 0 ? "now" : `${m > 0 ? "+" : "−"}${Math.abs(m)}m`;
        }),
      },
      {
        stroke: COLOR.axis, font: FONT, size: 70,
        grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        incrs: Y_INCRS,
        values: (u, splits) => splits.map((v) => (v / 100).toFixed(4)),
      },
    ],
    series: [
      {},
      { stroke: COLOR.ink, width: 1.5, paths: stepped({ align: 1 }), points: { show: false } },
    ],
    hooks: { draw: [(u) => overlay(u)] },
  };
  const u = new uPlot(opts, [[], []], el);
  follow(u, el);

  /** @returns {[number[], number[]]} */
  function visibleScreen() {
    const ts = store.screen.t;
    const ys = store.screen.y;
    let i = 0;
    while (i < ts.length && ts[i] < now - PAST_S) i += 1;
    const from = Math.max(0, i - 1); // one point before the window, so the line starts at the edge
    const xs = ts.slice(from);
    const yv = ys.slice(from);
    if (xs.length && xs[xs.length - 1] < now) {
      xs.push(now); // carry the last reading to the live line
      yv.push(yv[yv.length - 1]);
    }
    return [xs, yv];
  }

  /**
   * @param {number[]} yv
   * @param {Quote | null} quote
   * @returns {[number, number]}
   */
  function computeY(yv, quote) {
    const tick = store.tick;
    const vals = [...yv];
    for (const f of store.fills) if (f.t >= now - PAST_S && f.price !== null) vals.push(f.price);
    for (const d of store.doneAway) if (d.t >= now - PAST_S) vals.push(d.level);
    if (quote) for (const p of [quote.bid, quote.ask]) if (p !== null) vals.push(p);
    if (!vals.length) return yLimits;
    let lo = Math.min(...vals) - 2 * tick;
    let hi = Math.max(...vals) + 2 * tick;
    const minSpan = 3;
    if (hi - lo < minSpan) {
      const mid = (hi + lo) / 2;
      lo = mid - minSpan / 2;
      hi = mid + minSpan / 2;
    }
    return [Math.floor(lo * 2) / 2, Math.ceil(hi * 2) / 2];
  }

  /** @param {UPlot} u */
  function overlay(u) {
    const ctx = u.ctx;
    const r = ratio();
    const top = u.bbox.top;
    /** @param {number} v */
    const X = (v) => u.valToPos(v, "x", true);
    /** @param {number} v */
    const Y = (v) => u.valToPos(v, "y", true);
    /** @param {number} px */
    const font = (px) => `${px * r}px Consolas, monospace`;
    clip(u);

    // Prints: dashed lines; the next one carries its countdown.
    const next = store.printSteps.findIndex((s) => s * store.dt > now);
    store.printSteps.forEach((step, i) => {
      const t = step * store.dt;
      if (t < now - PAST_S || t > now + FUTURE_S) return;
      const isNext = i === next;
      const x = vline(u, t, isNext ? COLOR.next : COLOR.print, true);
      ctx.textAlign = "right";
      ctx.fillStyle = COLOR.ink;
      ctx.font = font(12);
      ctx.fillText(`Print #${i + 1} · ${mmss(t)}`, x - 6 * r, top + 16 * r);
      if (isNext) {
        ctx.fillStyle = COLOR.next;
        ctx.fillText(`in ${mmss(t - now)}`, x - 6 * r, top + 32 * r);
      }
    });

    // The live line.
    const xl = vline(u, now, COLOR.live, false);
    ctx.fillStyle = COLOR.axis;
    ctx.font = font(11);
    ctx.textAlign = "left";
    ctx.fillText(mmss(now), xl + 5 * r, top + 14 * r);

    // Done-away client sides, hollow; your fills, solid; hedges as squares.
    const s = 6 * r;
    for (const d of store.doneAway) {
      if (d.t < now - PAST_S) continue;
      triangle(ctx, X(d.t), Y(d.level), s, d.side === 1, false, d.side === 1 ? COLOR.buy : COLOR.sell);
    }
    for (const f of store.fills) {
      if (f.t < now - PAST_S || f.price === null) continue;
      const color = f.qty > 0 ? COLOR.buy : COLOR.sell;
      if (f.src === "client") triangle(ctx, X(f.t), Y(f.price), s, f.qty > 0, true, color);
      else { ctx.fillStyle = color; ctx.fillRect(X(f.t) - s * 0.7, Y(f.price) - s * 0.7, s * 1.4, s * 1.4); }
    }

    // Your market at the live line while a ticket is open.
    const quote = currentQuote();
    if (quote) {
      const seg = 46 * r;
      /** @type {[number | null, string, string][]} */
      const sides = [[quote.ask, COLOR.sell, "your ask"], [quote.bid, COLOR.buy, "your bid"]];
      for (const [p, color, label] of sides) {
        if (p === null) continue;
        const y = Y(p);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3 * r;
        ctx.beginPath(); ctx.moveTo(xl + 4 * r, y); ctx.lineTo(xl + 4 * r + seg, y); ctx.stroke();
        ctx.fillStyle = COLOR.ink;
        ctx.font = font(12);
        ctx.fillText(label, xl + 10 * r + seg, y + 4 * r);
      }
    }
    ctx.restore();
  }

  return {
    update() {
      if (!store.state) return;
      now = store.state.time_s;
      const [xs, yv] = visibleScreen();
      yLimits = computeY(yv, currentQuote());
      u.setData([xs, yv]);
    },
  };
}

/**
 * The P&L tab: P&L above, position below, one time axis for the whole
 * session, from the desk's series.
 * @param {HTMLElement} pnlEl
 * @param {HTMLElement} posEl
 * @param {Store} store
 */
export function createSessionCharts(pnlEl, posEl, store) {
  let now = 0;
  const total = () => store.nSteps * store.dt;

  /** @param {UPlot} u */
  function marks(u) {
    clip(u);
    const y0 = u.valToPos(0, "y", true);
    u.ctx.strokeStyle = COLOR.zero;
    u.ctx.lineWidth = ratio();
    u.ctx.beginPath(); u.ctx.moveTo(u.bbox.left, y0); u.ctx.lineTo(u.bbox.left + u.bbox.width, y0); u.ctx.stroke();
    for (const step of store.printSteps) vline(u, step * store.dt, COLOR.next, true);
    vline(u, now, COLOR.live, false);
    u.ctx.restore();
  }

  /**
   * @param {HTMLElement} el
   * @param {string} color
   * @param {(v: number) => string} label
   * @param {number} minHeight
   */
  function chart(el, color, label, minHeight) {
    /** @type {Options} */
    const opts = {
      ...size(el, minHeight),
      legend: { show: false },
      cursor: { sync: { key: "session" }, drag: { x: false, y: false }, points: { show: false } },
      scales: {
        x: { time: false, range: () => [0, total()] },
        y: {
          range: (u, min, max) => {
            const lo = Math.min(min ?? 0, 0);
            const hi = Math.max(max ?? 0, 0);
            const pad = Math.max((hi - lo) * 0.1, 1);
            return [lo - pad, hi + pad];
          },
        },
      },
      axes: [
        {
          stroke: COLOR.axis, font: FONT,
          grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
          splits: () => Array.from({ length: Math.floor(total() / 600) + 1 }, (_, i) => i * 600),
          values: (u, splits) => splits.map((v) => mmss(v)),
        },
        {
          stroke: COLOR.axis, font: FONT, size: 64,
          grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
          values: (u, splits) => splits.map(label),
        },
      ],
      series: [{}, { stroke: color, width: 1.5, paths: stepped({ align: 1 }), points: { show: false } }],
      hooks: { draw: [marks] },
    };
    const u = new uPlot(opts, [[], []], el);
    follow(u, el, minHeight);
    return u;
  }

  const pnl = chart(pnlEl, COLOR.ink, (v) => usd(v), 120);
  const pos = chart(posEl, COLOR.buy, (v) => String(Math.round(v)), 70);

  return {
    update() {
      if (!store.state || !store.desk) return;
      now = store.state.time_s;
      const s = store.desk.pnl.series;
      const xs = s.step.map((k) => k * store.dt);
      const ps = [...s.pnl_usd];
      const qs = [...s.position];
      if (!xs.length || xs[xs.length - 1] < now) {
        xs.push(now); // carry the last value to the live line
        ps.push(ps.length ? ps[ps.length - 1] : 0);
        qs.push(qs.length ? qs[qs.length - 1] : 0);
      }
      pnl.setData([xs, ps]);
      pos.setData([xs, qs]);
    },
  };
}
