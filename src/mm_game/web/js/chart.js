// The Market tab's chart: the screen only, never the truth. The last 15
// minutes plus 10 of future space; the price grid runs on through the
// future; a solid live-time line; prints as dashed lines, the next one
// with its countdown; your fills solid, done-away client sides hollow;
// your bid and ask drawn at the live line while a ticket is open.

import { mmss } from "./format.js";

const PAST_S = 15 * 60;
const FUTURE_S = 10 * 60;
const COLOR = {
  ink: "#e6e9ec", axis: "#8b95a1", grid: "#1f252c", buy: "#4ea1ff", sell: "#f39c3d",
  live: "#c9ced4", print: "#6b737c", next: "#f2cc4d",
};
// Y grid steps, in bp: every one a whole number of quarter-bp ticks.
const Y_INCRS = [0.25, 0.5, 1, 1.25, 2.5, 5, 12.5, 25, 50, 125];

export function createMarketChart(el, store, currentQuote) {
  let now = 0;
  let yLimits = [store.startLevel - 2, store.startLevel + 2];
  const ratio = () => window.devicePixelRatio || 1;

  const opts = {
    width: Math.max(el.clientWidth, 300),
    height: Math.max(el.clientHeight, 200),
    legend: { show: false },
    cursor: { show: false },
    select: { show: false },
    scales: {
      x: { time: false, range: () => [now - PAST_S, now + FUTURE_S] },
      y: { range: () => yLimits },
    },
    axes: [
      {
        stroke: COLOR.axis, font: "11px Consolas, monospace",
        grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        splits: () => [-15, -10, -5, 0, 5, 10].map((m) => now + m * 60),
        values: (u, splits) => splits.map((v) => {
          const m = Math.round((v - now) / 60);
          return m === 0 ? "now" : `${m > 0 ? "+" : "−"}${Math.abs(m)}m`;
        }),
      },
      {
        stroke: COLOR.axis, font: "11px Consolas, monospace", size: 70,
        grid: { stroke: COLOR.grid, width: 1 }, ticks: { stroke: COLOR.grid, width: 1 },
        incrs: Y_INCRS,
        values: (u, splits) => splits.map((v) => (v / 100).toFixed(4)),
      },
    ],
    series: [
      {},
      { stroke: COLOR.ink, width: 1.5, paths: uPlot.paths.stepped({ align: 1 }), points: { show: false } },
    ],
    hooks: { draw: [(u) => overlay(u)] },
  };
  const u = new uPlot(opts, [[], []], el);
  new ResizeObserver(() => u.setSize({ width: Math.max(el.clientWidth, 300), height: Math.max(el.clientHeight, 200) })).observe(el);

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

  function tri(ctx, x, y, s, up, filled, color) {
    ctx.beginPath();
    if (up) { ctx.moveTo(x, y - s); ctx.lineTo(x + s, y + s * 0.8); ctx.lineTo(x - s, y + s * 0.8); }
    else { ctx.moveTo(x, y + s); ctx.lineTo(x + s, y - s * 0.8); ctx.lineTo(x - s, y - s * 0.8); }
    ctx.closePath();
    if (filled) { ctx.fillStyle = color; ctx.fill(); }
    else { ctx.strokeStyle = color; ctx.lineWidth = 1.5 * ratio(); ctx.stroke(); }
  }

  function overlay(u) {
    const ctx = u.ctx;
    const r = ratio();
    const { left, top, width, height } = u.bbox;
    const X = (v) => u.valToPos(v, "x", true);
    const Y = (v) => u.valToPos(v, "y", true);
    const font = (px, bold = false) => `${bold ? "600 " : ""}${px * r}px Consolas, monospace`;
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.clip();

    // Prints: dashed lines; the next one carries its countdown.
    const next = store.printSteps.findIndex((s) => s * store.dt > now);
    store.printSteps.forEach((step, i) => {
      const t = step * store.dt;
      if (t < now - PAST_S || t > now + FUTURE_S) return;
      const x = X(t);
      const isNext = i === next;
      ctx.strokeStyle = isNext ? COLOR.next : COLOR.print;
      ctx.lineWidth = r;
      ctx.setLineDash([5 * r, 5 * r]);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
      ctx.setLineDash([]);
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
    const xl = X(now);
    ctx.strokeStyle = COLOR.live;
    ctx.lineWidth = 1.5 * r;
    ctx.beginPath(); ctx.moveTo(xl, top); ctx.lineTo(xl, top + height); ctx.stroke();
    ctx.fillStyle = COLOR.axis;
    ctx.font = font(11);
    ctx.textAlign = "left";
    ctx.fillText(mmss(now), xl + 5 * r, top + 14 * r);

    // Done-away client sides, hollow; your fills, solid; hedges as squares.
    const s = 6 * r;
    for (const d of store.doneAway) {
      if (d.t < now - PAST_S) continue;
      tri(ctx, X(d.t), Y(d.level), s, d.side === 1, false, d.side === 1 ? COLOR.buy : COLOR.sell);
    }
    for (const f of store.fills) {
      if (f.t < now - PAST_S || f.price === null) continue;
      const color = f.qty > 0 ? COLOR.buy : COLOR.sell;
      if (f.src === "client") tri(ctx, X(f.t), Y(f.price), s, f.qty > 0, true, color);
      else { ctx.fillStyle = color; ctx.fillRect(X(f.t) - s * 0.7, Y(f.price) - s * 0.7, s * 1.4, s * 1.4); }
    }

    // Your market at the live line while a ticket is open.
    const quote = currentQuote();
    if (quote) {
      const seg = 46 * r;
      for (const [p, color, label] of [[quote.ask, COLOR.sell, "your ask"], [quote.bid, COLOR.buy, "your bid"]]) {
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
