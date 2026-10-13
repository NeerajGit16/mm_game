// @ts-check
// Display formatting. Prices travel in bp (9550.25) and show as 95.5025.

export const MINUS = "−";

/** @type {Record<string, [string, string]>} */
export const CATEGORY = {
  hedge_fund: ["HF", "Hedge fund"],
  real_money: ["RM", "Real money"],
  corporate: ["Corp", "Corporate"],
};

/** @param {number} n */
const pad = (n) => String(n).padStart(2, "0");

/** @param {number | null | undefined} bp */
export function price(bp) {
  return bp === null || bp === undefined ? "—" : (bp / 100).toFixed(4);
}

/** @param {number} seconds */
export function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds + 1e-9));
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

/** @param {number} seconds */
export function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds + 1e-9));
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

/** @param {number} n */
export function signedInt(n) {
  if (n > 0) return `+${n}`;
  if (n < 0) return `${MINUS}${-n}`;
  return "0";
}

/**
 * Dollars, whole, signed unless asked not to: +$3,240, −$500, $0.
 * @param {number} x
 * @param {boolean} [signed]
 */
export function usd(x, signed = true) {
  const a = Math.round(Math.abs(x));
  const body = "$" + a.toLocaleString("en-US");
  if (!signed || a === 0) return body;
  return (x < 0 ? MINUS : "+") + body;
}

/**
 * Basis points, signed: +0.6bp, −1.1bp; — when there is nothing yet.
 * @param {number | null | undefined} x
 * @param {number} [digits]
 */
export function bp(x, digits = 1) {
  if (x === null || x === undefined) return "—";
  const r = Number(x.toFixed(digits));
  if (r === 0) return (0).toFixed(digits) + "bp";
  return (r > 0 ? "+" : MINUS) + Math.abs(r).toFixed(digits) + "bp";
}

/**
 * Two decimals as Python's format writes them. They differ only on an
 * exact tie, which for two decimals is an odd eighth (0.125, 0.375, ...):
 * toFixed rounds it up, Python to even.
 * @param {number} a  a non-negative number
 */
export function fixed2(a) {
  if (Number.isInteger(a * 8) && !Number.isInteger(a * 4)) {
    const lo = Math.floor(a * 100);
    return ((lo % 2 === 0 ? lo : lo + 1) / 100).toFixed(2);
  }
  return a.toFixed(2);
}

/**
 * bp to two places, signed, no sign on zero: +0.75, −0.50, 0.00 -- as
 * summary.md writes them.
 * @param {number} x
 */
export function signedBp2(x) {
  const text = fixed2(Math.abs(x));
  return text === "0.00" ? text : (x < 0 ? MINUS : "+") + text;
}

/**
 * A price against the best other dealer: +0.75, −0.50, 0.00 on a tie, and
 * blank when there was nothing to compare.
 * @param {number | null | undefined} x
 */
export function gap(x) {
  return x === null || x === undefined ? "" : signedBp2(x);
}

/**
 * The CSS class for a signed number: pos, neg or nothing.
 * @param {number | null | undefined} x
 */
export function tone(x) {
  if (x === null || x === undefined || Math.abs(x) < 1e-9) return "";
  return x > 0 ? "pos" : "neg";
}

/** @param {number} seconds */
export function ago(seconds) {
  if (seconds < 60) return `${Math.floor(seconds)} s ago`;
  return `${Math.floor(seconds / 60)} min ago`;
}

/** @param {unknown} s */
export function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => /** @type {Record<string, string>} */ ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

/** @param {string} category */
export const shortCategory = (category) => (CATEGORY[category] || [category])[0];
/** @param {string} category */
export const longCategory = (category) => (CATEGORY[category] || [category, category])[1];

/**
 * Round half to even, as Python's format does: the reveal's dollars must be
 * the same numbers as summary.md's.
 * @param {number} x  a non-negative number
 */
export function halfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Dollars as summary.md writes them: whole, halves to even; signed (+$/−$)
 * for P&L contributions, or with a minus only.
 * @param {number} x
 * @param {boolean} [signed]
 */
export function money(x, signed = true) {
  const negative = x < -0.5;
  const body = "$" + halfEven(Math.abs(x)).toLocaleString("en-US");
  if (negative) return MINUS + body;
  return signed ? "+" + body : body;
}

/**
 * A ratio as a percentage, or n/a.
 * @param {number | null | undefined} r
 */
export function percent(r) {
  return r === null || r === undefined ? "n/a" : `${Math.round(r * 100)}%`;
}
