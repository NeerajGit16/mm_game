// Display formatting. Prices travel in bp (9550.25) and show as 95.5025.

export const MINUS = "−";

export const CATEGORY = {
  hedge_fund: ["HF", "Hedge fund"],
  real_money: ["RM", "Real money"],
  corporate: ["Corp", "Corporate"],
};

const pad = (n) => String(n).padStart(2, "0");

export function price(bp) {
  return bp === null || bp === undefined ? "—" : (bp / 100).toFixed(4);
}

export function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds + 1e-9));
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
}

export function mmss(seconds) {
  const s = Math.max(0, Math.floor(seconds + 1e-9));
  return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
}

export function signedInt(n) {
  if (n > 0) return `+${n}`;
  if (n < 0) return `${MINUS}${-n}`;
  return "0";
}

export function usd(x, signed = true) {
  const a = Math.round(Math.abs(x));
  const body = "$" + a.toLocaleString("en-US");
  if (!signed || a === 0) return body;
  return (x < 0 ? MINUS : "+") + body;
}

export function bp(x, digits = 1) {
  if (x === null || x === undefined) return "—";
  const r = Number(x.toFixed(digits));
  if (r === 0) return (0).toFixed(digits) + "bp";
  return (r > 0 ? "+" : MINUS) + Math.abs(r).toFixed(digits) + "bp";
}

export function tone(x) {
  if (x === null || x === undefined || Math.abs(x) < 1e-9) return "";
  return x > 0 ? "pos" : "neg";
}

export function ago(seconds) {
  if (seconds < 60) return `${Math.floor(seconds)} s ago`;
  return `${Math.floor(seconds / 60)} min ago`;
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
