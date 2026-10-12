// @ts-check
// The (i) panels: what every column of a table means and which way its
// sign runs. The definitions are the server's (/glossary), the same ones
// summary.md prints, so the page never explains a number its own way.

import { escapeHtml } from "./format.js";

/** @typedef {import("./protocol.js").GlossaryTable} GlossaryTable */

/** @type {Record<string, GlossaryTable> | null} */
let tables = null;
/** @type {string | null} */
let shown = null;

/** @param {string} id */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** Fetch the definitions once; the panels stay empty without them. */
export async function loadGlossary() {
  const r = await fetch("/glossary");
  if (r.ok) tables = /** @type {import("./protocol.js").GlossaryMsg} */ (await r.json()).tables;
  return tables;
}

/**
 * Open the panel for one table.
 * @param {string} key
 */
export function showGlossary(key) {
  const t = tables ? tables[key] : undefined;
  $("gl-title").textContent = t ? t.title : "What the columns mean";
  $("gl-intro").textContent = t ? t.intro : "The definitions have not loaded.";
  $("gl-list").innerHTML = t
    ? t.entries.map((e) => `<dt>${escapeHtml(e.label)}</dt><dd>${escapeHtml(e.meaning)}</dd>`).join("")
    : "";
  $("glossary").dataset.table = key;
  $("glossary").hidden = false;
  shown = key;
}

export function hideGlossary() {
  $("glossary").hidden = true;
  shown = null;
}

export const glossaryShown = () => shown;

/**
 * Wire the close button and the backdrop to ``close``.
 * @param {() => void} close
 */
export function bindGlossary(close) {
  $("gl-close").addEventListener("click", close);
  $("glossary").addEventListener("click", (e) => { if (e.target === $("glossary")) close(); });
}

// For the browser tests: which definitions the page holds.
/** @type {any} */ (window).mmGlossary = { tables: () => tables };
