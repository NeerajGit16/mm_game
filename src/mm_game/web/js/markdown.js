// @ts-check
// Just enough Markdown for summary.md: headings, tables, nested lists,
// paragraphs, bold and italics. Everything is escaped first.

import { escapeHtml } from "./format.js";

/** @param {string} text */
function inline(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, "$1<em>$2</em>");
}

/** @param {string} line */
function cells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

/** @param {string[]} lines */
function table(lines) {
  const [head, , ...body] = lines;
  const th = cells(head).map((c) => `<th>${inline(c)}</th>`).join("");
  const rows = body.map((l) => `<tr>${cells(l).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${rows}</tbody></table>`;
}

/** @param {string[]} lines */
function list(lines) {
  let html = "<ul>";
  let nested = false;
  for (const l of lines) {
    const deep = /^\s{2,}- /.test(l);
    const text = inline(l.replace(/^\s*- /, ""));
    if (deep && !nested) { html += "<ul>"; nested = true; }
    if (!deep && nested) { html += "</ul>"; nested = false; }
    html += `<li>${text}</li>`;
  }
  return html + (nested ? "</ul>" : "") + "</ul>";
}

/** @param {string} source */
export function renderMarkdown(source) {
  const lines = source.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    const heading = l.match(/^(#{1,6}) (.*)$/);
    if (heading) {
      out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      i += 1;
    } else if (l.startsWith("|")) {
      const block = [];
      while (i < lines.length && lines[i].startsWith("|")) block.push(lines[i++]);
      out.push(table(block));
    } else if (/^\s*- /.test(l)) {
      const block = [];
      while (i < lines.length && /^\s*- /.test(lines[i])) block.push(lines[i++]);
      out.push(list(block));
    } else if (!l.trim()) {
      i += 1;
    } else {
      const block = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,6} |\||\s*- )/.test(lines[i])) block.push(lines[i++]);
      out.push(`<p>${inline(block.join(" "))}</p>`);
    }
  }
  return out.join("\n");
}
