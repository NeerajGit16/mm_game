"""Tests for what the columns mean, and for the words the player reads.

The definitions live in one place, ``glossary.py``, and both the page's (i)
panels and summary.md show them. What can go wrong silently: a column added
to a table with no definition, the page and summary.md explaining the same
column two ways, and a word the game has retired ("venue") left on a screen.
The page-side checks are in the browser tests; these cover summary.md, the
served definitions, and every player-facing string the Python side writes.
Each docstring says what it catches.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from mm_game import agents, game, glossary, handbook, records, server
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Settings
from mm_game.storage import MemoryStorage

from wire import LoopThread


@pytest.fixture(scope="module")
def session():
    """A 60-minute session with prints and a move, played to the close and
    written with its track record, so summary.md has every table."""
    cfg = GameConfig.for_session(60)
    g = game.Game(cfg, 12, storage=MemoryStorage(), settings=Settings(preset=60))
    game.run_headless(g, agents.ScriptedPlayer(cfg, BenchmarkConfig()))
    return g


def md_tables(md: str) -> list[list[list[str]]]:
    """Every table in a markdown document, as rows of cells (the divider dropped)."""
    tables, current = [], []
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= {"-"} for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def test_every_summary_md_header_has_a_definition(session):
    """Each table summary.md prints -- score, track record, where the P&L
    came from, adverse selection by type, clients -- in that order, has
    every column header (and, for a table read by rows, every row label)
    defined in its glossary table; and the closing "What the columns mean"
    lists exactly those definitions. A column added to the summary without
    a definition, or a definition that no longer matches its header, fails."""
    md = session.summary
    tables = md_tables(md)
    defs = records.glossary_tables(session.record["header"]["config"], session.record["capture_ratio"]["min_gap_usd"])
    assert len(tables) == len(glossary.SUMMARY_TABLES)
    for table, key in zip(tables, glossary.SUMMARY_TABLES):
        labels = set(defs[key].labels())
        heads = [h for h in table[0] if h]
        assert heads and set(heads) <= labels, (key, set(heads) - labels)
        if not table[0][0]:  # read by rows: the row labels are defined too
            rows = {r[0] for r in table[1:]}
            assert rows <= labels, (key, rows - labels)
    section = md[md.index("## What the columns mean"):]
    for key in glossary.SUMMARY_TABLES:
        for e in defs[key].entries:
            assert f"- **{glossary.md_escape(e.label)}**: {e.meaning}" in section, (key, e.label)


def test_the_page_is_served_the_same_definitions_summary_md_prints():
    """The page's (i) panels read /glossary; it serves the very tables
    summary.md is written from, for today's config and thresholds. Two
    copies of a definition would drift the first time one was edited."""
    lt = LoopThread()
    try:
        async def start():
            _, port = server.listen_reveal(".")
            return port

        port = lt.run(start())
        import urllib.request

        served = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/glossary", timeout=10).read())
    finally:
        lt.close()
    expected = glossary.message(records.glossary_tables(GameConfig().to_record(), ScoringConfig().capture_ratio_min_usd))
    assert served == json.loads(json.dumps(expected))
    track = served["tables"]["reveal.track"]
    assert "$10,000" in track["intro"]
    assert "$2,000" in next(e["meaning"] for e in track["entries"] if e["label"] == "Capture")


def test_the_intended_meanings_are_the_ones_written():
    """The meanings asked for, word for word where it matters: lots
    traded gross is buys plus sells; your net is + when you bought; a
    mark-out is size times how far the truth moved, edge excluded, + your
    way; the next-print/move mark-out runs to whichever comes first; the
    play screen's mark-outs say they are against the screen."""
    t = glossary.tables(
        horizons_s=(10.0, 60.0, 300.0), per_session_min_usd=2000.0, pooled_min_usd=10_000.0, tick_bp=0.25, win_half_bp=0.5
    )
    by = {e.label: e.meaning for e in t["reveal.clients"].entries}
    assert "buys plus your sells" in by["Lots traded (gross)"]
    assert "+ when on balance you bought" in by["Your net (+ = you bought)"]
    for h in ("60s", "300s"):
        m = by[f"Mark-out {h}"]
        assert "size times how far the true value moved" in m and "edge you charged is left out" in m and "+ means" in m
    assert "whichever comes first" in by["Mark-out to next print/move"]
    play = {e.label: e.meaning for e in t["play.clients"].entries}
    assert "Against the screen" in play["Mark-out 60s / lot"]
    track = t["reveal.track_gaps"]
    assert "$10,000" in track.intro and "(n − 1)" in {e.label: e.meaning for e in track.entries}["Standard error"]


# --------------------------------------------------------------------------
# "Broker", never "venue", in anything the player reads
# --------------------------------------------------------------------------

VENUE = re.compile(r"venue", re.IGNORECASE)


def test_no_text_the_python_side_writes_for_the_player_says_venue(session):
    """The handbook (all three parts), the bots' rules, every definition,
    summary.md, and the reveal's moments call it the broker. Code names and
    record fields keep "venue" -- saved records must keep loading -- so the
    check is on the words, not the keys."""
    cfg = GameConfig.for_session(60)
    texts = []
    for part in handbook.message(cfg, ScoringConfig(), scored_sessions=handbook.UNLOCK_AFTER_SCORED_SESSIONS)["parts"]:
        texts += [part["title"], part["subtitle"], part.get("note") or ""]
        texts += [f"{i['head'] or ''} {i['text']}" for i in part["items"]]
    rules = session.record["reveal"]["rules"]
    texts += [f"{i['head'] or ''} {i['text']}" for i in rules["disciplined"] + rules["oracle"]]
    tables = records.glossary_tables(cfg.to_record(), 2000.0)
    texts += [f"{t.title} {t.intro} " + " ".join(f"{e.label} {e.meaning}" for e in t.entries) for t in tables.values()]
    texts.append(session.summary)
    texts += [f"{m['title']} {m['sentence']}" for m in session.record["reveal"]["moments"]]
    found = [t for t in texts if VENUE.search(t)]
    assert not found, found


class _Visible(HTMLParser):
    """The text and the human-read attributes of a page, not its ids or classes."""

    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.text += [v for k, v in attrs if k in ("title", "alt", "placeholder", "aria-label") and v]

    def handle_data(self, data):
        self.text.append(data)


def js_strings(source: str) -> list[str]:
    """The string literals of a JavaScript module -- quoted strings and the
    literal parts of template strings -- with comments and code dropped."""
    out, i, n = [], 0, len(source)
    depth: list[int] = []  # brace depth inside each open ${...}
    last = ""  # the last code character, to tell a regex from a division
    while i < n:
        c = source[i]
        if source.startswith("//", i):
            i = source.find("\n", i) if source.find("\n", i) != -1 else n
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 2
        elif c == "/" and (last == "" or last in "(,=:[!&|?{};+-*%<>~^" or source[max(0, i - 7):i].rstrip().endswith("return")):
            j, in_class = i + 1, False  # a regex literal: skip it, quotes and all
            while source[j] != "/" or in_class:
                in_class = (in_class or source[j] == "[") and source[j] != "]"
                j += 2 if source[j] == "\\" else 1
            i, last = j + 1, "/"
        elif c in "'\"":
            j = i + 1
            while source[j] != c:
                j += 2 if source[j] == "\\" else 1
            out.append(source[i + 1 : j])
            i, last = j + 1, c
        elif c == "`" or (c == "}" and depth and depth[-1] == 0):
            if c == "}":
                depth.pop()
            j, buf = i + 1, []
            while j < n and source[j] != "`" and not source.startswith("${", j):
                buf.append(source[j : j + 2] if source[j] == "\\" else source[j])
                j += 2 if source[j] == "\\" else 1
            out.append("".join(buf))
            if source.startswith("${", j):
                depth.append(0)
                i, last = j + 2, "{"
            else:
                i, last = j + 1, "`"
        else:
            if depth:
                depth[-1] += c == "{"
                depth[-1] -= c == "}"
            if not c.isspace():
                last = c
            i += 1
    return out


def test_no_text_on_the_page_says_venue():
    """index.html's text and titles, and every string literal in the page's
    modules, say broker: the play screen, the risk strip, the P&L tiles,
    the reveal. Identifiers that read the protocol's ``venue`` fields stay,
    so only words are checked."""
    web = Path(server.WEB_DIR)
    page = _Visible()
    page.feed((web / "index.html").read_text(encoding="utf-8"))
    found = [t for t in page.text if VENUE.search(t)]
    for js in sorted((web / "js").glob("*.js")):
        strings = js_strings(js.read_text(encoding="utf-8"))
        assert strings, js.name  # the tokenizer found the module's strings
        found += [f"{js.name}: {s}" for s in strings if VENUE.search(s)]
    assert not found, found
    assert any("broker" in t.lower() for t in page.text)


def test_the_string_scan_sees_strings_and_not_code():
    """The scan reads string literals and template text, not identifiers or
    comments: a check that also matched ``st.venue`` would forbid the
    protocol's field names, one that missed template text would pass a
    "venue" written into a template."""
    src = ('const a = st.venue.quotes / 2; // the venue\nset("x", `at the ${st.venue.cost} venue`); /* venue */ f(\'one\');'
           '\nconst r = s.replace(/[\'"]/g, "&#39;");')
    assert js_strings(src) == ["x", "at the ", " venue", "one", "&#39;"]
