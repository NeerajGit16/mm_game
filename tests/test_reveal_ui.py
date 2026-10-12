"""Browser tests for the reveal: a real saved session, served by the
reveal-only server, opened in a real page.

The reveal must render from the coaching record alone -- no game, no world,
no engine -- agree with summary.md to the dollar, choose and jump to its
moments, keep the bots' fills off until asked, explain a fill on hover,
and open a record saved under an older format. Each docstring says what
silent failure it catches.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import urllib.error
import urllib.request

import pytest

from mm_game import agents, engine, game, records, server, world
from mm_game.config import BenchmarkConfig, GameConfig
from mm_game.engine import Settings
from mm_game.storage import DirectoryStorage

from fixtures.make_record_fixture import FIXTURES
from wire import LoopThread

pytest.importorskip("playwright.sync_api")

PRESET = 60


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A records folder holding a real 60-minute session with prints and an
    unannounced move, played by the scripted stand-in to the close, and a
    practice replay of the same world."""
    root = tmp_path_factory.mktemp("records")
    cfg = GameConfig.for_session(PRESET)
    seed = next(s for s in range(1, 200) if (w := world.generate(cfg, s)).path.move_list and w.path.event_steps.size)
    ids = []
    for _ in range(2):
        g = game.Game(cfg, seed, storage=DirectoryStorage(root), settings=Settings(preset=PRESET))
        game.run_headless(g, agents.ScriptedPlayer(cfg, BenchmarkConfig()))
        ids.append(g.session_id)
    return root, ids


def serve(records_dir) -> tuple[LoopThread, int]:
    lt = LoopThread()

    async def start():
        _, port = server.listen_reveal(str(records_dir))
        return port

    return lt, lt.run(start())


def open_reveal(browser, records_dir, session_id):
    lt, port = serve(records_dir)
    page = browser.new_page(viewport={"width": 1920, "height": 1200})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{port}/?reveal={session_id}")
    page.wait_for_function("window.mmReveal", timeout=30_000)
    return page, lt, errors


def load(records_dir, session_id) -> dict:
    return json.loads(DirectoryStorage(records_dir).read_text(f"sessions/{session_id}/coaching_record.json"))


def number(text: str) -> float:
    """$1,234 / −$5 / +$0 / +$-0 / -37 / +178 / 12% as a number."""
    t = text.replace("−", "-").replace("**", "").replace("$", "").replace(",", "").replace("+", "").strip()
    return float(t)


def test_the_reveal_renders_from_a_saved_record_with_no_engine_running(browser, saved, monkeypatch):
    """With the engine, world generation and replay all made to fail, the
    reveal of a saved session still renders in full: score, chart with every
    fill and every informed fill ringed, breakdown, moments, tabs. The same
    patches do stop an old-format record (which is rebuilt by replay), so
    they are live. A reveal that quietly regenerated the world would fail."""
    def refuse(*args, **kwargs):
        raise RuntimeError("the engine must not run for a reveal")

    monkeypatch.setattr(engine.Session, "__init__", refuse)
    monkeypatch.setattr(world, "generate", refuse)
    monkeypatch.setattr(records, "replay", refuse)
    root, ids = saved
    record = load(root, ids[0])
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        rows = page.eval_on_selector_all("#rv-score tbody tr", "rs => rs.map(r => [...r.children].map(c => c.textContent))")
        assert rows[0][0] == "You" and number(rows[0][1]) == round(record["scores"]["pnl_usd"])
        drawn = page.evaluate("window.mmReveal.charts.state.drawn")
        fills = record["reveal"]["fills"]
        assert drawn["you"] == len(fills) > 0
        assert drawn["rings"] == sum(1 for f in fills if f.get("fill_type", "uninformed") != "uninformed")
        assert page.eval_on_selector_all("#rv-moments .moment", "ms => ms.length") == len(record["reveal"]["moments"]) == 3
        assert not errors, errors
        old = FIXTURES / "protocol2-format1-config2"
        lt2, port2 = serve(old)
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"http://127.0.0.1:{port2}/record/30m-3-0001", timeout=30)
        assert refused.value.code == 500  # the old format rebuilds by replay, which the patch forbids
        lt2.close()
    finally:
        page.close()
        lt.close()


def summary_tables(md: str) -> dict:
    """The decomposition and by-type tables of summary.md, by row label and
    column header; and the clients table, by client and column header."""
    lines = md.splitlines()
    out: dict = {"lines": {}, "clients": {}}
    i = 0
    while i < len(lines):
        if lines[i].startswith("| | Mark-out 10s"):
            heads = [c.strip() for c in lines[i].strip("|").split("|")][1:]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out["lines"][cells[0].replace("**", "")] = dict(zip(heads, cells[1:]))
                i += 1
        elif lines[i].startswith("| Client |"):
            heads = [c.strip() for c in lines[i].strip("|").split("|")]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out["clients"][cells[0]] = dict(zip(heads, cells))
                i += 1
        else:
            i += 1
    return out


# The page's clients-tab headers under "Mark-out to next print/move", and
# the summary.md columns that carry the same numbers.
PAGE_TO_SUMMARY = {
    "Informed fills": "Mark-out to next print/move, informed fills",
    "Other fills": "Mark-out to next print/move, other fills",
}
TEXT_COLUMNS = {"Client", "Category", "True type", "Knew about", "Informed RFQs (filled)"}


REVEAL_TO_SUMMARY = {
    "Spread captured": "Spread captured",
    "Adverse selection · uninformed": "Uninformed",
    "Adverse selection · informed, prints": "Informed about a print",
    "Adverse selection · informed, moves": "Informed about a move",
    "Inventory beyond the mark-outs": "Inventory beyond the mark-outs",
    "Hedging and closing": "Hedging and closing",
    "Total": "Total",
}


def page_rows(page, table: str) -> list[dict[str, str]]:
    """A page table's rows, cell by its column header (the last header row)."""
    heads = page.eval_on_selector_all(f"#{table} thead tr:last-child th", "hs => hs.map(h => h.textContent)")
    rows = page.eval_on_selector_all(f"#{table} tbody tr", "rs => rs.map(r => [...r.children].map(c => c.textContent))")
    return [dict(zip(heads, r, strict=True)) for r in rows]


def same_cell(shown: str, want: str, label: str) -> bool:
    if label in TEXT_COLUMNS or want == "—":
        return shown == want
    return number(shown) == number(want)


def test_the_breakdown_and_the_clients_agree_with_summary_md(browser, saved):
    """At each horizon -- 60s, 300s, next print/move -- every breakdown line
    is the number summary.md prints for it, to the dollar; and the Clients
    tab has, client by client and column by column, what summary.md has:
    type, counts, informed RFQs and how many were filled, net lots,
    mark-outs, the next-print/move mark-out split into informed fills and
    other fills, and what each informed client knew. Two renderings of one
    record that disagreed would leave the review arguing about which is
    right."""
    root, ids = saved
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        md = DirectoryStorage(root).read_text(f"sessions/{ids[0]}/summary.md")
        tables = summary_tables(md)
        for horizon, column in (("60s", "Mark-out 60s"), ("300s", "Mark-out 300s"), ("next_event", "Mark-out to next print/move")):
            page.click(f"#rv-horizons [data-horizon='{horizon}']")
            shown = dict(page.eval_on_selector_all("#rv-breakdown tr", "rs => rs.map(r => [r.dataset.line, r.children[1].textContent])"))
            assert set(shown) == set(REVEAL_TO_SUMMARY)
            for label, md_label in REVEAL_TO_SUMMARY.items():
                assert number(shown[label]) == number(tables["lines"][md_label][column]), (horizon, label)
        page.click("#rv-tabs [data-rtab='clients']")
        rows = page_rows(page, "rv-clients-table")
        assert len(rows) == len(tables["clients"]) > 0
        for row in rows:
            want = tables["clients"][row["Client"]]
            compared = 0
            for label, shown in row.items():
                md_label = PAGE_TO_SUMMARY.get(label, label)
                if md_label in want:
                    assert same_cell(shown, want[md_label], label), (row["Client"], label, shown, want[md_label])
                    compared += 1
            assert compared == len(want)  # every summary.md column is on the page
            if row["Informed fills"] != "—" and row["Other fills"] != "—":
                assert number(row["All fills"]) == pytest.approx(number(row["Informed fills"]) + number(row["Other fills"]), abs=1)
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def test_the_moments_jump_the_chart_and_the_toggle_shows_the_bots(browser, saved):
    """Clicking a moment zooms both charts to its window; each move is shaded
    from its leak, not its start, and each run-up up to its print; the
    bots' fills are off until the box is ticked, then every one of the
    disciplined bot's and the oracle's fills is drawn; hovering a fill
    shows its ticket. A move shaded from its start would hide the minutes
    the flow foretold it; a toggle that started on, or drew one bot only,
    would mislead the comparison the chart exists for."""
    root, ids = saved
    record = load(root, ids[0])
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        dt = record["header"]["config"]["dt"]
        for i, m in enumerate(record["reveal"]["moments"]):
            page.click(f"#rv-moments [data-moment='{i}']")
            got = page.evaluate("window.mmReveal.charts.state.range")
            assert got == pytest.approx([m["from_step"] * dt, m["to_step"] * dt], abs=1e-6)
        page.dblclick("#rv-chart .u-over")
        bands = page.evaluate("window.mmReveal.charts.state.bands")
        rv = record["reveal"]
        assert [b for b in bands if b["kind"] == "move"] == [
            {"kind": "move", "from_step": m["inform_step"], "to_step": m["end_step"]} for m in rv["moves"]]
        assert [b for b in bands if b["kind"] == "run-up"] == [
            {"kind": "run-up", "from_step": p["runup"]["start_step"], "to_step": p["step"]} for p in rv["prints"] if p["runup"]]
        assert page.evaluate("window.mmReveal.charts.state.drawn.bots") == 0
        assert page.is_checked("#rv-bots") is False
        page.check("#rv-bots")
        bots = record["reveal"]["bots"]
        assert page.evaluate("window.mmReveal.charts.state.drawn.bots") == len(bots["disciplined"]["fills"]) + len(bots["oracle"]["fills"])
        page.uncheck("#rv-bots")
        fill = next(f for f in record["reveal"]["fills"] if f["source"] == "client")
        at = page.evaluate(f"window.mmReveal.charts.pointOf({fill['step']}, {fill['price']})")
        page.mouse.move(at["x"], at["y"])
        page.wait_for_function("!document.getElementById('rv-tip').hidden")
        assert f"RFQ {fill['rfq_ordinal']}" in page.text_content("#rv-tip")
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def test_the_tabs_show_every_client_every_ticket_and_the_track_record(browser, saved):
    """Clients: one row per client. Tickets: one row per RFQ of the session,
    each saying what the flow oracle did where it saw it. Track record:
    every session, the practice replay shown and marked not counted, the
    pooled figures over scored sessions only. And the replay's own reveal
    carries the practice badge."""
    root, ids = saved
    record = load(root, ids[0])
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        page.click("#rv-tabs [data-rtab='clients']")
        assert page.eval_on_selector_all("#rv-clients-table tbody tr", "r => r.length") == len(record["world"]["clients"])
        page.click("#rv-tabs [data-rtab='tickets']")
        rows = page.eval_on_selector_all("#rv-tickets-table tbody tr", "rs => rs.map(r => [...r.children].map(c => c.textContent))")
        assert len(rows) == len(record["rfqs"])
        shown = {r["ordinal"] for r in record["rfqs"] if r["shown"]}
        assert all(row[-1] != "—" for row in rows if int(row[1]) in shown)
        page.click("#rv-tabs [data-rtab='track']")
        track = page.eval_on_selector_all("#rv-track-table tbody tr", "rs => rs.map(r => [r.dataset.session, r.lastElementChild.textContent])")
        assert [t[0] for t in track] == ids
        assert track[1][1] == "practice replay — not counted" and track[0][1].startswith("scored")
        assert "1 scored session; 1 practice replay shown, not counted" in page.text_content("#rv-track-pooled")
        assert page.is_hidden("#rv-flag")
        assert not errors, errors
    finally:
        page.close()
        lt.close()
    page, lt, errors = open_reveal(browser, root, ids[1])
    try:
        assert page.is_visible("#rv-flag") and "practice replay" in page.text_content("#rv-flag")
        assert f"seed {record['header']['seed']}" in page.text_content("#rv-meta")
    finally:
        page.close()
        lt.close()


def track_table(md: str) -> tuple[dict, dict]:
    """summary.md's track-record table, by row; and its pooled ratios."""
    lines = md.splitlines()
    i = lines.index("| Against the disciplined bot | Pooled | Mean per session | Standard error |") + 2
    rows = {}
    while lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip("|").split("|")]
        rows[cells[0]] = cells[1:]
        i += 1
    ratio = next(re.match(r"Pooled capture ratio: P&L (.*); risk-adjusted (.*)\.$", ln) for ln in lines if ln.startswith("Pooled capture ratio"))
    return rows, {"pnl": ratio.group(1), "risk_adjusted": ratio.group(2)}


def same(page_text: str, md_text: str) -> bool:
    """One figure as the page and summary.md print it: dollars, n/a with
    its reason, or a ratio (0.42 in summary.md, 42% on the page)."""
    p, m = page_text.replace("−", "-"), md_text.replace("−", "-")
    if m.startswith("n/a"):
        return p == m
    if p.endswith("%"):
        return abs(number(p) - float(m) * 100) <= 1  # each rounds its own way
    return number(p) == number(m)


def test_the_track_record_agrees_with_summary_md_and_the_ratio_waits_for_10000(browser, saved, tmp_path):
    """The Track tab's pooled table -- your gap and the oracle's to the
    disciplined bot, pooled, mean per session, standard error -- and its
    pooled ratios are summary.md's, row by row, for the scored session and
    for the practice replay written after it (the replay left out of both,
    so they agree); the summary screen's track line shows the same pooled
    ratio. Then the same folder with a second scored session (so there is a
    standard error to show) and the oracles' pooled edge set to $9,999 and
    to exactly $10,000: n/a with how far it has got, then the ratio, and the
    pooled table still summary.md's. A page that gated on its own, or a
    summary.md that disagreed with it, would fail."""
    root, ids = saved
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        page.click("#rv-tabs [data-rtab='track']")
        shown = dict(page.eval_on_selector_all(
            "#rv-track-gaps tbody tr", "rs => rs.map(r => [r.dataset.row, [...r.children].slice(1).map(c => c.textContent)])"))
        ratios = {k: page.text_content(f"#rv-track-pooled [data-ratio='{k}']") for k in ("pnl", "risk_adjusted")}
        line = page.text_content("#rv-capture-pooled")
        assert not errors, errors
    finally:
        page.close()
        lt.close()
    for sid in ids:
        rows, md_ratios = track_table(DirectoryStorage(root).read_text(f"sessions/{sid}/summary.md"))
        assert set(shown) == set(rows) and len(rows) == 4
        for name, cells in rows.items():
            assert all(same(p, m) for p, m in zip(shown[name], cells, strict=True)), (sid, name, shown[name], cells)
        assert all(same(ratios[k], md_ratios[k]) for k in ratios), (ratios, md_ratios)
    assert line == ratios["pnl"]

    # A second scored session, so there is a standard error to show.
    folder = tmp_path / "records"
    shutil.copytree(root, folder)
    track = records.TrackRecord.load(DirectoryStorage(folder))
    scored = track.entries[0]
    extra = {**scored, "session_id": "extra", "pnl_usd": scored["pnl_usd"] + 3_000.0,
             "risk_adjusted_usd": scored["risk_adjusted_usd"] - 2_000.0,
             "disciplined_pnl_usd": 0.0, "oracle_pnl_usd": 4_000.0}
    track.entries.append(extra)
    you = scored["pnl_usd"] + extra["pnl_usd"]
    for oracle, want in ((5_999.0, "n/a — not enough flow-reading edge yet ($9,999 of $10,000)"),
                         (6_000.0, f"{math.floor(you / 10_000 * 100 + 0.5)}%")):  # Math.round
        # Disciplined at zero so the pooled gap is exactly the oracles' scores: 4,000 + this.
        scored.update(disciplined_pnl_usd=0.0, oracle_pnl_usd=oracle)
        track.save(DirectoryStorage(folder))
        rows, _ = track_table("\n".join(records.track_lines(track.aggregates(), counted=True)))
        page, lt, errors = open_reveal(browser, folder, ids[0])
        try:
            assert page.text_content("#rv-capture-pooled") == want
            page.click("#rv-tabs [data-rtab='track']")
            assert page.text_content("#rv-track-pooled [data-ratio='pnl']") == want
            shown = dict(page.eval_on_selector_all(
                "#rv-track-gaps tbody tr", "rs => rs.map(r => [r.dataset.row, [...r.children].slice(1).map(c => c.textContent)])"))
            assert all(cells[2] != "n/a" for cells in rows.values())
            for name, cells in rows.items():
                assert all(same(p, m) for p, m in zip(shown[name], cells, strict=True)), (name, shown[name], cells)
            assert not errors, errors
        finally:
            page.close()
            lt.close()


@pytest.mark.parametrize("name,version", [("protocol2-format1-config2", 1), ("protocol2-replay1-record2-track1-config2", 2)])
def test_an_old_format_record_opens_in_the_reveal(browser, name, version):
    """A session saved under record format 1 (before the reveal block) or 2
    (before the clients' split by ticket) opens in the reveal, rebuilt in
    today's layout, with the score it was saved with and the split filled
    in -- and its summary.md is left exactly as it was written."""
    old = FIXTURES / name
    storage = DirectoryStorage(old)
    saved_record = json.loads(storage.read_text("sessions/30m-3-0001/coaching_record.json"))
    summary = (old / "sessions" / "30m-3-0001" / "summary.md").read_bytes()
    assert saved_record["format_version"] == version
    page, lt, errors = open_reveal(browser, old, "30m-3-0001")
    try:
        assert page.evaluate("window.mmReveal.record.format_version") == records.RECORD_FORMAT_VERSION
        you = page.eval_on_selector("#rv-score tbody tr", "r => r.children[1].textContent")
        assert number(you) == round(saved_record["scores"]["pnl_usd"])
        assert page.evaluate("window.mmReveal.charts.state.drawn.you") > 0
        page.click("#rv-tabs [data-rtab='clients']")
        assert all(r["Informed RFQs (filled)"] for r in page_rows(page, "rv-clients-table"))
        assert not errors, errors
    finally:
        page.close()
        lt.close()
    assert (old / "sessions" / "30m-3-0001" / "summary.md").read_bytes() == summary


OVERFLOW = """() => {
  const bad = [];
  for (const el of document.querySelectorAll('#rv-summary *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'inline' || cs.display === 'none' || el.closest('.uplot') || el.tagName === 'CANVAS') continue;
    if (el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1)
      bad.push(`${el.tagName}#${el.id}.${el.className}: ${el.scrollWidth}x${el.scrollHeight} in ${el.clientWidth}x${el.clientHeight}`);
  }
  const d = document.documentElement;
  if (d.scrollWidth > d.clientWidth) bad.push(`the page: ${d.scrollWidth} wide in ${d.clientWidth}`);
  return bad;
}"""


@pytest.mark.parametrize("width,height", [(1920, 1080), (1440, 900)])
def test_the_summary_fits_with_nothing_clipped_or_overflowing(browser, saved, width, height):
    """At 1920x1080 and 1440x900 no element of the summary has more content
    than room -- scroll size never exceeds client size, so no text is
    clipped, spills, or hides behind a scroll bar inside a card -- and the
    page never scrolls sideways. The left column (reveal chart, position
    chart at least half its height, Clients revealed with five rows) fits
    the window under the tabs and stays put while the page scrolls; the
    benchmarks' rules start collapsed. A layout that squeezed the cards
    into scrolling boxes, as the first one did, fails here."""
    root, ids = saved
    record = load(root, ids[0])
    lt, port = serve(root)
    page = browser.new_page(viewport={"width": width, "height": height})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(f"http://127.0.0.1:{port}/?reveal={ids[0]}")
        page.wait_for_function("window.mmReveal", timeout=30_000)
        page.wait_for_timeout(300)
        assert page.evaluate(OVERFLOW) == []
        box = page.evaluate("(() => { const r = document.getElementById('rv-left').getBoundingClientRect(); return {top: r.top, bottom: r.bottom}; })()")
        assert box["bottom"] <= height
        h_chart, h_pos = page.evaluate("[document.getElementById('rv-chart').offsetHeight, document.getElementById('rv-pos').offsetHeight]")
        assert h_pos * 2 >= h_chart >= 200
        traded = sum(1 for c in record["diagnostics"]["clients"].values() if c["won"])
        assert page.eval_on_selector_all("#rv-clients-top tbody tr", "r => r.length") == min(5, traded) == 5
        assert page.evaluate("document.getElementById('rv-rules-box').open") is False
        page.click("#rv-rules-box summary")  # the rules open: the right column outgrows the window
        page.mouse.move(width * 0.8, height / 2)
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(200)
        assert page.evaluate("window.scrollY") > 0  # the page scrolls as a whole...
        left = page.evaluate("(() => { const r = document.getElementById('rv-left').getBoundingClientRect(); return {top: r.top, bottom: r.bottom}; })()")
        assert 0 <= left["top"] < box["top"] and left["bottom"] <= height  # ...and the left column stays in view
        assert page.evaluate(OVERFLOW) == []
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def test_the_position_chart_is_labelled_fitted_and_its_lines_distinct(browser, saved):
    """The position chart's y axis says what it is in, fits the three
    positions (zero in, little room either side) rather than mirroring the
    largest, and draws you, the disciplined bot and the oracle with
    different colours and different dashes. A symmetric scale around the
    largest position flattened the other two into the axis."""
    root, ids = saved
    rv = load(root, ids[0])["reveal"]
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        held = [p for w in ("you", "disciplined", "oracle") for p in rv["positions"][w]["positions"]]
        lo, hi = min(0, *held), max(0, *held)
        got = page.evaluate("window.mmReveal.charts.posState")
        assert got["axisLabel"] == "lots (+ long)"
        assert got["range"][0] <= lo and got["range"][1] >= hi
        assert got["range"][1] - got["range"][0] <= (hi - lo) * 1.2 + 10
        lines = got["lines"]
        assert len({l["stroke"] for l in lines}) == 3 and len({l["dash"] for l in lines}) == 3
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def test_a_moment_marks_both_charts_and_clicking_it_again_clears_it(browser, saved):
    """Clicking a moment draws its labelled line on both charts at its time
    -- the print, a move's ramp with its leak window shaded, a ticket with
    that fill's marker highlighted -- zooms to it and highlights its card;
    clicking another moves the mark (one at a time); clicking the active
    one again removes the mark and resets the zoom to the whole session. A
    mark left behind, two cards lit at once, or a zoom that stuck would
    misread the next moment."""
    root, ids = saved
    rv = load(root, ids[0])["reveal"]
    dt = load(root, ids[0])["header"]["config"]["dt"]
    page, lt, errors = open_reveal(browser, root, ids[0])
    state = "window.mmReveal.charts.state"
    active = "[...document.querySelectorAll('#rv-moments .moment.active')].map(m => Number(m.dataset.moment))"
    try:
        full = page.evaluate("window.mmReveal.charts.full")
        for i, m in enumerate(rv["moments"]):
            page.click(f"#rv-moments [data-moment='{i}']")
            at = {"print": lambda: rv["prints"][m["index"]]["step"],
                  "move": lambda: next(v for v in rv["moves"] if v["index"] == m["index"])["start_step"],
                  "ticket": lambda: m["step"]}[m["kind"]]()
            s = page.evaluate(state)
            assert page.evaluate(active) == [i]
            assert s["marker"]["kind"] == m["kind"] and s["markerDrawn"] == {"price": at, "position": at}
            assert s["range"] == pytest.approx([m["from_step"] * dt, m["to_step"] * dt], abs=1e-6)
            assert s["drawn"]["highlighted"] == (1 if m["kind"] == "ticket" else 0)
        last = len(rv["moments"]) - 1
        page.click(f"#rv-moments [data-moment='{last}']")
        s = page.evaluate(state)
        assert page.evaluate(active) == [] and s["marker"] is None
        assert s["markerDrawn"] == {"price": None, "position": None} and s["drawn"]["highlighted"] == 0
        assert s["range"] == pytest.approx(full, abs=1e-6)
        # A move: the leak window shaded and the line at the ramp.
        if rv["moves"]:
            mv = rv["moves"][0]
            page.evaluate(f"window.mmReveal.charts.mark({{kind: 'move', index: {mv['index']}, step: {mv['start_step']}}})")
            s = page.evaluate(state)
            assert s["marker"]["leak"] == [mv["inform_step"], mv["start_step"]]
            assert s["markerDrawn"] == {"price": mv["start_step"], "position": mv["start_step"]}
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def headers_of(page, scope: str) -> list[tuple[str, list[str]]]:
    """Every table (or tile row) under ``scope`` with a glossary key: its key
    and the labels a reader sees -- column headers, and the row labels of a
    table read by rows."""
    return page.evaluate(f"""[...document.querySelectorAll('{scope} [data-glossary]')].map(t => [
        t.dataset.glossary,
        [...t.querySelectorAll('thead th, .tile .lbl, tr[data-line] td:first-child')]
          .map(h => h.textContent.trim()).filter(Boolean)
          .concat(t.id === 'rv-score' ? [...t.querySelectorAll('tbody td:first-child')].map(c => c.textContent.trim()) : [])
      ])""")


def test_every_column_header_on_the_reveal_has_a_definition(browser, saved):
    """Every table on the reveal -- the summary's panels and the Clients,
    Tickets and Track record tabs -- has an (i), and every header on it (and
    every breakdown line and score row) is defined in that table's
    glossary; opening the (i) lists exactly those definitions. A column
    added without one, or a header renamed on the page only, fails."""
    root, ids = saved
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        tables = page.evaluate("window.mmGlossary.tables()")
        found = headers_of(page, "#reveal")
        keys = {k for k, _ in found}
        assert {"reveal.clients_top", "reveal.score", "reveal.breakdown", "reveal.clients", "reveal.tickets",
                "reveal.track", "reveal.track_gaps"} <= keys
        for key, labels in found:
            defined = {e["label"] for e in tables[key]["entries"]}
            assert labels and set(labels) <= defined, (key, set(labels) - defined)
        infos = page.eval_on_selector_all("#reveal [data-info]", "bs => bs.map(b => b.dataset.info)")
        assert keys <= set(infos) and set(infos) <= set(tables)
        for key in infos:
            page.evaluate(f"document.querySelector('#reveal [data-info=\"{key}\"]').click()")
            page.wait_for_function("!document.getElementById('glossary').hidden")
            shown = page.eval_on_selector_all("#gl-list dt", "ds => ds.map(d => d.textContent)")
            assert shown == [e["label"] for e in tables[key]["entries"]], key
            assert page.text_content("#gl-title") == tables[key]["title"]
            page.keyboard.press("Escape")
            page.wait_for_function("document.getElementById('glossary').hidden")
        assert not errors, errors
    finally:
        page.close()
        lt.close()


def test_no_text_on_the_reveal_says_venue(browser, saved):
    """Every tab of the reveal, and every (i) panel on it, says broker. The
    record's field names keep "venue" (old records must load); the words
    the player reads do not."""
    root, ids = saved
    page, lt, errors = open_reveal(browser, root, ids[0])
    try:
        texts = []
        for tab in ("summary", "clients", "tickets", "track"):
            page.click(f"#rv-tabs [data-rtab='{tab}']")
            texts.append(page.inner_text("#reveal"))
        page.click("#rv-tabs [data-rtab='summary']")
        for key in page.eval_on_selector_all("#reveal [data-info]", "bs => bs.map(b => b.dataset.info)"):
            page.evaluate(f"document.querySelector('#reveal [data-info=\"{key}\"]').click()")
            texts.append(page.inner_text("#glossary"))
            page.keyboard.press("Escape")
        assert [t for t in texts if "venue" in t.lower()] == []
        assert any("broker" in t.lower() for t in texts)
        assert not errors, errors
    finally:
        page.close()
        lt.close()
