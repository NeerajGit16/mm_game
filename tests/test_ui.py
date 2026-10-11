"""Browser tests: real key presses in a real page, against a real server.

The server runs on a background thread with the wall clock sped up thirty
times, exactly as in play otherwise; the page is the shipped front end in
headless Edge (or Chrome). What a key did is read from the engine's own
log on the server, not from the page, so a page that drew the right thing
but sent the wrong message would still fail. Each docstring says what
silent failure it catches.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from mm_game import host as host_module, records, server
from wire import LoopThread, serve

sync_api = pytest.importorskip("playwright.sync_api")

FAST = 30.0  # game seconds per wall second at 1x
PRESET, SEED = 60, 7
TICK = 0.25


def launch(p):
    for kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kwargs)
        except Exception:
            continue
    pytest.skip("no browser for Playwright: install Edge or Chrome, or run `playwright install chromium`")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = launch(p)
        yield b
        b.close()


@dataclass
class Play:
    page: object
    host: host_module.Host
    lt: LoopThread

    def events(self, kind: str) -> list:
        return self.lt.call(lambda: [e for e in self.host.session.log if e.kind.value == kind])

    def js(self, expr: str):
        return self.page.evaluate(expr)

    def wait(self, expr: str, timeout: float = 20_000) -> None:
        self.page.wait_for_function(expr, timeout=timeout)

    def wait_events(self, kind: str, count: int) -> list:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            found = self.events(kind)
            if len(found) >= count:
                return found
            time.sleep(0.02)
        raise AssertionError(f"fewer than {count} {kind} events")

    def press(self, *keys: str) -> None:
        for k in keys:
            self.page.keyboard.press(k)


@pytest.fixture
def play(browser, tmp_path, request):
    seed = getattr(request, "param", SEED)
    lt = LoopThread()
    host = lt.call(lambda: server.new_host(PRESET, seed, records_dir=str(tmp_path), now=lambda: FAST * time.monotonic()))
    port = serve(lt, host, tick_ms=server.TICK_MS)
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{port}/")
    page.wait_for_function("window.mm && window.mm.store.state")
    yield Play(page, host, lt)
    page.close()
    lt.close()
    assert not errors, errors


def next_ticket(play: Play, after: int) -> dict:
    play.wait(f"(s => s && s.ticket && s.ticket.ordinal > {after})(window.mm.store.state)")
    return play.js("({ticket: window.mm.store.state.ticket, mid: window.mm.store.state.screen.mid, step: window.mm.store.state.step})")


def sent_quote(play: Play, ordinal: int) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        found = [e.data for e in play.events("quote_sent") if e.data["rfq_ordinal"] == ordinal]
        if found:
            return found[0]
        time.sleep(0.02)
    raise AssertionError(f"no quote sent for RFQ {ordinal}")


def expected(t: dict, skew: int, half: int) -> tuple:
    bid = t["mid"] + (skew - half) * TICK
    ask = t["mid"] + (skew + half) * TICK
    if t["ticket"]["one_way"]:
        bid, ask = (None, ask) if t["ticket"]["side"] == 1 else (bid, None)
    return bid, ask


def test_the_default_quote_goes_out_with_one_keystroke_and_skew_and_width_carry_over(play):
    """The first ticket: Enter alone sends screen ± 2 ticks. The second:
    ← ← ↑ then Enter sends skew −2, half-width 3. The third: Enter alone
    sends skew −2, half-width 3 again around its own screen -- the last
    ticket's market, not a default and not anything computed from the
    position. One-way tickets send only the client's side. Checked against
    the quote the engine logged, and the screen the page priced off is the
    public screen at the RFQ's step."""
    view = play.host.game._world.path.public
    play.press("Space")
    t1 = next_ticket(play, -1)
    assert t1["mid"] == view.screen_at(t1["step"])
    play.press("Enter")
    q1 = sent_quote(play, t1["ticket"]["ordinal"])
    assert (q1["requested_bid"], q1["requested_ask"]) == expected(t1, 0, 2)

    t2 = next_ticket(play, t1["ticket"]["ordinal"])
    play.press("ArrowLeft", "ArrowLeft", "ArrowUp", "Enter")
    q2 = sent_quote(play, t2["ticket"]["ordinal"])
    assert (q2["requested_bid"], q2["requested_ask"]) == expected(t2, -2, 3)

    t3 = next_ticket(play, t2["ticket"]["ordinal"])
    play.press("Enter")
    q3 = sent_quote(play, t3["ticket"]["ordinal"])
    assert (q3["requested_bid"], q3["requested_ask"]) == expected(t3, -2, 3)
    assert play.js("document.getElementById('q-skew').textContent") == "−2 ticks"


@pytest.mark.parametrize("play", [21], indirect=True)  # one-way sells at RFQ 0, one-way buys at RFQ 5
def test_a_one_way_ticket_sends_only_the_side_the_client_asked_for(play):
    """Enter on a one-way ticket sends only the client's side -- the ask
    when the client buys, the bid when they sell -- and the ticket shows
    the other side as not asked; two-way tickets in between are passed.
    Sending both would show the client a price they never asked for, and
    the wrong one would be a price nobody can trade."""
    play.press("Space")
    last, seen = -1, set()
    while seen != {1, -1}:
        t = next_ticket(play, last)
        last = t["ticket"]["ordinal"]
        if not t["ticket"]["one_way"]:
            play.press("p")
            continue
        side = t["ticket"]["side"]
        not_asked = "q-bid-pos" if side == 1 else "q-ask-pos"
        assert play.js(f"document.getElementById('{not_asked}').textContent") == "not asked"
        play.press("Enter")
        q = sent_quote(play, last)
        assert (q["requested_bid"], q["requested_ask"]) == expected(t, 0, 2)
        assert (q["requested_bid"] is None, q["requested_ask"] is None) == ((True, False) if side == 1 else (False, True))
        seen.add(side)


def test_digits_reach_the_size_box_only_while_it_is_open(play):
    """Outside the box a digit is a session key (4: speed). H opens the box;
    then digits are the size -- 2 does not change the speed -- and S sells
    the typed size, not 25. Esc closes it with nothing traded, and digits
    are session keys again. A box that leaked keys would change the speed or
    sell 25 lots in the middle of typing a size."""
    play.press("4")
    play.wait("window.mm.store.state.clock.speed === 4")
    play.press("h")
    play.wait("!document.getElementById('sizebox').hidden")
    play.page.keyboard.type("372")
    play.press("Backspace")
    assert play.js("document.getElementById('sizebox-value').textContent") == "37"
    play.press("s")
    hedges = play.wait_events("hedge_executed", 1)
    assert [h.data["requested"] for h in hedges] == [-37]
    play.wait("window.mm.store.state.book.position === -37")
    assert play.js("document.getElementById('sizebox').hidden")
    assert [e.data["speed"] for e in play.events("speed_changed")] == [4]  # the 2 typed in the box went nowhere

    play.press("h")
    play.page.keyboard.type("12")
    play.press("Escape")
    play.wait("document.getElementById('sizebox').hidden")
    play.press("1")
    play.wait("window.mm.store.state.clock.speed === 1")
    assert [e.data["speed"] for e in play.events("speed_changed")] == [4, 1]
    assert len(play.events("hedge_executed")) == 1


def test_the_position_limit_clips_a_hedge_and_the_clip_is_shown(play):
    """At +190, buying 50 fills 10 and the page says it was clipped, asked
    against filled; at +200 the next buy is refused and says so. A clip
    that happened silently would leave the player believing they had
    hedged 50."""
    play.press("h")
    play.page.keyboard.type("190")
    play.press("b")
    play.wait("window.mm.store.state.book.position === 190")
    play.press("e")
    hedges = play.wait_events("hedge_executed", 2)
    assert (hedges[1].data["requested"], hedges[1].data["quantity"], hedges[1].data["clipped"]) == (50, 10, True)
    play.wait("document.getElementById('hedge-notice').textContent.includes('clipped')")
    notice = play.js("document.getElementById('hedge-notice').textContent")
    assert "asked +50, filled +10" in notice
    assert play.js("document.getElementById('rs-pos').textContent") == "+200 lots"
    play.press("e")
    play.wait_events("hedge_executed", 3)
    play.wait("document.getElementById('hedge-notice').textContent.includes('refused')")


def test_hedge_keys_trade_their_sizes_and_f_twice_flattens(play):
    """Q W E buy 10 25 50; A S D sell them. F once only warns; F twice
    within a second flattens; two presses further apart do not."""
    sizes = []
    for key, q in zip("qweasd", (10, 25, 50, -10, -25, -50)):
        play.press(key)
        sizes.append(q)
        play.wait_events("hedge_executed", len(sizes))
    assert [h.data["requested"] for h in play.events("hedge_executed")] == sizes
    play.press("w")
    play.wait("window.mm.store.state.book.position === 25")
    play.press("f")
    play.wait("!document.getElementById('hint').hidden")
    time.sleep(1.3)
    play.press("f")  # too late to pair with the first: a fresh first press
    time.sleep(0.2)
    assert play.js("window.mm.store.state.book.position") == 25
    play.press("f")
    play.wait("window.mm.store.state.book.position === 0")
    assert play.events("hedge_executed")[-1].data["requested"] == -25


def test_the_end_screen_shows_the_summary(play):
    """Quit (confirmed) ends the session; the end screen shows summary.md
    as written for the coaching record, and where the record was saved."""
    play.page.on("dialog", lambda d: d.accept())
    play.press("Space")
    play.page.click("#quit-btn")
    play.wait("!document.getElementById('endscreen').hidden")
    summary = records.markdown_summary(play.host.game.record)
    title = summary.splitlines()[0].removeprefix("# ")
    assert play.js("document.querySelector('#end-summary h1').textContent") == title
    tables = play.js("document.querySelectorAll('#end-summary table').length")
    assert tables == sum(1 for i, l in enumerate(summary.splitlines()) if l.startswith("|---"))
    assert play.js("document.getElementById('end-saved').textContent").endswith(f"sessions/{play.host.game.session_id}/")


# --------------------------------------------------------------------------
# The handbook and the other tabs
# --------------------------------------------------------------------------


def play_some(play: Play, tickets: int, hedge_every: int = 3) -> None:
    """Quote a few tickets with Enter, hedging now and then, then pause."""
    play.press("Space")
    last = -1
    for i in range(tickets):
        t = next_ticket(play, last)
        last = t["ticket"]["ordinal"]
        play.press("Enter")
        if i % hedge_every == 1:
            play.press("w")
    play.wait(f"!window.mm.store.state.ticket || window.mm.store.state.ticket.ordinal > {last}")
    play.press("Space")
    play.wait("window.mm.store.state.clock.paused")


def text(play: Play, selector: str) -> str:
    return play.js(f"document.querySelector({selector!r}).textContent")


def test_the_handbook_is_on_the_start_screen_and_question_mark_pauses_the_clock(play):
    """Before the session starts, the start screen shows the handbook below
    the keys -- your desk, the market, and Part 3 locked with its count.
    Once playing, ? opens it and pauses the clock (the engine logs the
    pause); while it is open no key trades; ? or Esc closes it and the clock
    stays paused."""
    start = text(play, "#start-handbook")
    assert "Your desk" in start and "The market" in start
    assert "Unlocks after 10 scored sessions (0 so far)." in start
    assert "an hour of time clear of prints" not in start  # Part 3 is not on the page at all
    play.press("Space")
    play.wait("window.mm.store.begun")
    play.press("?")
    play.wait("!document.getElementById('handbookscreen').hidden")
    play.wait_events("paused", 1)
    play.wait("window.mm.store.state.clock.paused")
    play.press("w")  # swallowed while the handbook is open
    play.press("Escape")
    play.wait("document.getElementById('handbookscreen').hidden")
    assert play.events("hedge_executed") == []
    assert play.js("window.mm.store.state.clock.paused") is True
    play.press("?")
    play.wait("!document.getElementById('handbookscreen').hidden")
    play.press("?")
    play.wait("document.getElementById('handbookscreen').hidden")
    assert len(play.events("paused")) == 1  # already paused: not paused twice


def test_the_pnl_tab_lists_every_trade_and_its_footer_is_the_top_bar_pnl(play):
    """Every trade, hedges included, is a row; the footer's total, the P&L
    tile and the top bar show the same P&L, and the tiles add up to it.
    Clicking Edge sorts the rows by edge, biggest first. A row the table
    dropped would leave the footer short of the top bar."""
    play_some(play, 6)
    play.press("Tab")
    play.wait("!document.getElementById('tab-pnl').hidden")
    trades = play.lt.call(lambda: list(play.host.session.book.trades))
    play.wait(f"document.querySelectorAll('#pnl-table tbody tr').length === {len(trades)}")
    assert any(t.source.value == "hedge" for t in trades)
    top = text(play, "#tb-pnl")
    assert text(play, "#pnl-foot-total") == top == text(play, "#tile-total")
    tiles = play.js("window.mm.store.desk.pnl.tiles")
    assert tiles["edge_client_usd"] + tiles["venue_charge_usd"] + tiles["move_usd"] == pytest.approx(
        play.js("window.mm.store.state.book.total_usd"), abs=0.005)
    play.page.click("#pnl-table th[data-sort='edge_usd']")
    play.wait("document.querySelector('#pnl-table th[data-sort=edge_usd]').classList.contains('sorted')")
    edge = {r["seq"]: r["edge_usd"] for r in play.js("window.mm.store.desk.pnl.rows")}
    play.wait(f"Number(document.querySelector('#pnl-table tbody tr').dataset.seq) === {max(edge, key=lambda k: (edge[k], -k))}")
    order = play.js("[...document.querySelectorAll('#pnl-table tbody tr')].map(tr => Number(tr.dataset.seq))")
    shown = [edge[s] for s in order]
    assert shown == sorted(shown, reverse=True) and len(order) == len(trades)


def test_the_clients_tab_lists_every_client_with_every_desk_column_sortably(play):
    """The full roster: one row per client on the roster, thirteen columns,
    the ticket's client highlighted; clicking RFQs sorts by it, most first,
    and again reverses. A roster that kept only clients you had traded with
    would hide the ones asking and never trading."""
    play_some(play, 5)
    for _ in range(2):
        play.press("Tab")
    play.wait("!document.getElementById('tab-clients').hidden")
    roster = play.js("window.mm.store.roster.length")
    play.wait(f"document.querySelectorAll('#clients-table tbody tr').length === {roster}")
    assert play.js("document.querySelectorAll('#clients-table thead th').length") == 13
    rfqs = {c["client_id"]: c["rfqs"] for c in play.js("window.mm.store.desk.clients")}
    order_js = "[...document.querySelectorAll('#clients-table tbody tr')].map(tr => Number(tr.dataset.client))"
    for want in (sorted(rfqs.values(), reverse=True), sorted(rfqs.values())):
        play.page.click("#clients-table th[data-sort='rfqs']")
        by = json.dumps({str(c): n for c, n in rfqs.items()})
        target = json.dumps(want, separators=(",", ":"))
        play.wait(f"JSON.stringify({order_js}.map(c => ({by})[c])) === '{target}'")
        order = play.js(order_js)
        assert [rfqs[c] for c in order] == want


def test_the_calendar_and_log_show_every_event_in_order_and_filter_by_kind(play):
    """The calendar lists the prints, times only; the log has one row per
    event the page was sent, in sequence order; unticking a kind removes
    exactly its rows, ticking it back restores them. A log that dropped or
    reordered events would mislead the review of what happened when."""
    play_some(play, 4)
    for _ in range(3):
        play.press("Tab")
    play.wait("!document.getElementById('tab-log').hidden")
    play.wait("document.querySelectorAll('#log-table tbody tr').length === window.mm.store.events.length")
    prints = play.js("window.mm.store.printSteps.length")
    assert play.js("document.querySelectorAll('#calendar-table tbody tr').length") == max(prints, 1)
    assert "bp" not in text(play, "#calendar-table")  # times only, never outcomes
    seqs = play.js("[...document.querySelectorAll('#log-table tbody tr')].map(tr => Number(tr.children[1].textContent))")
    events = play.js("window.mm.store.events.map(e => e.seq)")
    server_log = play.lt.call(lambda: [e.seq for e in play.host.session.log])
    assert seqs == events == server_log[: len(events)] and len(seqs) > 20
    screens = play.js("window.mm.store.events.filter(e => e.kind === 'screen_update').length")
    play.page.click("#log-filters input[data-kind='screen_update']")
    play.wait(f"document.querySelectorAll('#log-table tbody tr').length === {len(seqs) - screens}")
    assert play.js("document.querySelectorAll('#log-table tbody tr[data-kind=screen_update]').length") == 0
    play.page.click("#log-filters input[data-kind='screen_update']")
    play.wait(f"document.querySelectorAll('#log-table tbody tr').length === {len(seqs)}")


def _typescript() -> str | None:
    found = os.environ.get("MM_TYPESCRIPT_JS")
    if found:
        return found
    roots = [os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", ""), "/usr/share", "/Applications"]
    for root in roots:
        for pattern in ("Programs/Microsoft VS Code/*/resources/app/extensions/node_modules/typescript/lib/typescript.js",
                        "Programs/Microsoft VS Code/resources/app/extensions/node_modules/typescript/lib/typescript.js",
                        "Microsoft VS Code/resources/app/extensions/node_modules/typescript/lib/typescript.js",
                        "code/resources/app/extensions/node_modules/typescript/lib/typescript.js"):
            hits = sorted(glob.glob(os.path.join(root, pattern)))
            if hits:
                return hits[-1]
    return None


def test_the_front_end_type_checks_clean():
    """Every module in web/js/ is // @ts-check, and the TypeScript compiler
    VS Code uses, run over web/jsconfig.json (strict), reports nothing: the
    message shapes in protocol.js are what the code reads. A field renamed on
    the server and not in the page -- or a typo in a field name -- is an
    error here rather than a blank on screen. Runs with Playwright's own Node
    and the TypeScript VS Code ships; set MM_TYPESCRIPT_JS to point elsewhere."""
    web = Path(server.WEB_DIR)
    unchecked = [p.name for p in (web / "js").glob("*.js") if not p.read_text(encoding="utf-8").startswith("// @ts-check")]
    assert not unchecked, unchecked
    import playwright

    node = Path(playwright.__file__).parent / "driver" / ("node.exe" if os.name == "nt" else "node")
    typescript = _typescript()
    if not node.exists() or typescript is None:
        pytest.skip("no Node or TypeScript found to type-check with (set MM_TYPESCRIPT_JS)")
    out = subprocess.run(
        [str(node), str(Path(__file__).parent / "typecheck.js"), typescript, str(web / "jsconfig.json")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
