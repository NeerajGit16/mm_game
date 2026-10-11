"""Browser tests: real key presses in a real page, against a real server.

The server runs on a background thread with the wall clock sped up thirty
times, exactly as in play otherwise; the page is the shipped front end in
headless Edge (or Chrome). What a key did is read from the engine's own
log on the server, not from the page, so a page that drew the right thing
but sent the wrong message would still fail. Each docstring says what
silent failure it catches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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
