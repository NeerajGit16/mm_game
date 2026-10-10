"""Tests for the engine, the protocol, the event log and the book.

The ones that matter most: the engine draws nothing and cannot read the
truth (by construction and by perturbing the truth under a fixed screen);
nothing hidden and nothing from the future ever appears in an in-session
message; a refused input changes nothing; and the clock stops on every RFQ so
quotes are stamped where they belong. Each docstring says what silent failure
it catches.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game import agents, protocol, world
from mm_game.book import Book, Source
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings
from mm_game.events import FIELDS, Event, EventLog, Kind
from mm_game.world import PublicWorld

SRC_DIR = Path(mm_game.__file__).resolve().parent
V = protocol.PROTOCOL_VERSION
DRILL = GameConfig.for_session(30)
SCORING = ScoringConfig()
BENCH = BenchmarkConfig()


def new_session(w: world.World, **settings) -> Session:
    return Session(w.public, w.resolve, settings=Settings(**settings), scoring=SCORING)


def code_only(source: str) -> str:
    """Source with docstrings and comments removed: prose may discuss the
    truth, code may not touch it."""
    source = re.sub(r'"""(.|\n)*?"""', "", source)
    return re.sub(r"#.*", "", source)


class Recorder(agents.Strategy):
    """Wraps a strategy and keeps every message it is sent."""

    def __init__(self, inner: agents.Strategy) -> None:
        self.inner = inner
        self.messages: list[dict] = []

    def on_start(self, start, state):
        self.inner.on_start(start, state)
        self.n_steps = self.inner.n_steps

    def next_wake(self):
        return self.inner.next_wake()

    def pop_wake(self):
        return self.inner.pop_wake()

    def observe(self, message):
        self.messages.append(message)
        self.inner.observe(message)

    def answer(self, ticket, state):
        return self.inner.answer(ticket, state)

    def after_answer(self, state):
        return self.inner.after_answer(state)

    def on_wake(self, state, wake):
        return self.inner.on_wake(state, wake)


def recorded_run(public: PublicWorld, resolve, strategy: agents.Strategy, **settings) -> tuple[Session, list[dict]]:
    session = Session(public, resolve, settings=Settings(**settings), scoring=SCORING)
    rec = Recorder(strategy)
    agents.play(session, rec)
    return session, rec.messages


# --------------------------------------------------------------------------
# The governing rule
# --------------------------------------------------------------------------

RANDOM_APIS = re.compile(r"RngStreams|np\.random|numpy\.random|^\s*import random|^\s*from random", re.MULTILINE)


@pytest.mark.parametrize(
    "module", ["engine", "book", "events", "scoring", "protocol", "records", "storage", "game", "agents"]
)
def test_engine_side_draws_nothing(module):
    """Every module that runs or records a session uses no random stream and
    no random API. The world is drawn once, in ``world``; a draw anywhere
    downstream would make two replays of the same actions differ -- or, if
    it came from a session stream, move the market with the player."""
    assert not RANDOM_APIS.search((SRC_DIR / f"{module}.py").read_text(encoding="utf-8"))
    assert "RngStreams(" in (SRC_DIR / "world.py").read_text(encoding="utf-8")


def test_session_is_handed_the_public_world_and_a_resolver_only():
    """Structural: the session is constructed from the public half of the
    world and the one-door resolver, and its code names no hidden field.
    Handing it the world instead would make reading the truth a typo away --
    a live P&L marked to true fair value, a ticket that says "informed"."""
    params = inspect.signature(Session.__init__).parameters
    assert params["public"].annotation in (PublicWorld, "PublicWorld")
    assert {f.name for f in dataclasses.fields(PublicWorld)} == {"config", "view", "flow"}
    code = code_only((SRC_DIR / "engine.py").read_text(encoding="utf-8"))
    body = code.split("def open_session")[0]  # open_session builds the world and hands over the halves
    for hidden in (
        ".values", ".relative", "jump_sizes", "screen_error", "drift_windows", ".diffusion", ".informed",
        "signal_bp", "client_limit", "competitor_", "dealers_in_comp", "tie_break", ".schedule", ".sides", ".path",
    ):
        assert hidden not in body, hidden


def perturb_truth(w: world.World) -> world.World:
    """The same world with a different truth under an unchanged screen,
    schedule and market: a 6bp ramp and every print reversed."""
    ramp = np.linspace(0.0, 6.0, len(w.path.values))
    path = dataclasses.replace(
        w.path, values=w.path.values + ramp, relative=w.path.relative + ramp, jump_sizes=-w.path.jump_sizes
    )
    return dataclasses.replace(w, path=path)


def test_nothing_visible_in_play_moves_with_the_truth():
    """Behavioural: play the same strategy on a world and on the same world
    with a different truth under the same screen. Every in-session message
    -- events, snapshots, the live P&L and risk -- is identical. A leak of
    the truth into anything the player sees shows up as a difference."""
    w = world.generate(GameConfig(), 4)
    other = perturb_truth(w)
    assert not np.array_equal(other.path.values, w.path.values)
    _, a = recorded_run(w.public, w.resolve, agents.Disciplined(w.config, BENCH))
    _, b = recorded_run(other.public, other.resolve, agents.Disciplined(w.config, BENCH))
    assert a == b


# --------------------------------------------------------------------------
# What in-session messages may contain
# --------------------------------------------------------------------------

HIDDEN_NAMES = {
    "informed", "signal_bp", "event_index", "base_size", "dealers_in_comp", "client_limit",
    "competitor_bid_ticks", "competitor_ask_ticks", "competitor_mids", "tie_break_u", "true_fair_value",
    "values", "relative", "jump_sizes", "jump_bp", "drift_windows", "screen_error", "screen_error_bp",
    "informed_type", "client_informed_type", "seed", "canary", "markouts", "hidden", "dealers", "market",
}


def keys_in(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(keys_in(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(keys_in(v) for v in obj)) if obj else set()
    return set()


def full_exercise(w: world.World) -> tuple[Session, list[dict]]:
    """A session that sends every input and so produces every event kind:
    quotes, passes, a timer expiry, a blocked side, a clipped hedge, speed,
    pause, resume and an early quit -- plus a refused input."""
    strategy = agents.ScriptedPlayer(
        w.config, BENCH, quit_at_step=int(w.config.n_steps * 0.8), let_timer_expire_every=7
    )
    session = Session(w.public, w.resolve, settings=Settings(pricing_timer_s=10.0), scoring=SCORING)
    msgs = list(session.start())
    strategy.on_start(next(m for m in msgs if m.get("kind") == "session_start")["data"], msgs[-1])
    state = msgs[-1]
    msgs += session.handle({"v": V, "type": "advance", "to_step": -1})  # refused
    position_goal = 210  # buy into the limit so sides get blocked and a hedge gets clipped
    clipped = False
    while not session.ended:
        if state["ticket"] is not None:
            t = state["ticket"]
            if state["book"]["position"] < 180 and t["ordinal"] % 7:
                s = state["screen"]["mid"]
                out = session.handle(agents.quote(t["ordinal"], s, s + 2.0))  # bid at the screen: buys
            else:
                out = session.handle(strategy.answer(t, state))
            msgs += out
            state = out[-1]
            if not clipped and state["book"]["position"] > 150 and not session.ended:
                out = session.handle(agents.hedge(position_goal - state["book"]["position"] + 30))
                msgs += out
                state = out[-1]
                clipped = True
            continue
        wake = strategy.next_wake()
        if wake is not None and wake <= state["step"]:
            strategy.pop_wake()
            for action in strategy.on_wake(state, wake):
                if session.ended:
                    break
                out = session.handle(action)
                msgs += out
                state = out[-1]
            continue
        out = session.handle(agents.advance(w.config.n_steps if wake is None else wake))
        msgs += out
        state = out[-1]
    return session, msgs


def test_no_hidden_field_appears_in_any_in_session_message():
    """Every message type and every event kind is produced, and none carries
    a hidden field name anywhere in its structure -- not the informed flag,
    not the signal, not the limit, not the competitors, not the truth, not
    the seed (from which everything hidden regenerates). The check that the
    names are real: the post-session record does contain them, so a renamed
    field cannot make this pass by no longer matching."""
    w = world.generate(GameConfig(), 11)
    session, msgs = full_exercise(w)
    kinds = {m["kind"] for m in msgs if m["type"] == "event"}
    assert kinds == {k.value for k in Kind}, set(k.value for k in Kind) - kinds
    assert {m["type"] for m in msgs} == {"event", "state", "error"}
    leaked = keys_in(msgs) & HIDDEN_NAMES
    assert not leaked, leaked
    for m in msgs:
        assert json.loads(protocol.dumps(m)) == m  # plain JSON, nothing that only serialises by luck
    # A hidden value under a public name: a two-way RFQ's side is the
    # client's secret until it trades, on the ticket and in the arrival.
    tickets = [m["data"]["rfq"] for m in msgs if m.get("kind") == "rfq_arrived"]
    tickets += [m["ticket"] for m in msgs if m["type"] == "state" and m["ticket"]]
    assert any(not t["one_way"] for t in tickets)
    for t in tickets:
        assert (t["side"] is None) == (not t["one_way"])
        assert t["side"] is None or t["side"] == int(w.schedule.rfqs[t["ordinal"]].side)

    from mm_game import records

    record = records.coaching_record(w, session, scoring=SCORING, benchmark=BENCH, bots={
        b: agents.run_bot(b, w, settings=session.settings, scoring=SCORING, benchmark=BENCH) for b in agents.BOTS
    }, flags=records.flags(session, False))
    assert {"informed", "signal_bp", "client_limit", "true_fair_value", "seed", "canary", "dealers"} <= keys_in(record)


def perturb_after(w: world.World, k: int) -> world.World:
    """The same world up to step k, different after it: screen readings,
    the truth, print outcomes, and every later RFQ's size, client, side,
    competition, limit and dealer quotes. Print times are unchanged -- the
    calendar is public in advance."""
    path, view = w.path, w.path.public
    screen = view.screen.copy()
    screen[view.screen_steps > k] += 1.0
    values = w.path.values.copy()
    values[k + 1 :] += 2.0
    relative = w.path.relative.copy()
    relative[k + 1 :] += 2.0
    jumps = path.jump_sizes.copy()
    jumps[path.event_steps > k] *= -1
    new_view = dataclasses.replace(view, screen=screen)
    new_path = dataclasses.replace(path, values=values, relative=relative, jump_sizes=jumps, public=new_view)
    n_clients = len(w.schedule.clients)
    rfqs, sides = [], []
    for q, s in zip(w.schedule.rfqs, w.sides):
        if q.step > k:
            q = dataclasses.replace(
                q,
                size=q.size % 60 + 3,
                client_id=(q.client_id + 1) % n_clients,
                side=type(q.side)(-int(q.side)),
                one_way=not q.one_way,
                dealers_in_comp=tuple(not d for d in q.dealers_in_comp),
            )
            s = dataclasses.replace(
                s,
                client_limit=s.client_limit + 5.0,
                competitor_bid_ticks=tuple(b + 3 for b in s.competitor_bid_ticks),
                competitor_ask_ticks=tuple(a + 3 for a in s.competitor_ask_ticks),
            )
        rfqs.append(q)
        sides.append(s)
    schedule = dataclasses.replace(w.schedule, rfqs=tuple(rfqs))
    return dataclasses.replace(w, path=new_path, schedule=schedule, sides=tuple(sides))


@pytest.mark.parametrize("fraction", [0.25, 0.6])
def test_no_in_session_message_reveals_anything_from_a_later_step(fraction):
    """Change everything after step k -- screen, truth, prints, RFQs, covers,
    limits -- and replay the same strategy: every message stamped at step k
    or earlier is identical. A snapshot that peeked at the next RFQ, a chart
    feed that sent the screen ahead of the clock, a countdown that carried
    the print's size: each would differ before k. Print times are the one
    thing allowed to be known in advance, and they are left unchanged."""
    w = world.generate(GameConfig(), 5)
    k = int(w.config.n_steps * fraction)
    other = perturb_after(w, k)
    _, a = recorded_run(w.public, w.resolve, agents.Disciplined(w.config, BENCH))
    _, b = recorded_run(other.public, other.resolve, agents.Disciplined(w.config, BENCH))

    def upto(msgs):
        out = []
        for m in msgs:
            if m["step"] > k:
                break
            out.append(m)
        return out

    early_a, early_b = upto(a), upto(b)
    assert len(early_a) > 100
    assert early_a == early_b
    assert a != b  # the perturbation is real and does show once the clock passes k


def test_every_message_carries_values_from_its_own_step_or_earlier():
    """Directly, message by message: a snapshot's screen is the reading
    showing at its own step, with that reading's true age; a ticket is an
    RFQ that has arrived by then; a screen update is the observation taken
    at its step; the countdown is the calendar's. Perturbing the future only
    catches a leak that happens to surface just before the cut; this
    catches a lookahead of any length, including one reading ahead."""
    w = world.generate(GameConfig(), 13)
    view = w.path.public
    _, msgs = recorded_run(w.public, w.resolve, agents.ScriptedPlayer(w.config, BENCH))
    obs = {int(s): float(v) for s, v in zip(view.screen_steps, view.screen)}
    checked = 0
    for m in msgs:
        step = m["step"]
        if m["type"] == "state":
            assert m["screen"]["mid"] == view.screen_at(step)
            assert m["screen"]["age_s"] == view.screen_age_s(step)
            if m["ticket"] is not None:
                assert w.schedule.rfqs[m["ticket"]["ordinal"]].step <= step
            nxt = m["risk"]["next_print"]
            later = [int(e) for e in view.event_steps if e > step]
            assert (nxt is None) == (not later) and (nxt is None or nxt["step"] == later[0])
            checked += 1
        elif m["kind"] == "screen_update":
            assert obs[step] == m["data"]["mid"]
        elif m["kind"] in ("rfq_arrived", "rfq_resolved", "quote_sent"):
            ordinal = m["data"]["rfq"]["ordinal"] if m["kind"] == "rfq_arrived" else m["data"]["rfq_ordinal"]
            assert w.schedule.rfqs[ordinal].step == step
    assert checked > 100


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------


def test_clock_stops_at_every_rfq_and_quotes_are_stamped_there():
    """Advancing to the end stops at the first RFQ; every quote is stamped
    at its RFQ's step whatever the player did in between; the clock will not
    move past an unanswered RFQ. A clock that ran on while the player
    thought would stamp quotes late and make replays of the same seed differ
    in timing as well as decisions."""
    w = world.generate(DRILL, 3)
    s = new_session(w)
    s.start()
    first = w.schedule.rfqs[0]
    out = s.handle(agents.advance(w.config.n_steps))
    assert s.step == first.step and out[-1]["ticket"]["ordinal"] == 0
    err = s.handle(agents.advance(w.config.n_steps))
    assert err[0]["type"] == "error" and s.step == first.step
    s.handle({"v": V, "type": "pause"})
    s.handle({"v": V, "type": "resume"})
    s.handle(agents.hedge(5))  # while paused on the RFQ: stamped at its step
    s.handle(agents.pass_(0))
    assert all(e.step == first.step for e in s.log if e.seq >= 1 and e.kind in (Kind.QUOTE_SENT, Kind.HEDGE_EXECUTED))
    for e in s.log.of(Kind.QUOTE_SENT):
        assert e.step == w.schedule.rfqs[e.data["rfq_ordinal"]].step


def test_close_rfqs_queue_and_the_next_appears_only_once_answered():
    """Two RFQs a few seconds apart: advancing past both stops at the first;
    the second is not on the ticket, nor in any message, until the first is
    answered and the clock reaches it."""
    w = world.generate(DRILL, 3)
    rfqs = w.schedule.rfqs
    i = next(j for j in range(len(rfqs) - 1) if rfqs[j + 1].step - rfqs[j].step < 150)
    s = new_session(w)
    s.start()
    while s.step < rfqs[i].step:
        s.handle(agents.advance(rfqs[i].step))
        if s.pending is not None and s.pending < i:
            s.handle(agents.pass_(s.pending))
    assert s.pending == i
    out = s.handle(agents.advance(rfqs[i + 1].step + 100))
    assert out[0]["type"] == "error"
    assert not any(e.kind is Kind.RFQ_ARRIVED and e.data["rfq"]["ordinal"] == i + 1 for e in s.log)
    s.handle(agents.pass_(i))
    out = s.handle(agents.advance(rfqs[i + 1].step + 100))
    assert s.pending == i + 1 and s.step == rfqs[i + 1].step


def test_items_are_revealed_at_their_own_steps():
    """Every screen reading appears at its observation step with its value,
    a print and its forced screen observation appear together at the print
    step, and every RFQ at its arrival step."""
    w = world.generate(GameConfig(), 2)
    s, _ = recorded_run(w.public, w.resolve, agents.Naive(w.config, BENCH))
    view = w.path.public
    updates = s.log.of(Kind.SCREEN_UPDATE)
    assert [e.step for e in updates] == [int(x) for x in view.screen_steps]
    assert [e.data["mid"] for e in updates] == [float(x) for x in view.screen]
    printed = s.log.of(Kind.PRINT_RELEASED)
    assert [e.step for e in printed] == [int(x) for x in view.event_steps]
    for e in printed:
        assert any(u.step == e.step for u in updates)
    assert [e.step for e in s.log.of(Kind.RFQ_ARRIVED)] == [q.step for q in w.schedule.rfqs]


def test_front_end_inputs_are_logged_and_change_nothing_in_the_world():
    """Speed changes, pauses and resumes are events at their steps, and a
    session played with them is identical in every world event and outcome
    to one played without them."""
    w = world.generate(DRILL, 6)
    plain_s, _ = recorded_run(w.public, w.resolve, agents.Disciplined(w.config, BENCH))
    busy = agents.ScriptedPlayer(w.config, BENCH)
    busy_s, _ = recorded_run(w.public, w.resolve, busy)
    assert busy_s.log.of(Kind.SPEED_CHANGED) and busy_s.log.of(Kind.PAUSED) and busy_s.log.of(Kind.RESUMED)
    world_kinds = (Kind.SCREEN_UPDATE, Kind.PRINT_RELEASED, Kind.RFQ_ARRIVED)
    assert [e.to_record() | {"seq": 0} for e in plain_s.log.of(*world_kinds)] == [
        e.to_record() | {"seq": 0} for e in busy_s.log.of(*world_kinds)
    ]


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        {"v": 2, "type": "pass", "rfq_ordinal": 0},  # future protocol version
        {"type": "pass", "rfq_ordinal": 0},  # no version
        {"v": V, "type": "teleport"},
        {"v": V, "type": "hedge", "quantity": 0},
        {"v": V, "type": "hedge", "quantity": 1.5},
        {"v": V, "type": "hedge", "quantity": 5, "price": 9550.0},  # unknown parameter
        {"v": V, "type": "set_speed", "speed": 3},
        {"v": V, "type": "quote", "rfq_ordinal": 0, "bid": float("nan"), "ask": None},
        {"v": V, "type": "quote", "rfq_ordinal": 0, "bid": 9550.1, "ask": None},  # off the grid
        {"v": V, "type": "quote", "rfq_ordinal": 0, "bid": 9551.0, "ask": 9550.0},  # crossed
        {"v": V, "type": "quote", "rfq_ordinal": 1, "bid": 9549.0, "ask": 9551.0},  # not the RFQ on the ticket
        {"v": V, "type": "timer_expired", "rfq_ordinal": 0},  # timer off
        {"v": V, "type": "resume"},  # not paused
        "not a dict",
    ],
)
def test_refused_inputs_change_nothing(message):
    """A refused input answers with an error and leaves the session exactly
    as it was: no event, no replay action, no book change. A half-applied
    refusal -- a block logged before the quote is found off-grid -- would
    put things in the record that never happened."""
    w = world.generate(DRILL, 3)
    s = new_session(w)
    s.start()
    s.handle(agents.advance(w.config.n_steps))
    assert s.pending == 0
    before = (len(s.log), list(s.actions), s.book.trades, s.state())
    out = s.handle(message)
    assert out[0]["type"] == "error" and out[-1]["type"] == "state"
    assert (len(s.log), list(s.actions), s.book.trades, s.state()) == before


def test_actions_record_every_accepted_input_with_its_step_and_no_wall_time():
    w = world.generate(DRILL, 3)
    s, _ = recorded_run(w.public, w.resolve, agents.ScriptedPlayer(w.config, BENCH))
    assert s.actions and all(set(a) >= {"seq", "step", "type"} for a in s.actions)
    assert not any(a["type"] == "advance" for a in s.actions)
    text = json.dumps(s.actions)
    assert "time" not in text and "clock" not in text


def test_events_refuse_undeclared_fields():
    """The in-session vocabulary is declared per kind; an event carrying any
    other field -- a hidden one added to a dict by mistake -- is refused."""
    with pytest.raises(ValueError):
        Event(seq=0, step=0, kind=Kind.SCREEN_UPDATE, data={"mid": 9550.0, "true_fair_value": 9550.1})
    log = EventLog()
    log.append(5, Kind.PAUSED)
    with pytest.raises(ValueError):
        log.append(4, Kind.RESUMED)  # steps never go backwards
    assert all(isinstance(v, frozenset) for v in FIELDS.values())


def test_engine_needs_no_threads_or_processes():
    """In-browser capable: nothing in the package imports threading,
    multiprocessing, asyncio or subprocess."""
    for path in SRC_DIR.glob("*.py"):
        code = code_only(path.read_text(encoding="utf-8"))
        assert not re.search(r"^\s*(import|from)\s+(threading|multiprocessing|asyncio|subprocess|concurrent)", code, re.M), path.name


# --------------------------------------------------------------------------
# Book
# --------------------------------------------------------------------------


def test_book_average_cost_and_screen_mark():
    """Realised plus unrealised equals cash plus the position at the mark
    through buys, partial closes and a flip; unrealised is marked at the
    price it is handed, which in play is the screen."""
    cfg = GameConfig()
    b = Book(cfg)
    L = cfg.start_level
    b.trade(step=1, source=Source.CLIENT, quantity=10, price=L + 1.0)
    b.trade(step=2, source=Source.CLIENT, quantity=10, price=L + 2.0)  # avg +1.5
    b.trade(step=3, source=Source.HEDGE, quantity=-5, price=L + 3.0)  # realises 5 * 1.5
    assert b.realised_bp() == pytest.approx(7.5)
    b.trade(step=4, source=Source.CLIENT, quantity=-25, price=L + 0.5)  # closes 15 at +0.5 vs 1.5, opens -10
    assert b.position == -10 and b.realised_bp() == pytest.approx(7.5 - 15.0)
    for mark in (L - 1.0, L, L + 2.25):
        assert b.realised_bp() + b.unrealised_bp(mark) == pytest.approx(b.total_bp(mark))
    assert b.unrealised_bp(L) == pytest.approx(-10 * (0.0 - 0.5))
    assert b.usd(1.0) == cfg.dollars_per_bp_per_lot
    with pytest.raises(ValueError):
        b.trade(step=5, source=Source.HEDGE, quantity=-195, price=L)  # through the limit
