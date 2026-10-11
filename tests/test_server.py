"""Tests for the session server, its clock and the desk.

The server is where the governing rule becomes structural: the browser can
only ever know what crosses the websocket. So these tests watch the wire,
not the engine -- every frame a real websocket client receives -- and ask
what a page could learn from it. Each docstring says what silent failure it
catches.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from mm_game import agents, desk as desk_module, game, host as host_module, protocol, records, server, world
from mm_game.book import Source
from mm_game.config import BenchmarkConfig, GameConfig
from mm_game.engine import Settings
from mm_game.events import FIELDS, Kind
from mm_game.storage import DirectoryStorage, MemoryStorage
from test_engine import HIDDEN_NAMES, keys_in
from wire import LoopThread, ManualClock, Wire, WireSession, serve

V = protocol.PROTOCOL_VERSION
BENCH = BenchmarkConfig()
WEB = Path(server.__file__).resolve().parent / "web"


@pytest.fixture
def lt():
    loop = LoopThread()
    yield loop
    loop.close()


def hosted(lt, cfg: GameConfig, seed: int, settings: Settings) -> tuple[host_module.Host, Wire]:
    g = game.Game(cfg, seed, storage=MemoryStorage(), settings=settings)
    host = lt.call(lambda: host_module.Host(g, ManualClock(), records_dir="records"))
    return host, Wire(lt, host, serve(lt, host, tick_ms=None))


# --------------------------------------------------------------------------
# What may cross the wire during play
# --------------------------------------------------------------------------

LEAF = "leaf"


class Opt:
    def __init__(self, schema):
        self.schema = schema


RFQ = {k: LEAF for k in ("ordinal", "client_id", "category", "size", "one_way", "side", "in_comp")}
STATE = {
    "v": LEAF, "type": LEAF, "step": LEAF, "time_s": LEAF, "status": LEAF,
    "clock": {"speed": LEAF, "paused": LEAF, "awaiting_quote": LEAF},
    "screen": {"mid": LEAF, "age_s": LEAF},
    "book": {k: LEAF for k in (
        "position", "realised_usd", "unrealised_usd", "total_usd", "limit", "room_to_buy", "room_to_sell")},
    "risk": {"penalty_so_far_usd": LEAF, "next_print": Opt({k: LEAF for k in ("print_index", "step", "in_s", "penalty_if_held_usd")})},
    "ticket": Opt({**RFQ, "blocked_sides": [LEAF]}),
    "venue": {"quotes": [{"size": LEAF, "cost_bp_per_lot": LEAF}], "near_print": LEAF},
}
HORIZONS = {"60s": LEAF, "300s": LEAF}
LEARNED = ("learned", {"cover"}, {"side", "level"})  # None, or one of these key sets
DESK = {
    "v": LEAF, "type": LEAF, "step": LEAF,
    "blotter": [{
        **{k: LEAF for k in ("ordinal", "step", "client_id", "category", "size", "one_way", "side", "in_comp", "bid",
                             "ask", "outcome", "quantity", "price", "revealed_side", "position_after")},
        "learned": LEARNED,
    }],
    "clients": [{
        **{k: LEAF for k in ("client_id", "category", "rfqs", "quoted", "traded", "lots", "net", "last_trade_step",
                             "hit_rate")},
        "markout_bp_per_lot": HORIZONS, "markout_lots": HORIZONS,
        "recent": [{**{k: LEAF for k in ("ordinal", "step", "size", "outcome", "quantity", "level", "client_side")},
                    "markout_bp_per_lot": HORIZONS}],
    }],
    "pnl": {
        "rows": [{k: LEAF for k in ("seq", "step", "kind", "ordinal", "client_id", "category", "quantity", "price",
                                    "screen_then", "edge_usd", "move_usd", "total_usd")}],
        "tiles": {k: LEAF for k in ("edge_client_usd", "venue_charge_usd", "move_usd", "total_usd", "realised_usd",
                                    "unrealised_usd")},
        "series": {"step": [LEAF], "pnl_usd": [LEAF], "position": [LEAF]},
    },
}
HOST = {"v": LEAF, "type": LEAF, "begun": LEAF}
HANDBOOK = {
    "v": LEAF, "type": LEAF, "scored_sessions": LEAF, "unlock_after": LEAF,
    "parts": [{"title": LEAF, "subtitle": LEAF, "note": LEAF, "items": [{"head": LEAF, "text": LEAF}]}],
}
ERROR = {"v": LEAF, "type": LEAF, "step": LEAF, "detail": LEAF}


def conforms(value, schema, path="$") -> None:
    """``value`` has exactly the declared shape: no extra key anywhere."""
    if isinstance(schema, Opt):
        if value is not None:
            conforms(value, schema.schema, path)
    elif schema == LEAF:
        assert value is None or isinstance(value, (bool, int, float, str)), path
    elif isinstance(schema, tuple):  # the learned field
        assert value is None or set(value) in schema[1:], (path, value)
        assert value is None or all(v is None or isinstance(v, (int, float)) for v in value.values()), path
    elif isinstance(schema, list):
        assert isinstance(value, list), path
        for i, v in enumerate(value):
            conforms(v, schema[0], f"{path}[{i}]")
    else:
        assert isinstance(value, dict) and set(value) == set(schema), (path, sorted(value) if isinstance(value, dict) else value)
        for k, s in schema.items():
            conforms(value[k], s, f"{path}.{k}")


def check_message(m: dict) -> None:
    t = m["type"]
    if t == "event":
        assert set(m) == {"v", "type", "seq", "step", "kind", "data"}
        assert set(m["data"]) == FIELDS[Kind(m["kind"])], m["kind"]
        if m["kind"] == "rfq_arrived":
            conforms(m["data"]["rfq"], RFQ)
        if m["kind"] == "session_start":
            assert set(m["data"]["config"]) == set(GameConfig().to_record())  # the scenario, which is public
            for c in m["data"]["roster"]:
                assert set(c) == {"client_id", "category"}
    else:
        conforms(m, {"state": STATE, "desk": DESK, "host": HOST, "handbook": HANDBOOK, "error": ERROR}[t])


def hidden_values(w: world.World) -> set[float]:
    """Hidden numbers that are not on the tick grid, so no public number can
    equal one by coincidence: the truth at every step, every signal, print
    and move size, client limit and competitor mid."""
    vals = {float(x) for x in w.path.values}
    vals |= {float(q.signal_bp) for q in w.schedule.rfqs if q.signal_bp is not None}
    vals |= {float(x) for x in w.path.jump_sizes} | {m.total_bp for m in w.path.move_list}
    vals |= {float(s.client_limit) for s in w.sides} | {float(m) for s in w.sides for m in s.competitor_mids}
    tick = w.config.tick_size
    return {v for v in vals if abs(v / tick - round(v / tick)) > 1e-6}


def floats_in(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from floats_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from floats_in(v)
    elif isinstance(obj, float):
        yield obj


def assert_nothing_hidden(w: world.World, wire: Wire) -> list[dict]:
    """Every message before the reveal: declared types, declared shapes,
    no hidden name, no hidden value. The reveal comes once, after the end."""
    msgs = wire.messages()
    reveals = [i for i, m in enumerate(msgs) if m["type"] == "reveal"]
    assert len(reveals) == 1
    end = next(i for i, m in enumerate(msgs) if m.get("kind") == "session_end")
    assert reveals[0] > end
    play = msgs[: reveals[0]]
    assert {m["type"] for m in play} <= {"event", "state", "error", "desk", "host", "handbook"}
    for m in play:
        check_message(m)
    assert not keys_in(play) & HIDDEN_NAMES, keys_in(play) & HIDDEN_NAMES
    secret = hidden_values(w)
    leaked = [x for m in play for x in floats_in(m) if x in secret]
    assert not leaked, leaked[:5]
    # A two-way RFQ's side is the client's secret until the outcome shows it.
    rows = [r for m in play if m["type"] == "desk" for r in m["blotter"]]
    tickets = [m["data"]["rfq"] for m in play if m.get("kind") == "rfq_arrived"]
    tickets += [m["ticket"] for m in play if m["type"] == "state" and m["ticket"]] + rows
    assert any(not t["one_way"] for t in tickets)
    for t in tickets:
        assert (t["side"] is None) == (not t["one_way"])
    for r in rows:
        assert (r["revealed_side"] is not None) == (r["outcome"] in ("won", "done_away"))
    return play


def drive(wire: Wire, host: host_module.Host, rng: np.random.Generator, *, quit_at: int | None, scripted_start: bool) -> None:
    """Play a whole session through the websocket with random inputs, some
    refused. With ``scripted_start``, first send the inputs that make every
    event kind appear: a hedge, a clipped hedge, a speed change, a pause and
    resume, a blocked quote side, a timer expiry."""
    tick = host.game.session.config.tick_size
    n = host.game.session.n_steps
    state = [m for m in wire.send({"v": V, "type": "begin"}) + wire.first if m["type"] == "state"][-1]

    def keep(out):
        nonlocal state
        states = [m for m in out if m["type"] == "state"]
        if states:
            state = states[-1]

    def act(message):  # anything, sent over the websocket
        keep(wire.send(message))

    def clock(to_step):  # the server's clock moving
        keep(wire.advance(to_step))

    if scripted_start:
        for m in ({"v": V, "type": "hedge", "quantity": 190}, {"v": V, "type": "hedge", "quantity": 50},
                  {"v": V, "type": "set_speed", "speed": 4}, {"v": V, "type": "pause"}, {"v": V, "type": "resume"}):
            act(m)
    timer_done = False
    while state["status"] == "live":
        if quit_at is not None and state["step"] >= quit_at:
            act({"v": V, "type": "quit"})
            break
        t = state["ticket"]
        r = rng.random()
        if t is not None:
            mid = state["screen"]["mid"]
            if scripted_start and not timer_done and state["book"]["position"] >= 190:
                act({"v": V, "type": "quote", "rfq_ordinal": t["ordinal"], "bid": mid - tick, "ask": mid + tick})
                act({"v": V, "type": "hedge", "quantity": -190})
                continue
            if scripted_start and not timer_done:
                timer_done = True
                act({"v": V, "type": "timer_expired", "rfq_ordinal": t["ordinal"]})
            elif r < 0.6:
                skew, half = int(rng.integers(-4, 5)), int(rng.integers(1, 5))
                act({"v": V, "type": "quote", "rfq_ordinal": t["ordinal"], "bid": mid + (skew - half) * tick,
                     "ask": mid + (skew + half) * tick})
            elif r < 0.75:
                act({"v": V, "type": "pass", "rfq_ordinal": t["ordinal"]})
            elif r < 0.85:
                bad = [
                    {"v": V, "type": "quote", "rfq_ordinal": t["ordinal"], "bid": mid + 0.1, "ask": None},
                    {"v": V, "type": "quote", "rfq_ordinal": t["ordinal"] + 1, "bid": mid - 1, "ask": mid + 1},
                    {"v": V, "type": "advance", "to_step": n},
                    {"v": V, "type": "teleport"},
                    "not a dict",
                ]
                act(bad[int(rng.integers(len(bad)))])
            else:
                act({"v": V, "type": "hedge", "quantity": int(rng.integers(-60, 61)) or 1})
        else:
            if r < 0.75:
                clock(min(n, state["step"] + int(rng.integers(1, 4000))))
            elif r < 0.85:
                act({"v": V, "type": "hedge", "quantity": int(rng.integers(-80, 81)) or -1})
            elif r < 0.9:
                act({"v": V, "type": "set_speed", "speed": int(rng.choice([1, 2, 3, 4]))})
            elif r < 0.95:
                act({"v": V, "type": "pause"})
                clock(n)  # nothing moves while paused
                act({"v": V, "type": "resume"})
            else:
                act({"v": V, "type": "set_speed", "speed": 2})


def test_no_hidden_field_crosses_the_wire_during_play(lt):
    """A whole session through a real websocket, every event kind produced,
    random inputs and refusals between: every message a page receives
    before the reveal has a declared type and exactly its declared shape --
    event payloads exactly as ``events.FIELDS`` says, snapshots, the desk
    and host status key for key -- carries no hidden field name, and no
    number equal to a hidden one (the truth at any step, a signal, a print
    or move size, a client limit, a competitor's mid). The coaching record
    arrives once, after the session has ended. A new snapshot field, a desk
    column fed from the wrong place, or the server forwarding something of
    its own would each fail here before reaching a page."""
    cfg = GameConfig()
    seed = next(
        s for s in range(11, 120)
        if (w := world.generate(cfg, s)).path.move_list and (w.path.event_steps < 0.8 * cfg.n_steps).any()
    )
    host, wire = hosted(lt, cfg, seed, Settings(pricing_timer_s=10.0))
    drive(wire, host, np.random.default_rng(1), quit_at=int(0.8 * cfg.n_steps), scripted_start=True)
    play = assert_nothing_hidden(host.game._world, wire)
    kinds = {m["kind"] for m in play if m["type"] == "event"}
    assert kinds == {k.value for k in Kind}, {k.value for k in Kind} - kinds
    assert {m["type"] for m in play} == {"event", "state", "error", "desk", "host", "handbook"}
    assert any(m["type"] == "summary" for m in wire.messages())


@pytest.mark.parametrize("preset,seed,fuzz,quit_frac", [(30, 1, 2, None), (60, 4, 3, None), (120, 9, 4, 0.5)])
def test_fuzzed_sessions_leak_nothing(lt, preset, seed, fuzz, quit_frac):
    """Random play over full sessions -- quotes at any skew and width,
    passes, hedges of any size, speeds, pauses, refused inputs -- to the
    close or a quit: the same checks as above, every message."""
    cfg = GameConfig.for_session(preset)
    host, wire = hosted(lt, cfg, seed, Settings(preset=preset))
    quit_at = None if quit_frac is None else int(quit_frac * cfg.n_steps)
    drive(wire, host, np.random.default_rng(fuzz), quit_at=quit_at, scripted_start=False)
    assert_nothing_hidden(host.game._world, wire)


def test_the_front_end_cannot_move_the_clock(lt):
    """``advance`` from a page is refused and changes nothing: the server
    owns the clock. Otherwise a page could step past an RFQ's timing or
    fast-forward the world."""
    host, wire = hosted(lt, GameConfig.for_session(30), 2, Settings(preset=30))
    wire.send({"v": V, "type": "begin"})
    out = wire.send({"v": V, "type": "advance", "to_step": 5000})
    assert [m["type"] for m in out][:2] == ["error", "state"] and out[1]["step"] == 0
    assert lt.call(lambda: host.session.step) == 0 and not any(m["type"] == "event" for m in out)


def test_the_page_speaks_the_servers_protocol_version():
    """The front end's messages carry the version the server reads; a bump
    on one side only would refuse every key press."""
    text = (WEB / "js" / "protocol.js").read_text(encoding="utf-8")
    assert re.search(r"export const V = (\d+);", text).group(1) == str(V)


# --------------------------------------------------------------------------
# The same session, through the wire or headless
# --------------------------------------------------------------------------


class Recording:
    """The headless driver's view of a game, keeping every message."""

    def __init__(self, g: game.Game) -> None:
        self.g = g
        self.messages: list[dict] = []

    @property
    def ended(self) -> bool:
        return self.g.session.ended

    def start(self):
        out = self.g.start()
        self.messages += out
        return out

    def handle(self, message):
        out = self.g.handle(message)
        self.messages += out
        return out


@pytest.mark.parametrize("preset,seed,quit_minute", [(60, 7, None), (30, 3, 20)])
def test_a_session_through_the_websocket_reproduces_the_headless_run(lt, preset, seed, quit_minute):
    """The scripted player -- quotes, passes, hedges, timer expiries, a
    quit -- driven through the server and a real websocket produces exactly
    the headless run: the same engine messages in the same order, the same
    event log and replay actions, the same coaching record and summary. If
    the server's batching, its clock or the JSON on the wire changed any
    step, price or ordering, the two would part."""
    cfg = GameConfig.for_session(preset)
    settings = Settings(preset=preset, pricing_timer_s=15.0)
    quit_at = None if quit_minute is None else round(quit_minute * 60 / cfg.dt)

    def player():
        return agents.ScriptedPlayer(cfg, BENCH, quit_at_step=quit_at, let_timer_expire_every=25)

    headless = game.Game(cfg, seed, storage=MemoryStorage(), settings=settings)
    rec = Recording(headless)
    agents.play(rec, player())

    host, wire = hosted(lt, cfg, seed, settings)
    agents.play(WireSession(wire), player())

    on_wire = [m for m in wire.messages() if m["type"] in ("event", "state", "error", "reveal")]
    assert on_wire == json.loads(protocol.dumps({"m": rec.messages}))["m"]
    assert host.game.session.log.to_record() == headless.session.log.to_record()
    assert host.game.session.actions == headless.session.actions
    assert host.game.record == headless.record
    summary = next(m for m in wire.messages() if m["type"] == "summary")
    assert summary["markdown"] == headless.storage.read_text(f"sessions/{headless.session_id}/summary.md")
    assert summary["markdown"] == records.markdown_summary(headless.record, headless.track.aggregates())
    assert summary["session_id"] == headless.session_id


def test_a_page_that_reconnects_gets_the_public_history(lt):
    """A second page connecting mid-session receives every event so far,
    the latest snapshot and desk, and the host status -- enough to rebuild
    the screen -- and nothing else."""
    cfg = GameConfig.for_session(30)
    host, wire = hosted(lt, cfg, 5, Settings(preset=30))
    session = WireSession(wire)
    strategy = agents.ScriptedPlayer(cfg, BENCH, quit_at_step=None, let_timer_expire_every=None)
    first = session.start()
    strategy.on_start(next(m for m in first if m.get("kind") == "session_start")["data"], first[-1])
    for step in range(2000, 9000, 1000):
        out = session.handle(agents.advance(step))
        while out and out[-1]["ticket"] is not None:
            out = session.handle(strategy.answer(out[-1]["ticket"], out[-1]))
    port = serve(lt, host, tick_ms=None)
    again = Wire(lt, host, port)
    events = [m for m in wire.messages() if m["type"] == "event"]
    history = again.first
    assert [m for m in history if m["type"] == "event"] == events
    latest = {m["type"]: m for m in wire.messages()}
    assert [m["type"] for m in history if m["type"] != "event"] == ["state", "desk", "host", "handbook"]
    assert history[-4] == latest["state"] and history[-3] == latest["desk"] and history[-2]["begun"] is True
    assert history[-1] == next(m for m in wire.first if m["type"] == "handbook")


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------


def test_wall_clock_runs_at_speed_and_never_catches_up_stopped_time():
    """Wall seconds times speed over dt is steps. Stopped time -- pause, an
    RFQ on the ticket -- is dropped, not caught up: the clock restarts from
    the session's own step. A speed change counts only the time after it.
    A clock that caught up would fast-forward past RFQs the moment you
    finished pricing one."""
    t = [100.0]
    c = host_module.WallClock(0.1, now=lambda: t[0])
    assert c.target(0, 1, True) == 0  # the first call anchors
    t[0] = 102.0
    assert c.target(0, 1, True) == 20
    assert c.target(20, 4, True) == 20  # speed change: re-anchored here
    t[0] = 103.0
    assert c.target(20, 4, True) == 60
    t[0] = 133.0
    assert c.target(60, 4, False) == 60  # stopped for 30 s
    t[0] = 134.0
    assert c.target(60, 4, True) == 60  # restarts from 60, nothing caught up
    t[0] = 134.5
    assert c.target(60, 4, True) == 80


def test_the_clock_stops_while_an_rfq_is_on_the_ticket():
    """Hosted with a wall clock: ticks advance the session until an RFQ
    arrives, then nothing moves however long the ticket sits; once it is
    answered the clock runs on from the RFQ's step."""
    t = [0.0]
    cfg = GameConfig.for_session(30)
    g = game.Game(cfg, 3, storage=MemoryStorage(), settings=Settings(preset=30))
    host = host_module.Host(g, host_module.WallClock(cfg.dt, now=lambda: t[0]), records_dir="records")
    assert not host.tick()  # not begun
    host.receive(json.dumps({"v": V, "type": "begin"}))
    host.tick()
    while host.session.pending is None:
        t[0] += 1.0
        host.tick()
    rfq_step = host.session.step
    assert rfq_step == g._world.schedule.rfqs[0].step
    t[0] += 600.0
    assert not host.tick() and host.session.step == rfq_step
    host.receive(json.dumps({"v": V, "type": "pass", "rfq_ordinal": host.session.pending}))
    host.tick()  # restarts the clock here
    t[0] += 0.5
    host.tick()
    assert host.session.step == min(rfq_step + 5, g._world.schedule.rfqs[1].step)


# --------------------------------------------------------------------------
# The desk
# --------------------------------------------------------------------------


def test_the_desk_reads_only_the_message_stream():
    """``desk.py`` imports nothing from the package but the protocol's
    version: it cannot reach the world, the flow or the fills, so whatever
    it computes the player could have computed from the screen."""
    tree = ast.parse(Path(desk_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert {m for m in imported if m.startswith("mm_game")} == {"mm_game.protocol"}
    assert imported - {"mm_game.protocol"} <= {"__future__", "bisect", "typing"}


def played_desk(preset: int, seed: int):
    cfg = GameConfig.for_session(preset)
    g = game.Game(cfg, seed, storage=MemoryStorage(), settings=Settings(preset=preset))
    rec = Recording(g)
    agents.play(rec, agents.ScriptedPlayer(cfg, BENCH, quit_at_step=None, let_timer_expire_every=None))
    d = desk_module.Desk()
    snapshots = []
    for m in rec.messages:
        d.feed(m)
        if m["type"] == "state":
            snapshots.append(d.message())
    return g, rec, d.message(), snapshots


def test_desk_markouts_are_your_fills_against_the_screen():
    """Per client: RFQs, trades, lots and net from the book, and 60 s and
    300 s mark-outs per lot recomputed from the trades and the public
    screen, a different route from the desk's own: sum of q x (screen at
    t + h - screen at t) over fills whose horizon has passed, over their
    lots. Mid-session, a fill younger than the horizon is not counted yet.
    A mark-out that included the edge, used the truth, or counted unmatured
    fills would show a toxic client as clean or the reverse."""
    g, rec, final, snapshots = played_desk(60, 7)
    view = g._world.path.public
    dt = g.session.config.dt
    fills = [t for t in g.session.book.trades if t.source is Source.CLIENT]
    client_of = {q.ordinal: q.client_id for q in g._world.schedule.rfqs}
    checked = 0
    for snap in (snapshots[len(snapshots) // 2], final):
        for c in snap["clients"]:
            mine = [f for f in fills if client_of[f.rfq_ordinal] == c["client_id"] and f.step <= snap["step"]]
            assert c["traded"] == len(mine) and c["net"] == sum(f.quantity for f in mine)
            assert c["lots"] == sum(abs(f.quantity) for f in mine)
            for h in (60, 300):
                hs = round(h / dt)
                done = [f for f in mine if f.step + hs <= snap["step"]]
                lots = sum(abs(f.quantity) for f in done)
                expected = (
                    sum(f.quantity * (view.screen_at(f.step + hs) - view.screen_at(f.step)) for f in done) / lots
                    if lots else None
                )
                got = c["markout_bp_per_lot"][f"{h}s"]
                assert (got is None) == (expected is None) and (got is None or math.isclose(got, expected, abs_tol=1e-9))
                checked += got is not None
    assert checked > 10


def test_blotter_says_what_each_outcome_revealed_and_nothing_more():
    """Won: the cover. Done away: the client's side and the level. Not done
    and passed: nothing -- the side of a client who walked is not shown.
    Your bid and ask are what the engine logged as sent."""
    g, rec, final, _ = played_desk(60, 7)
    resolved = {m["data"]["rfq_ordinal"]: m["data"] for m in rec.messages if m.get("kind") == "rfq_resolved"}
    sent = {m["data"]["rfq_ordinal"]: m["data"] for m in rec.messages if m.get("kind") == "quote_sent"}
    outcomes = set()
    for row in final["blotter"]:
        d = resolved[row["ordinal"]]
        outcomes.add(d["outcome"])
        assert (row["bid"], row["ask"]) == (sent[row["ordinal"]]["bid"], sent[row["ordinal"]]["ask"])
        if d["outcome"] == "won":
            assert row["learned"] == {"cover": d["cover"]}
        elif d["outcome"] == "done_away":
            assert row["learned"] == {"side": d["side"], "level": d["cover"]}
        else:
            assert row["learned"] is None and row["revealed_side"] is None
    assert outcomes == {"won", "done_away", "not_done", "passed"}
    assert [r["ordinal"] for r in final["blotter"]] == sorted(resolved, reverse=True)


def test_the_snapshot_prices_the_venue_as_a_hedge_pays():
    """The risk strip's venue cost is what a hedge of that size actually
    pays at that step, near a print and away from one, and the near-print
    flag matches the venue's widening window. A snapshot cost from its own
    formula would drift from the venue's and mislead every hedge."""
    cfg = GameConfig.for_session(60)
    e = int(world.generate(cfg, 7).path.event_steps[0])
    for step in (100, e - 300, e + 20):
        s = game.Game(cfg, 7, storage=MemoryStorage(), settings=Settings(preset=60)).session
        s.start()
        while s.step < step:
            s.handle(agents.advance(step))
            if s.pending is not None:
                s.handle(agents.pass_(s.pending))
        venue = s.state()["venue"]
        assert venue["near_print"] == (abs(step - e) * cfg.dt < cfg.hedge_print_window_s)
        quoted = {q["size"]: q["cost_bp_per_lot"] for q in venue["quotes"]}
        for size in protocol.VENUE_QUOTE_SIZES:
            fill = s.handle(agents.hedge(size))
            executed = next(m for m in fill if m.get("kind") == "hedge_executed")["data"]
            assert executed["cost_bp_per_lot"] == quoted[size] and executed["quantity"] == size


@pytest.mark.parametrize("preset,seed,quit_minute", [(30, 3, 20), (60, 7, None), (120, 9, None)])
def test_pnl_rows_add_up_to_the_top_bar_pnl_at_every_step(preset, seed, quit_minute):
    """After every snapshot the page receives, the P&L tab's rows -- every
    trade, hedges and the close included, each marked to the screen showing
    now -- add up to the top bar's P&L to the cent; each row's edge and
    move since are q x (screen then - price) and q x (screen now - screen
    then), recomputed from the public screen; the tiles and the footer
    they make agree with the rows; and the P&L chart's series, point by
    point, is everything held marked at the screen then. A row left out, a
    trade marked to the wrong screen, or an edge and a move that did not
    split the total exactly would each show up as a gap at some step."""
    cfg = GameConfig.for_session(preset)
    g = game.Game(cfg, seed, storage=MemoryStorage(), settings=Settings(preset=preset, pricing_timer_s=15.0))
    rec = Recording(g)
    quit_at = None if quit_minute is None else round(quit_minute * 60 / cfg.dt)
    agents.play(rec, agents.ScriptedPlayer(cfg, BENCH, quit_at_step=quit_at, let_timer_expire_every=11))
    view = g._world.path.public
    usd = cfg.dollars_per_bp_per_lot
    d = desk_module.Desk()
    states = 0
    for m in rec.messages:
        d.feed(m)
        if m["type"] != "state":
            continue
        states += 1
        pnl = d.message()["pnl"]
        rows, tiles = pnl["rows"], pnl["tiles"]
        total = m["book"]["total_usd"]
        assert sum(r["total_usd"] for r in rows) == pytest.approx(total, abs=0.005), m["step"]
        assert tiles["total_usd"] == pytest.approx(total, abs=0.005)
        assert tiles["realised_usd"] + tiles["unrealised_usd"] == pytest.approx(total, abs=0.005)
        assert sum(r["edge_usd"] for r in rows) == pytest.approx(tiles["edge_client_usd"] + tiles["venue_charge_usd"], abs=1e-6)
        assert sum(r["move_usd"] for r in rows) == pytest.approx(tiles["move_usd"], abs=1e-6)
        now = view.screen_at(m["step"])
        for r in rows:
            then = view.screen_at(r["step"])
            assert r["screen_then"] == pytest.approx(then, abs=1e-9)
            assert r["edge_usd"] == pytest.approx(r["quantity"] * (then - r["price"]) * usd, abs=1e-6)
            assert r["move_usd"] == pytest.approx(r["quantity"] * (now - then) * usd, abs=1e-6)
        if pnl["series"]["step"]:
            assert pnl["series"]["pnl_usd"][-1] == pytest.approx(total, abs=0.005)
    trades = g.session.book.trades
    assert [(r["step"], r["quantity"], r["price"]) for r in rows] == [(t.step, t.quantity, t.price) for t in trades]
    assert {r["kind"] for r in rows} >= {"client", "hedge"} and states > 20
    # The series: at every screen reading the page was shown and every trade,
    # all held, marked at the public screen then -- values from the book's
    # trades and the public view, timing from the stream.
    level = cfg.start_level
    expected, position, cash = [], 0, 0.0
    book = iter(trades)
    for m in rec.messages:
        if m["type"] != "event":
            continue
        k, data = m["step"], m["data"]
        traded = (m["kind"] == "rfq_resolved" and data["outcome"] == "won") or (
            m["kind"] in ("hedge_executed", "close_liquidation") and data["quantity"])
        if traded:
            t = next(book)
            assert (t.step, t.quantity) == (k, data["quantity"])
            position, cash = position + t.quantity, cash - t.quantity * (t.price - level)
        if traded or m["kind"] == "screen_update":
            expected.append((k, (cash + position * (view.screen_at(k) - level)) * usd, position))
    series = pnl["series"]
    got = list(zip(series["step"], series["pnl_usd"], series["position"]))
    assert len(got) == len(expected)
    for (s1, p1, q1), (s2, p2, q2) in zip(got, expected):
        assert (s1, q1) == (s2, q2) and p1 == pytest.approx(p2, abs=1e-6)


def test_a_random_seed_that_lands_on_a_played_world_is_drawn_again(tmp_path):
    """Without --seed the host draws a seed; if the draw lands on a (config,
    seed) the track record has already played, it draws again, so a scored
    session is never on a world the player has seen. Forced here: the first
    two draws are the played seed. Another preset's config with the same
    seed is a different world and is accepted."""
    cfg = GameConfig.for_session(30)
    played = game.Game(cfg, 41, storage=DirectoryStorage(tmp_path), settings=Settings(preset=30))
    game.run_headless(played, agents.ScriptedPlayer(cfg, BENCH))  # plays and records seed 41
    track = records.TrackRecord.load(DirectoryStorage(tmp_path))
    assert track.played(cfg, 41)

    draws = iter([41, 41, 42])
    again = server.new_host(30, None, records_dir=str(tmp_path), draw=lambda: next(draws))
    assert again.game._world.seed == 42 and again.game.replay_of_played_seed is False
    other = server.new_host(60, None, records_dir=str(tmp_path), draw=lambda: 41)
    assert other.game._world.seed == 41 and other.game.replay_of_played_seed is False
    explicit = server.new_host(30, 41, records_dir=str(tmp_path))
    assert explicit.game.replay_of_played_seed is True  # --seed replays it, flagged
    with pytest.raises(RuntimeError):
        game.fresh_seed(cfg, track, lambda: 41, attempts=5)
