"""Tests for the benchmark players.

Bots must use only what their role allows -- the public messages, plus the
informed flags for the oracle and nothing else -- and must behave as their
definitions say, or the capture ratio compares the player against something
other than the benchmark it names. Each docstring says what silent failure
it catches.
"""

from __future__ import annotations

import inspect

import pytest

from mm_game import agents, world
from mm_game.book import Source
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Settings
from mm_game.events import Kind

BENCH = BenchmarkConfig()
SCORING = ScoringConfig()


def run(name: str, w: world.World):
    return agents.run_bot(name, w, settings=Settings(), scoring=SCORING, benchmark=BENCH)


def test_bots_are_given_public_information_only():
    """The naive and disciplined bots are built from the config and their
    parameters; the oracle additionally from a set of RFQ ordinals -- the
    informed flags -- and nothing else: no world, no path, no schedule. A
    "disciplined" bot handed the world could read the truth by accident,
    and every capture ratio after that would be measured against a cheat."""
    for cls in (agents.Naive, agents.Disciplined):
        assert list(inspect.signature(cls.__init__).parameters) == ["self", "config", "benchmark"]
    for cls in (agents.FlowOracle, agents.PassingOracle):
        assert list(inspect.signature(cls.__init__).parameters) == ["self", "config", "benchmark", "informed"]
    assert list(inspect.signature(agents.Trend.__init__).parameters) == ["self", "config", "benchmark", "window_s", "beta"]
    w = world.generate(GameConfig(), 3)
    for name in ("oracle", "passing_oracle"):
        oracle = agents.make_bot(name, w, BENCH)
        assert oracle.informed == w.informed_ordinals() and isinstance(oracle.informed, frozenset)
        assert all(isinstance(i, int) for i in oracle.informed)
    # Informed about moves as well as prints, and told nothing about which.
    kinds = {q.informed_about.kind for q in w.schedule.rfqs if q.informed}
    assert w.informed_ordinals() == frozenset(q.ordinal for q in w.schedule.rfqs if q.informed) and kinds


def test_naive_never_hedges_and_quotes_a_fixed_width():
    w = world.generate(GameConfig(), 4)
    s = run("naive", w)
    assert not any(t.source is Source.HEDGE for t in s.book.trades)
    widths = {round(e.data["ask"] - e.data["bid"], 9) for e in s.log.of(Kind.QUOTE_SENT) if e.data["bid"] is not None and e.data["ask"] is not None}
    assert widths == {2 * BENCH.naive_half_width_bp}


def test_passing_oracle_passes_exactly_the_informed_rfqs_and_otherwise_quotes_like_the_disciplined_bot():
    """The passing oracle (a diagnostic) declines every informed RFQ it is
    shown and no other until the closing lead; on an RFQ outside any run-up
    to a print, with the same position, it quotes exactly what the
    disciplined bot quotes. An oracle that also knew the truth or the print
    would show up as quotes the disciplined bot never makes."""
    w = world.generate(GameConfig(), 6)
    s = run("passing_oracle", w)
    close_lead = w.config.n_steps - round(BENCH.disciplined_close_lead_s / w.config.dt)
    informed = w.informed_ordinals()
    assert informed
    for e in s.log.of(Kind.QUOTE_SENT):
        o = e.data["rfq_ordinal"]
        passed = e.data["bid"] is None and e.data["ask"] is None
        assert passed == (o in informed or e.step >= close_lead)
    d = agents.Disciplined(w.config, BENCH)
    o = agents.PassingOracle(w.config, BENCH, informed)
    for x in (d, o):
        x.on_start({"config": w.config.to_record(), "print_steps": [10_000]}, {})
    ticket = {"ordinal": 999, "size": 30, "in_comp": 2}
    state = {"step": 100, "screen": {"mid": 9550.0}, "book": {"position": 40}}
    assert o.answer(ticket, state) == d.answer(ticket, state)


def test_flow_oracle_quotes_informed_wide_and_leans_with_what_it_learns():
    """The flow oracle quotes every informed RFQ (it does not pass them),
    wider than the disciplined bot by the configured amount; with no lean
    active and no informed RFQ it quotes exactly what the disciplined bot
    does outside a run-up; once a client's side is known it leans its mid
    that way for the configured time and then stops. An oracle that knew
    which way an RFQ was going before the outcome showed it would lean
    without a logged reason."""
    cfg = GameConfig()
    w = world.generate(cfg, 6)
    informed = w.informed_ordinals()
    s = run("oracle", w)
    close_lead = cfg.n_steps - round(BENCH.disciplined_close_lead_s / cfg.dt)
    quoted_informed = [
        e for e in s.log.of(Kind.QUOTE_SENT) if e.data["rfq_ordinal"] in informed and e.step < close_lead
    ]
    assert quoted_informed and all(e.data["bid"] is not None or e.data["ask"] is not None for e in quoted_informed)

    o = agents.FlowOracle(cfg, BENCH, frozenset({7}))
    d = agents.Disciplined(cfg, BENCH)
    for x in (o, d):
        x.on_start({"config": cfg.to_record(), "print_steps": [60_000]}, {})
    state = {"step": 100, "screen": {"mid": 9550.0}, "book": {"position": 0}}
    plain = {"ordinal": 3, "size": 20, "in_comp": 2, "side": None}
    assert o.answer(plain, state) == d.answer(plain, state)
    wide = o.answer({**plain, "ordinal": 7}, state)
    base = d.answer({**plain, "ordinal": 7}, state)
    assert wide["ask"] - wide["bid"] == pytest.approx(base["ask"] - base["bid"] + 2 * BENCH.oracle_informed_widen_bp)
    # The outcome shows the client bought: lean up for oracle_lean_s.
    o.observe({"type": "event", "kind": "rfq_resolved", "step": 100, "data": {"rfq_ordinal": 7, "side": 1}})
    later = {**state, "step": 100 + round(BENCH.oracle_lean_s / cfg.dt) - 1}
    leaned = o.answer(plain, later)
    assert leaned["bid"] == pytest.approx(d.answer(plain, later)["bid"] + BENCH.oracle_lean_bp)
    expired = {**state, "step": 100 + round(BENCH.oracle_lean_s / cfg.dt) + 1}
    assert o.answer(plain, expired) == d.answer(plain, expired)
    assert o.leans == [(100, 1, 100 + round(BENCH.oracle_lean_s / cfg.dt))]


HEDGE_SEEDS = range(12)


def oracle_session(seed: int):
    cfg = GameConfig()
    w = world.generate(cfg, seed)
    session = agents.Session(w.public, w.resolve, settings=Settings(), scoring=SCORING)
    bot = agents.FlowOracle(cfg, BENCH, w.informed_ordinals())
    agents.play(session, bot)
    return session, bot


@pytest.mark.parametrize("seed", HEDGE_SEEDS)
def test_every_flow_oracle_hedge_cuts_its_position_and_never_flips_it(seed):
    """Every venue trade the flow oracle asks for, and every one it gets,
    reduces |position| without taking it through zero, and it is flat
    before the close, so the closing liquidation trades nothing. Its hedges
    are the disciplined bot's -- the inventory cap, carry into a print, flat
    before the close -- every one risk-reducing: a hedge that added to the
    position or turned it round would be the benchmark trading a view on
    the venue. The next test checks the cases this covers occurred."""
    session, _ = oracle_session(seed)
    for e in session.log.of(Kind.HEDGE_EXECUTED):
        before = e.data["position_after"] - e.data["quantity"]
        for after in (before + e.data["requested"], e.data["position_after"]):
            assert abs(after) < abs(before) and after * before >= 0, (e.step, e.data)
    assert session.log.of(Kind.CLOSE_LIQUIDATION)[0].data["quantity"] == 0


def test_the_hedge_check_had_cases_including_hedges_the_way_of_a_lean():
    """Across the checked sessions the oracle hedged for every reason -- the
    cap, carry into a print, flat before the close -- and some of those
    hedges went the way of a lean active at the time: cutting inventory it
    had taken against its own lean. So the check above was not passed
    vacuously; and a rule that held such hedges back -- leaving the oracle
    carrying risk the disciplined bot would have cut -- fails here."""
    cfg = GameConfig()
    dt = cfg.dt
    kinds = {"cap": 0, "carry": 0, "close": 0}
    with_lean = 0
    for seed in HEDGE_SEEDS:
        session, bot = oracle_session(seed)
        for e in session.log.of(Kind.HEDGE_EXECUTED):
            if e.step == bot.close_wake:
                kinds["close"] += 1
            elif e.step in bot.print_wakes:
                kinds["carry"] += 1
            else:
                kinds["cap"] += 1
            active = [d for start, d, until in bot.leans if start <= e.step <= until]
            with_lean += bool(active) and (e.data["requested"] > 0) == (active[-1] > 0)
    assert all(kinds.values()), kinds
    assert with_lean > 0
    assert bot.close_wake == cfg.n_steps - round(BENCH.disciplined_close_lead_s / dt)


def test_disciplined_hedges_toward_the_carry_before_prints_and_flattens_before_the_close():
    """Before every print it carries at most the configured carry, hedged
    outside the venue's widening window; it is flat going into the close
    and pays no closing liquidation. Without the pre-print hedge the
    "disciplined" bot would carry whatever flow left it through every
    print, and the risk-adjusted benchmark would mean nothing."""
    cfg = GameConfig()
    for seed in range(5):
        w = world.generate(cfg, seed)
        s = run("disciplined", w)
        close = s.log.of(Kind.CLOSE_LIQUIDATION)[0]
        assert close.data["quantity"] == 0
        lead = round(BENCH.disciplined_hedge_lead_s / cfg.dt)
        for e in (int(x) for x in w.path.event_steps):
            at_hedge = [t for t in s.book.trades if t.step <= e - lead]
            position = at_hedge[-1].position_after if at_hedge else 0
            assert abs(position) <= max(BENCH.disciplined_print_carry_lots, 0) or e - lead <= 0


def test_disciplined_widens_with_size_thin_competition_and_prints_and_skews_against_inventory():
    cfg = GameConfig()
    d = agents.Disciplined(cfg, BENCH)
    d.on_start({"config": cfg.to_record(), "print_steps": [20_000]}, {})
    calm = {"step": 100, "screen": {"mid": 9550.0}, "book": {"position": 0}}
    hw = lambda **t: d.half_width({"ordinal": 0, "size": 10, "in_comp": 3, **t}, calm)
    assert hw(size=80) > hw(size=10)
    assert hw(in_comp=0) > hw(in_comp=3)
    near = {**calm, "step": 20_000 - 3000}
    assert d.half_width({"ordinal": 0, "size": 10, "in_comp": 3}, near) > hw()
    long = {**calm, "book": {"position": 120}}
    q_flat = d.answer({"ordinal": 0, "size": 10, "in_comp": 3}, calm)
    q_long = d.answer({"ordinal": 0, "size": 10, "in_comp": 3}, long)
    assert q_long["bid"] < q_flat["bid"] and q_long["ask"] < q_flat["ask"]


@pytest.mark.parametrize("name", agents.BOTS)
def test_bots_are_deterministic(name):
    w = world.generate(GameConfig.for_session(60), 2)
    a, b = run(name, w), run(name, w)
    assert a.log.to_record() == b.log.to_record() and a.actions == b.actions
