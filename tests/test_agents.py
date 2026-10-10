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
    assert list(inspect.signature(agents.FlowOracle.__init__).parameters) == ["self", "config", "benchmark", "informed"]
    w = world.generate(GameConfig(), 3)
    oracle = agents.make_bot("oracle", w, BENCH)
    assert oracle.informed == w.informed_ordinals() and isinstance(oracle.informed, frozenset)
    assert all(isinstance(i, int) for i in oracle.informed)


def test_naive_never_hedges_and_quotes_a_fixed_width():
    w = world.generate(GameConfig(), 4)
    s = run("naive", w)
    assert not any(t.source is Source.HEDGE for t in s.book.trades)
    widths = {round(e.data["ask"] - e.data["bid"], 9) for e in s.log.of(Kind.QUOTE_SENT) if e.data["bid"] is not None and e.data["ask"] is not None}
    assert widths == {2 * BENCH.naive_half_width_bp}


def test_oracle_passes_exactly_the_informed_rfqs_and_otherwise_quotes_like_the_disciplined_bot():
    """The oracle declines every informed RFQ it is shown and no other until
    the closing lead; on an RFQ outside any run-up to a print, with the same
    position, it quotes exactly what the disciplined bot quotes. An oracle
    that also knew the truth or the print would show up as quotes the
    disciplined bot never makes."""
    w = world.generate(GameConfig(), 6)
    s = run("oracle", w)
    close_lead = w.config.n_steps - round(BENCH.disciplined_close_lead_s / w.config.dt)
    informed = w.informed_ordinals()
    assert informed
    for e in s.log.of(Kind.QUOTE_SENT):
        o = e.data["rfq_ordinal"]
        passed = e.data["bid"] is None and e.data["ask"] is None
        assert passed == (o in informed or e.step >= close_lead)
    d = agents.Disciplined(w.config, BENCH)
    o = agents.FlowOracle(w.config, BENCH, informed)
    for x in (d, o):
        x.on_start({"config": w.config.to_record(), "print_steps": [10_000]}, {})
    ticket = {"ordinal": 999, "size": 30, "in_comp": 2}
    state = {"step": 100, "screen": {"mid": 9550.0}, "book": {"position": 40}}
    assert o.answer(ticket, state) == d.answer(ticket, state)


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
