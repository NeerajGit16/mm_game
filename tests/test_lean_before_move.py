"""Tests for the lean before each unannounced move: the mean of (quote mid -
screen) over the two-sided quotes from the leak to the start of the ramp,
signed + toward the move, yours with the flow oracle's and the disciplined
bot's beside it.

What can go wrong silently: the sign taken from the price rather than the
move (a lean into a down move read as against it), the window shifted by a
step at either end, a one-sided quote's lone price counted as a mid, the
bots' figures taken from your quotes or the wrong bot's, a move you had
stopped before shown as 0.00 instead of "—". Each docstring says which.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mm_game import agents, game, records, scoring
from mm_game.config import BenchmarkConfig, GameConfig
from mm_game.engine import Settings
from mm_game.events import Kind
from mm_game.storage import MemoryStorage

BENCH = BenchmarkConfig()
QUIT_STEP = 15_000


@pytest.fixture(scope="module")
def two_moves():
    """60 minutes, seed 2: no prints and two unannounced moves, both up. The
    scripted player quits at step 15000 -- after move 0 has run (its ramp
    ends at 12494) and before move 1 leaks (22333) -- while the bots play
    the whole session."""
    cfg = GameConfig.for_session(60)
    g = game.Game(cfg, 2, storage=MemoryStorage(), settings=Settings(preset=60))
    game.run_headless(g, agents.ScriptedPlayer(cfg, BENCH, quit_at_step=QUIT_STEP))
    return g


def quotes_in(session, move, view) -> dict[int, tuple]:
    """Each quote in the move's lead window: RFQ -> (bid, ask, screen)."""
    return {
        e.data["rfq_ordinal"]: (e.data["bid"], e.data["ask"], view.screen_at(e.step))
        for e in session.log.of(Kind.QUOTE_SENT)
        if move.inform_step <= e.step < move.start_step
    }


def test_the_lean_before_each_move_matches_a_hand_working(two_moves):
    """Move 0 (+2.82bp, leak at step 10637, ramp at 11950) had three two-way
    RFQs in its lead window. Quote mid - screen, + toward the (upward) move:

        you          31: 9549.875 - 9550.00 = -0.125   32: 9550.625 - 9550.75 = -0.125   33: 9550.75 - 9550.75 = 0
                     -> -0.25 / 3 = -0.0833 over 3 quotes
        flow oracle  31: 9549.25 - 9550.00 = -0.75     32: 9550.00 - 9550.75 = -0.75     33: 9550.125 - 9550.75 = -0.625
                     -> -2.125 / 3 = -0.7083 over 3
        disciplined  the oracle's very quotes (no informed RFQ to lean on) -> -0.7083 over 3

    Move 1 (+2.81bp, 22333 -> 24306): you had stopped, so no quotes -- "—",
    not 0.00. The bots quoted RFQs 72-77, the same until RFQ 76 (informed,
    which the oracle quoted wider round the same mid) taught the oracle a
    side; at 77 it leaned +0.25 where the disciplined bot sat on the screen:

        flow oracle  -0.125, -0.125, 0, +0.125, 0, +0.25 -> +0.125 / 6 = +0.0208
        disciplined  -0.125, -0.125, 0, +0.125, 0, 0     -> -0.125 / 6 = -0.0208

    Each figure is checked against the record, and the quotes behind it
    against the sessions' own logs."""
    g = two_moves
    w = g._world
    view = w.path.public
    m0, m1 = w.path.move_list
    assert (m0.inform_step, m0.start_step, m1.inform_step, m1.start_step) == (10637, 11950, 22333, 24306)
    assert m0.end_step < QUIT_STEP < m1.inform_step and m0.total_bp > 0 and m1.total_bp > 0
    oracle = agents.run_bot("oracle", w, settings=g.session.settings, scoring=g.scoring, benchmark=BENCH)
    disciplined = agents.run_bot("disciplined", w, settings=g.session.settings, scoring=g.scoring, benchmark=BENCH)
    assert quotes_in(g.session, m0, view) == {
        31: (9549.25, 9550.5, 9550.0), 32: (9550.0, 9551.25, 9550.75), 33: (9550.25, 9551.25, 9550.75)}
    assert quotes_in(oracle, m0, view) == quotes_in(disciplined, m0, view) == {
        31: (9548.75, 9549.75, 9550.0), 32: (9549.5, 9550.5, 9550.75), 33: (9549.5, 9550.75, 9550.75)}
    assert quotes_in(g.session, m1, view) == {}
    assert quotes_in(oracle, m1, view) == {
        72: (9552.75, 9554.0, 9553.5), 73: (9552.5, 9553.75, 9553.25), 74: (9552.75, 9553.75, 9553.25),
        75: (9552.5, 9553.75, 9553.0), 76: (9553.0, 9555.0, 9554.0), 77: (9553.25, 9554.25, 9553.5)}
    assert quotes_in(disciplined, m1, view) == {
        **quotes_in(oracle, m1, view), 76: (9553.5, 9554.5, 9554.0), 77: (9553.0, 9554.0, 9553.5)}

    moves = g.record["diagnostics"]["position"]["moves"]
    bots = g.record["benchmarks"]
    assert moves[0]["lean_before"] == {"mean_bp": pytest.approx(-0.25 / 3), "quotes": 3}
    assert moves[1]["lean_before"] == {"mean_bp": None, "quotes": 0}
    assert bots["oracle"]["summary"]["lean_before_moves"] == [
        {"mean_bp": pytest.approx(-2.125 / 3), "quotes": 3}, {"mean_bp": pytest.approx(0.125 / 6), "quotes": 6}]
    assert bots["disciplined"]["summary"]["lean_before_moves"] == [
        {"mean_bp": pytest.approx(-2.125 / 3), "quotes": 3}, {"mean_bp": pytest.approx(-0.125 / 6), "quotes": 6}]


def test_summary_md_prints_each_lean_beside_the_bots(two_moves):
    """summary.md's line for each move ends with the three leans as the
    hand-working has them, "—" for the move you had stopped before, and the
    rule is stated once above the moves."""
    md = two_moves.summary
    assert records.LEAN_RULE in md
    assert (
        "Lean before the move: you −0.08bp over 3 quotes; flow oracle −0.71bp over 3 quotes; "
        "disciplined bot −0.71bp over 3 quotes." in md
    )
    assert (
        "Lean before the move: you — (no two-sided quotes); flow oracle +0.02bp over 6 quotes; "
        "disciplined bot −0.02bp over 6 quotes." in md
    )


def _row(step: int, bid, ask) -> dict:
    return {"step": step, "quote": {"bid": bid, "ask": ask}}


def test_the_lean_is_signed_toward_the_move_and_counts_only_its_window():
    """On made-up quotes against a flat 100.00 screen. A move down: a quote
    mid 0.5bp under the screen leans toward it, +0.50; one 0.25bp over leans
    against it, -0.25. The window is the leak's step up to, not including,
    the ramp's first: a quote on the leak step counts, one on the ramp step
    and one before the leak do not; a one-sided quote never counts. The same
    quotes against a move up read the other way. A sign taken from anything
    but the move, or a window off by a step, changes these numbers."""
    down = SimpleNamespace(inform_step=100, start_step=200, total_bp=-3.0)
    up = SimpleNamespace(inform_step=100, start_step=200, total_bp=+3.0)
    table = {
        0: _row(99, 99.0, 100.0),  # before the leak
        1: _row(100, 99.0, 100.0),  # mid 99.50: 0.50 under the screen, on the leak step
        2: _row(150, 99.75, 100.75),  # mid 100.25
        3: _row(160, None, 99.0),  # one-sided
        4: _row(200, 90.0, 91.0),  # on the ramp's first step
        5: {"step": 170},  # never quoted
    }

    def screen(step):
        return 100.0

    assert scoring.lean_before_move(table, screen, down) == {"mean_bp": pytest.approx((0.5 - 0.25) / 2), "quotes": 2}
    assert scoring.lean_before_move(table, screen, up) == {"mean_bp": pytest.approx((-0.5 + 0.25) / 2), "quotes": 2}
    assert scoring.lean_before_move({5: {"step": 170}}, screen, up) == {"mean_bp": None, "quotes": 0}
