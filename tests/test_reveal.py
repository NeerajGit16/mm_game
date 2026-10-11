"""Tests for the reveal's data: the block the coaching record carries so the
page can draw the reveal from the record alone.

What can go wrong silently: a moment chosen by some other rule than the one
stated, or with the oracle's side computed from the wrong path; a path or
window that is not the session's (a move shaded from its start instead of
its leak); a fill marked informed that was not; the bots' rules on the
reveal drifting from what the bots do; an old record that no longer opens.
Each docstring says which.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mm_game import agents, game, handbook, records, reveal, world
from mm_game.book import Source, Trade
from mm_game.config import BenchmarkConfig, GameConfig
from mm_game.engine import Settings
from mm_game.storage import DirectoryStorage, MemoryStorage

from fixtures.make_record_fixture import FIXTURES
from test_handbook import check_items

BENCH = BenchmarkConfig()


@pytest.fixture(scope="module")
def played():
    """A 60-minute session with prints and unannounced moves, played by the
    scripted stand-in through to the close."""
    cfg = GameConfig.for_session(60)
    seed = next(
        s for s in range(1, 200)
        if (w := world.generate(cfg, s)).path.move_list and w.path.event_steps.size
    )
    g = game.Game(cfg, seed, storage=MemoryStorage(), settings=Settings(preset=60))
    game.run_headless(g, agents.ScriptedPlayer(cfg, BENCH))
    return g


def held_into(trades, step: int) -> int:
    """Position carried over the increment into ``step``: trades before it."""
    return sum(t.quantity for t in trades if t.step < step)


def independent_candidates(g: game.Game) -> list[dict]:
    """Every print, move and client fill with its P&L, worked out from the
    world and the blotters directly."""
    w, cfg = g._world, g.session.config
    usd = cfg.dollars_per_bp_per_lot
    path = w.path
    you = list(g.session.book.trades)
    out = []
    for i, e in enumerate(int(x) for x in path.event_steps):
        out.append(("print", i, e, held_into(you, e) * float(path.jump_sizes[i]) * usd))
    for m in path.move_list:
        pnl = sum(
            sum(t.quantity for t in you if t.step <= k) * (path.moves[k + 1] - path.moves[k])
            for k in range(m.start_step, m.end_step)
        )
        out.append(("move", m.index, m.start_step, float(pnl) * usd))
    ends = sorted([int(e) for e in path.event_steps] + [m.end_step for m in path.move_list])
    for t in you:
        if t.source is not Source.CLIENT:
            continue
        end = next((e for e in ends if e > t.step), path.n_steps)
        out.append(("ticket", t.rfq_ordinal, t.step, t.quantity * (float(path.values[end]) - t.price) * usd))
    return out


def test_the_moments_are_the_three_largest_by_the_stated_rule(played):
    """The three moments are the three candidates -- every print, every
    move, every single ticket -- with the largest absolute P&L, worked out
    independently: a print as the position carried into it times its jump,
    a move as its increments times the position held over its ramp, a
    ticket against the truth at its next event. Each says what the oracle
    did. Every candidate's P&L is checked, not only the three chosen. A
    rule that ranked by signed P&L, or looked only at prints, or counted a
    move's diffusion as the move, would pick different moments."""
    cands = independent_candidates(played)
    kinds = {c[0] for c in cands}
    assert {"print", "move", "ticket"} <= kinds
    # Every candidate, chosen or not, carries the P&L the rule defines: a
    # print or move miscounted would otherwise hide until it was big enough.
    oracle = agents.run_bot("oracle", played._world, settings=played.session.settings, scoring=played.scoring, benchmark=BENCH)
    built = {(c["kind"], c["index"]): c["pnl_usd"]
             for c in reveal.candidates(played._world, list(played.session.book.trades), list(oracle.book.trades))}
    assert set(built) == {(c[0], c[1]) for c in cands}
    for kind, index, _, pnl in cands:
        assert built[(kind, index)] == pytest.approx(pnl, abs=1e-6), (kind, index)
    order = {"print": 0, "move": 1, "ticket": 2}
    expected = sorted(cands, key=lambda c: (-abs(c[3]), c[2], order[c[0]], c[1]))[:3]
    got = played.record["reveal"]["moments"]
    assert [(m["kind"], m["index"], m["step"]) for m in got] == [(c[0], c[1], c[2]) for c in expected]
    for m, c in zip(got, expected):
        assert m["pnl_usd"] == pytest.approx(c[3], abs=1e-6)
        assert "flow oracle" in m["sentence"] and m["title"]
        assert m["from_step"] <= m["step"] <= m["to_step"]


def test_ties_go_to_the_earlier_moment():
    """Equal sizes rank by time, then print before move before ticket: the
    choice never depends on dictionary order."""
    cands = [
        {"kind": "ticket", "index": 4, "step": 50, "pnl_usd": -100.0},
        {"kind": "print", "index": 0, "step": 50, "pnl_usd": 100.0},
        {"kind": "move", "index": 0, "step": 10, "pnl_usd": 100.0},
        {"kind": "ticket", "index": 9, "step": 5, "pnl_usd": 99.0},
    ]
    assert [(c["kind"], c["step"]) for c in reveal.choose(cands)] == [("move", 10), ("print", 50), ("ticket", 50)]


def test_the_oracles_side_of_each_moment_is_from_its_own_book(played):
    """Where a moment is a print or a move, the oracle's P&L on it is worked
    out from the oracle's own trades the same way: a moment told against
    the wrong book would misstate what reading the flow was worth."""
    w = played._world
    oracle = agents.run_bot("oracle", w, settings=played.session.settings, scoring=played.scoring, benchmark=BENCH)
    o_trades = list(oracle.book.trades)
    usd = played.session.config.dollars_per_bp_per_lot
    cands = reveal.candidates(w, list(played.session.book.trades), o_trades)
    for c in cands:
        if c["kind"] == "print":
            assert c["oracle_pnl_usd"] == pytest.approx(held_into(o_trades, c["step"]) * c["size_bp"] * usd)
        elif c["kind"] == "move":
            m = w.path.move_list[c["index"]]
            expect = sum(
                sum(t.quantity for t in o_trades if t.step <= k) * (w.path.moves[k + 1] - w.path.moves[k])
                for k in range(m.start_step, m.end_step)
            ) * usd
            assert c["oracle_pnl_usd"] == pytest.approx(float(expect), abs=1e-6)


def test_the_reveal_block_is_the_sessions_paths_windows_and_fills(played):
    """The truth every second, the screen as shown, each print's run-up,
    each move from its leak to its end, every fill with its counterparty's
    type, and the three positions as step functions that end flat. A move
    shaded from its start would hide the minutes the flow foretold it; a
    fill marked informed by the client's type rather than the RFQ would
    ring the wrong tickets."""
    w, s, rv = played._world, played.session, played.record["reveal"]
    path = w.path
    every = rv["truth"]["every_steps"]
    assert every == round(1.0 / w.config.dt)
    assert rv["truth"]["values"] == pytest.approx(path.values[::every], abs=5e-5)
    assert rv["screen"]["steps"] == [int(x) for x in path.public.screen_steps]
    assert rv["screen"]["mids"] == [float(x) for x in path.public.screen]
    windows = {d.event_index: d for d in path.drift_windows}
    for p in rv["prints"]:
        assert p["step"] == int(path.event_steps[p["index"]]) and p["jump_bp"] == float(path.jump_sizes[p["index"]])
        assert (p["runup"]["start_step"], p["runup"]["end_step"]) == (windows[p["index"]].start_step, windows[p["index"]].end_step)
    assert [(m["inform_step"], m["start_step"], m["end_step"]) for m in rv["moves"]] == [
        (m.inform_step, m.start_step, m.end_step) for m in path.move_list]
    trades = list(s.book.trades)
    assert [(f["step"], f["quantity"], f["price"], f["source"]) for f in rv["fills"]] == [
        (t.step, t.quantity, t.price, t.source.value) for t in trades]
    for f in rv["fills"]:
        if f["source"] == "client":
            q = w.schedule.rfqs[f["rfq_ordinal"]]
            assert (f["fill_type"] != "uninformed") == q.informed
    for who, book in (("you", trades), *((b, None) for b in ("disciplined", "oracle"))):
        pos = rv["positions"][who]
        assert pos["positions"][0] == 0 and pos["positions"][-1] == 0
        assert pos["steps"] == sorted(pos["steps"])
    bot_fills = rv["bots"]["oracle"]["fills"]
    assert sum(f["quantity"] for f in bot_fills) == 0 and bot_fills
    shown = {r["ordinal"] for r in played.record["rfqs"] if r["shown"]}
    assert {r["rfq_ordinal"] for r in rv["bots"]["oracle"]["rfqs"]} == shown


@pytest.mark.parametrize("bench", [BENCH, BenchmarkConfig(
    disciplined_half_width_bp=(1.25, 0.75, 0.5, 0.25), disciplined_size_widen_bp=0.5, disciplined_pre_print_widen_bp=0.75,
    disciplined_pre_print_window_s=900.0, disciplined_print_widen_bp=0.5, disciplined_print_window_s=60.0,
    disciplined_skew_bp_per_100_lots=1.0, disciplined_print_carry_lots=40, disciplined_hedge_lead_s=240.0,
    disciplined_inventory_cap_lots=150, disciplined_inventory_target_lots=120, disciplined_close_lead_s=300.0,
    oracle_informed_widen_bp=0.75, oracle_lean_bp=0.5, oracle_lean_s=600.0)])
def test_the_bots_rules_on_the_reveal_are_their_config(bench):
    """The disciplined bot's rules -- shown on every reveal, as the handbook
    promises -- and the flow oracle's are generated from the benchmark
    config, number by number, and follow it when it changes. A rule typed
    into the text would describe a bot that no longer exists."""
    cfg = GameConfig()
    expected = {
        **{f"w{k}": w for k, w in enumerate(bench.disciplined_half_width_bp)},
        "size": bench.disciplined_size_widen_bp, "per": 100,
        "pre": bench.disciplined_pre_print_widen_bp, "pre_window": bench.disciplined_pre_print_window_s / 60,
        "at_print": bench.disciplined_print_widen_bp, "at_window": bench.disciplined_print_window_s / 60,
        "lead": bench.disciplined_hedge_lead_s / 60, "carry": bench.disciplined_print_carry_lots,
        "skew": bench.disciplined_skew_bp_per_100_lots,
        "cap": bench.disciplined_inventory_cap_lots, "target": bench.disciplined_inventory_target_lots,
        "close": bench.disciplined_close_lead_s / 60,
        "widen": bench.oracle_informed_widen_bp, "lean": bench.oracle_lean_bp, "lean_min": bench.oracle_lean_s / 60,
    }
    items = handbook.disciplined_rules(bench, cfg) + handbook.oracle_rules(bench)
    assert check_items(items, expected) >= 14
    text = " ".join(i.text for i in items)
    assert ("within" in text) == (bench.disciplined_print_widen_bp > 0)


def test_an_old_format_record_opens_in_the_current_layout():
    """A coaching record saved under format 1 -- before the reveal block --
    loads as the current format: rebuilt from what it carries (header,
    settings, configs, actions), with the reveal block, and with exactly
    the scores, decomposition and summary it was saved with. A loader that
    refused it, or rebuilt a different session, strands a real session."""
    storage = DirectoryStorage(FIXTURES / "protocol2-format1-config2")
    base = "sessions/30m-3-0001"
    old = json.loads(storage.read_text(f"{base}/coaching_record.json"))
    assert old["format_version"] == 1 and "reveal" not in old
    new = records.load_coaching_record(old)
    assert new["format_version"] == records.RECORD_FORMAT_VERSION and "reveal" in new
    for key in ("scores", "decomposition", "diagnostics", "capture_ratio", "benchmarks", "rfqs", "events"):
        assert new[key] == old[key], key
    assert records.markdown_summary(new) == storage.read_text(f"{base}/summary.md")


def test_a_print_counts_the_position_carried_into_it_not_a_hedge_on_its_step(played):
    """Hedging flat on the print step itself is too late -- the print lands
    on the step's own increment, as the risk penalty has it -- so the print
    moment counts the 50 lots carried in. Counting the position after the
    step would credit a hedge the market had already beaten."""
    w = played._world
    e = int(w.path.event_steps[0])
    price = w.path.public.screen_at(e - 100)
    trades = [
        Trade(0, e - 100, Source.HEDGE, 50, price, None, 1.0, 50),
        Trade(1, e, Source.HEDGE, -50, price, None, 1.0, 0),
    ]
    usd = w.config.dollars_per_bp_per_lot
    first = next(c for c in reveal.candidates(w, trades, trades) if c["kind"] == "print" and c["index"] == 0)
    assert first["you_lots"] == 50
    assert first["pnl_usd"] == pytest.approx(50 * float(w.path.jump_sizes[0]) * usd)
