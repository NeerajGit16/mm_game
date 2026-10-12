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
from pathlib import Path

import numpy as np
import pytest

from mm_game import agents, game, handbook, records, reveal, world
from mm_game.book import Source, Trade
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
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
    prints = [int(e) for e in path.event_steps]
    ends = sorted(prints + [m.end_step for m in path.move_list])
    window = round(300 / cfg.dt)
    for t in you:
        if t.source is not Source.CLIENT:
            continue
        q = w.schedule.rfqs[t.rfq_ordinal]
        if q.informed and q.informed_about is not None:
            # An informed fill: through its next print or move end, else the close.
            end = next((e for e in ends if e > t.step), path.n_steps)
        else:
            # Any other: 300 s, never through a print's jump or into a move's ramp.
            end = min(
                [min(t.step + window, path.n_steps)]
                + [e - 1 for e in prints if e > t.step]
                + [max(m.start_step, t.step) for m in path.move_list if m.end_step > t.step]
            )
        out.append(("ticket", t.rfq_ordinal, t.step, t.quantity * (float(path.values[end]) - t.price) * usd))
    return out


def test_the_moments_are_the_three_largest_by_the_stated_rule(played):
    """The three moments are the three candidates -- every print, every
    move, every single ticket -- with the largest absolute P&L, worked out
    independently: a print as the position carried into it times its jump,
    a move as its increments times the position held over its ramp, an
    informed ticket against the truth at its next event, any other ticket
    as its spread plus its 300 s mark-out stopped short of a print or a
    move. Each says what the oracle did. Every candidate's P&L is checked,
    not only the three chosen. A rule that ranked by signed P&L, or looked
    only at prints, or counted a move's diffusion as the move, or ran an
    uninformed ticket through the print, would pick different moments."""
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


CLIENT_SPLIT = {"informed_rfqs", "informed_fills", "markouts_by_fill_usd"}


@pytest.mark.parametrize("name,version", [("protocol2-format1-config2", 1), ("protocol2-replay1-record2-track1-config2", 2)])
def test_an_old_format_record_opens_in_the_current_layout(name, version):
    """A coaching record saved under format 1 (before the reveal block) or
    2 (before the clients' split by ticket) loads as the current format:
    rebuilt from what it carries (header, settings, configs, actions) with
    today's definitions -- the reveal block, the split -- and with exactly
    the scores, decomposition, events and per-client numbers it was saved
    with. Its summary.md stays as it was written. A loader that refused it,
    rebuilt a different session, or rewrote what the player read at the
    time, strands or falsifies a real session."""
    storage = DirectoryStorage(FIXTURES / name)
    base = "sessions/30m-3-0001"
    old = json.loads(storage.read_text(f"{base}/coaching_record.json"))
    summary = storage.read_text(f"{base}/summary.md")
    assert old["format_version"] == version and ("reveal" in old) == (version >= 2)
    new = records.load_coaching_record(old)
    assert new["format_version"] == records.RECORD_FORMAT_VERSION and "reveal" in new
    for key in ("scores", "decomposition", "capture_ratio", "benchmarks", "rfqs", "events"):
        assert new[key] == old[key], key
    assert {k: v for k, v in new["diagnostics"].items() if k != "clients"} == {
        k: v for k, v in old["diagnostics"].items() if k != "clients"}
    for cid, c in old["diagnostics"]["clients"].items():
        rebuilt = new["diagnostics"]["clients"][cid]
        assert {k: v for k, v in rebuilt.items() if k not in CLIENT_SPLIT} == c, cid
        assert set(rebuilt) - set(c) == CLIENT_SPLIT
    if version == 2:
        for key in ("truth", "screen", "prints", "moves", "fills", "positions", "bots", "rules"):
            assert new["reveal"][key] == old["reveal"][key], key
    assert storage.read_text(f"{base}/summary.md") == summary
    assert records.markdown_summary(new) != summary  # today's columns, in a summary nobody stored


def test_a_ticket_moment_stops_where_the_rule_says():
    """An informed fill runs through its next event; any other fill runs
    300 s but stops on the step before a print lands, at the start of a
    move's ramp, or where it is when it was filled inside a ramp. Each case
    against a hand-placed fill on a real world, so the rule is checked where
    the session happened not to put a fill."""
    cfg = GameConfig.for_session(60)
    w = next(
        w for w in (world.generate(cfg, s) for s in range(1, 300))
        if w.path.move_list and w.path.event_steps.size
        and any(q.informed and q.informed_about is not None for q in w.schedule.rfqs)
        and w.path.move_list[0].start_step > 4000 and int(w.path.event_steps[0]) > 4000
    )
    plain = next(q for q in w.schedule.rfqs if not q.informed)
    knew = next(q for q in w.schedule.rfqs if q.informed and q.informed_about is not None)
    window = round(300 / cfg.dt)
    e = int(w.path.event_steps[0])
    m = w.path.move_list[0]

    def fill(step, q):
        return Trade(0, step, Source.CLIENT, 10, w.path.public.screen_at(step), q.ordinal, 0.0, 10)

    assert reveal.ticket_end(w, fill(e - 1000, plain)) == (e - 1, "print", 0)
    assert reveal.ticket_end(w, fill(m.start_step - 1000, plain))[:2] == (m.start_step, "move")
    assert reveal.ticket_end(w, fill(m.start_step + 1, plain))[:2] == (m.start_step + 1, "move")
    far = next(s for s in range(0, cfg.n_steps - window)
               if all(not (s < x <= s + window + 1) for x in w.path.event_steps)
               and all(not (s < mv.end_step and mv.start_step <= s + window) for mv in w.path.move_list))
    assert reveal.ticket_end(w, fill(far, plain)) == (far + window, "horizon", None)
    ev = w.path.next_event_markout(e - 1000)
    assert reveal.ticket_end(w, fill(e - 1000, knew)) == (ev.event_step, ev.kind, ev.index)


FIRST_REAL = Path(__file__).resolve().parent / "fixtures" / "play" / "30m-1314533384-0001"


@pytest.fixture(scope="module")
def first_real():
    """The first session played for real (30 minutes), rebuilt from its replay file."""
    rf = records.load_replay(json.loads((FIRST_REAL / "replay.json").read_text(encoding="utf-8")))
    session, w = records.replay(rf)
    bots = {b: agents.run_bot(b, w, settings=rf.settings, scoring=rf.scoring, benchmark=rf.benchmark) for b in agents.BOTS}
    record = records.coaching_record(w, session, scoring=rf.scoring, benchmark=rf.benchmark, bots=bots, flags=rf.flags)
    return w, session, record


def summary_clients(md: str) -> dict[str, dict[str, str]]:
    """summary.md's clients table, row by client, cell by header."""
    lines = md.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| Client |"))
    heads = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = {}
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows[cells[0]] = dict(zip(heads, cells))
    return rows


def test_a_real_sessions_informed_clients_show_no_informed_fills(first_real):
    """In the first real session clients 1 and 3 are informed types, but every
    fill he did with them was on an RFQ outside their information window:
    their informed RFQs show "(0)" filled, the informed-fill mark-out is
    empty, and all of their next-print/move mark-out is under other fills.
    Lumped by client type, as before, it read as if informed clients had
    picked the player off when there were no informed fills at all."""
    _, _, record = first_real
    clients = record["diagnostics"]["clients"]
    md = summary_clients(records.markdown_summary(record))
    for cid in ("1", "3"):
        c = clients[cid]
        assert c["informed_type"] and c["informed_rfqs"] > 0 and c["won"] > 0
        assert c["informed_fills"] == 0
        assert c["markouts_by_fill_usd"]["informed"] == {"60s": 0.0, "300s": 0.0, "next_event": 0.0}
        assert c["markouts_by_fill_usd"]["other"]["next_event"] == pytest.approx(c["markouts_usd"]["next_event"]) != 0
        row = md[cid]
        assert row["Informed RFQs (filled)"] == f"{c['informed_rfqs']} (0)"
        assert row["Mark-out to next print/move, informed fills"] == "—"
        assert row["Mark-out to next print/move, other fills"] == records._signed(c["markouts_usd"]["next_event"])
    assert sum(c["informed_fills"] for c in clients.values()) == 0


def test_summary_md_splits_each_clients_mark_out_by_ticket():
    """summary.md's clients table prints, for each client, its informed
    RFQs and how many were filled, and the next-print/move mark-out of its
    informed fills and of its other fills -- each from its own part of the
    split, never the client's total. Checked on a session (30 minutes, seed
    8) where a client has both kinds of fill, so each part differs from the
    total."""
    cfg = GameConfig.for_session(30)
    g = game.Game(cfg, 8, storage=MemoryStorage(), settings=Settings(preset=30))
    game.run_headless(g, agents.ScriptedPlayer(cfg, BENCH))
    clients = g.record["diagnostics"]["clients"]
    rows = summary_clients(g.summary)
    mixed = [cid for cid, c in clients.items() if c["informed_fills"] and c["won"] > c["informed_fills"]]
    assert mixed, "the session should have a client with informed and other fills"
    for cid, c in clients.items():
        row, split = rows[cid], c["markouts_by_fill_usd"]
        assert row["Informed RFQs (filled)"] == f"{c['informed_rfqs']} ({c['informed_fills']})"
        informed = records._signed(split["informed"]["next_event"]) if c["informed_fills"] else "—"
        other = records._signed(split["other"]["next_event"]) if c["won"] - c["informed_fills"] else "—"
        assert row["Mark-out to next print/move, informed fills"] == informed, cid
        assert row["Mark-out to next print/move, other fills"] == other, cid


def test_a_real_sessions_1752_sale_is_not_scored_through_the_print(first_real):
    """The 34-lot sale to client 5 at 17:52, 38 seconds before print 1, was
    uninformed: its ticket moment is its spread plus its mark-out up to the
    step before the print, not through it. Through the print it read as a
    $4,726 loss -- the print's own move, already the print moment's -- and
    was one of the three moments; it is neither now."""
    w, session, record = first_real
    dt = w.config.dt
    t = next(t for t in session.book.trades if t.source is Source.CLIENT and t.rfq_ordinal == 34)
    assert (t.quantity, w.schedule.rfqs[34].client_id, f"{int(t.step * dt) // 60}:{int(t.step * dt) % 60}") == (-34, 5, "17:52")
    e = int(w.path.event_steps[0])
    assert t.step < e < t.step + round(300 / dt)
    assert reveal.ticket_end(w, t) == (e - 1, "print", 0)
    usd = w.config.dollars_per_bp_per_lot
    through = t.quantity * (float(w.path.values[e]) - t.price) * usd
    assert through == pytest.approx(-4726, abs=1)
    oracle = agents.run_bot("oracle", w, settings=session.settings, scoring=ScoringConfig(), benchmark=BENCH)
    cand = next(c for c in reveal.candidates(w, list(session.book.trades), list(oracle.book.trades))
                if c["kind"] == "ticket" and c["index"] == 34)
    assert cand["pnl_usd"] == pytest.approx(t.quantity * (float(w.path.values[e - 1]) - t.price) * usd)
    assert cand["pnl_usd"] > 0 > through
    assert ("ticket", 34) not in {(m["kind"], m["index"]) for m in record["reveal"]["moments"]}


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
