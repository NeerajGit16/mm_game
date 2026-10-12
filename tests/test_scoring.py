"""Tests for scoring: the risk penalty, the P&L decomposition, mark-outs and
the capture ratio. Each docstring says what silent failure it catches.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mm_game import agents, records, scoring, world
from mm_game.book import Source, Trade
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings
from mm_game.events import Kind
from mm_game.fair_value import sigma_per_step

SCORING = ScoringConfig()
BENCH = BenchmarkConfig()


def types_of(w) -> dict[int, str]:
    return {q.ordinal: scoring.fill_type(q) for q in w.schedule.rfqs}


def decompose(cfg, w, trades):
    return scoring.decomposition(cfg, w.path, w.path.public.screen_at, trades, types_of(w))


def played(cfg: GameConfig, seed: int, bot: str = "naive", quit_at: int | None = None):
    w = world.generate(cfg, seed)
    s = Session(w.public, w.resolve, settings=Settings(), scoring=SCORING)
    strategy = (
        agents.ScriptedPlayer(cfg, BENCH, quit_at_step=quit_at)
        if bot == "scripted"
        else agents.make_bot(bot, w, BENCH)
    )
    agents.play(s, strategy)
    return w, s


SESSIONS = [
    (GameConfig.for_session(30), 1, "naive", None),
    (GameConfig.for_session(60), 2, "disciplined", None),
    (GameConfig(), 3, "oracle", None),
    (GameConfig(), 6, "passing_oracle", None),
    (GameConfig(), 4, "scripted", None),
    (GameConfig.for_session(60), 5, "scripted", 15000),
]


# --------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cfg,seed,bot,quit_at", SESSIONS)
def test_decomposition_sums_to_the_cent(cfg, seed, bot, quit_at):
    """One sign convention: every line is a signed P&L contribution and the
    lines add up. Spread captured + inventory + hedging and closing equals
    the session's P&L to the cent, every term computed independently; at
    every mark-out horizon spread + adverse selection + inventory beyond the
    mark-outs + hedging and closing equals it too. Catches a trade left out
    of one term, a sign slip, a cost booked against the screen in one place
    and the truth in another -- each moves the total by real money while
    every line still looks plausible. Inventory is cross-checked against the
    position path integrated over the truth's moves, a different route to
    the same number."""
    w, s = played(cfg, seed, bot, quit_at)
    dec = decompose(cfg, w, s.book.trades)
    pnl = s.log.of(Kind.SESSION_END)[0].data["pnl_usd"]
    cent = 0.005
    assert dec["total_usd"] == pytest.approx(pnl, abs=cent)
    assert dec["spread_captured_usd"] + dec["inventory_usd"] + dec["hedging_and_closing_usd"] == pytest.approx(
        pnl, abs=cent
    )
    for key, h in dec["by_horizon"].items():
        line = (
            dec["spread_captured_usd"] + h["adverse_selection_usd"] + h["inventory_beyond_usd"]
            + dec["hedging_and_closing_usd"]
        )
        assert line == pytest.approx(pnl, abs=cent), key
    held = scoring.position_path(s.book.trades, cfg.n_steps)
    integrated = float(np.sum(held * np.diff(w.path.relative))) * cfg.dollars_per_bp_per_lot
    assert integrated == pytest.approx(dec["inventory_usd"], abs=cent)
    assert dec["venue_charge_usd"] + dec["screen_error_on_hedges_usd"] == pytest.approx(
        dec["hedging_and_closing_usd"], abs=cent
    )
    if any(t.source is not Source.CLIENT for t in s.book.trades):
        assert dec["venue_charge_usd"] < 0  # a charge is a negative contribution, never a positive "cost"


@pytest.mark.parametrize("cfg,seed,bot,quit_at", SESSIONS)
def test_adverse_selection_by_type_adds_up_to_the_line_at_every_horizon(cfg, seed, bot, quit_at):
    """The per-type adverse-selection lines -- uninformed, informed about a
    print, informed about a move -- sum to the decomposition's adverse
    selection at every horizon, and the diagnostics' per-type mark-outs are
    the same numbers. A fill counted under two types, dropped from all, or
    excluded in one place and not the other would make the split disagree
    with the line it explains, while each figure still looked plausible."""
    w, s = played(cfg, seed, bot, quit_at)
    dec = decompose(cfg, w, s.book.trades)
    diag = scoring.diagnostics(w, s.log, s.book.trades)
    for key, h in dec["by_horizon"].items():
        parts = h["adverse_selection_by_type_usd"]
        assert set(parts) == set(scoring.FILL_TYPES)
        assert sum(parts.values()) == pytest.approx(h["adverse_selection_usd"], abs=0.005), key
        for t in scoring.FILL_TYPES:
            bucket = diag["flow"][t]["markouts"].get(key)
            assert (bucket["pnl_usd"] if bucket else 0.0) == pytest.approx(parts[t], abs=0.005), (key, t)


def test_spread_captured_is_edge_against_the_truth_at_the_fill():
    """Spread captured counts client fills only, each at q * (truth at the
    fill - price); hedges and the close are costs, never spread. Mixing the
    two -- or marking fills at a later truth -- puts inventory into edge,
    and a split by counterparty marked to a later truth has exactly that flaw."""
    cfg = GameConfig()
    w, s = played(cfg, 7, "disciplined")
    expected = sum(
        t.quantity * (w.path.values[t.step] - t.price) for t in s.book.trades if t.source is Source.CLIENT
    ) * cfg.dollars_per_bp_per_lot
    dec = decompose(cfg, w, s.book.trades)
    assert dec["spread_captured_usd"] == pytest.approx(expected, abs=1e-6)
    assert any(t.source is Source.HEDGE for t in s.book.trades)


# --------------------------------------------------------------------------
# Mark-outs
# --------------------------------------------------------------------------


def test_truncated_and_undefined_markouts_are_flagged_and_excluded():
    """A fill 50 s before the close has truncated 60 s and 300 s mark-outs and
    is left out of that horizon's adverse selection and mark-out totals. A
    fill after the last print but before a move ends marks through the move;
    one at or after the last move's end, with no print after it either, has
    no next-event mark-out -- None, not zero -- and is left out of that one.
    Averaging truncated values in makes late fills look harmless; a zero for
    "nothing followed" drags the next-event average toward nothing; a
    horizon that ignored moves would drop every fill after the last print."""
    cfg = GameConfig()
    n = cfg.n_steps
    w = next(
        w for w in (world.generate(cfg, s) for s in range(100))
        if w.path.move_list and w.path.event_steps.size
        and w.path.move_list[-1].inform_step > w.path.event_steps[-1] and w.path.event_steps[-1] < n - 500
    )
    last = w.path.move_list[-1]
    at = w.path.public.screen_at
    trades = [
        Trade(0, 100, Source.CLIENT, 10, at(100), 0, 0.0, 10),
        Trade(1, n - 500, Source.CLIENT, -10, at(n - 500), 1, 0.0, 0),
        Trade(2, last.inform_step, Source.CLIENT, 5, at(last.inform_step), 2, 0.0, 5),
        Trade(3, last.end_step, Source.CLIENT, -5, at(last.end_step), 3, 0.0, 0),
        Trade(4, n, Source.CLOSE, 0, at(n), None, 1.0, 0),
    ]
    marks = scoring.fill_markouts(cfg, w.path, trades)
    ne = scoring.NEXT_EVENT
    assert marks[0]["300s"]["truncated"] is False and marks[1]["300s"]["truncated"] is True
    assert marks[1]["10s"]["truncated"] is False
    assert marks[0][ne]["pnl_bp"] is not None
    assert marks[2][ne]["through"] == {"kind": "move", "index": last.index, "step": last.end_step}
    rel = w.path.relative
    assert marks[2][ne]["pnl_bp"] == 5 * (rel[last.end_step] - rel[last.inform_step])
    assert marks[1][ne] == {"pnl_bp": None, "through": None} and marks[3][ne] == {"pnl_bp": None, "through": None}
    types = {i: "uninformed" for i in range(4)}
    dec = scoring.decomposition(cfg, w.path, w.path.public.screen_at, trades, types)
    assert dec["by_horizon"]["300s"]["fills_excluded"] == 1
    assert dec["by_horizon"]["10s"]["fills_excluded"] == 0
    assert dec["by_horizon"][ne]["fills_excluded"] == 2
    expected_300 = sum(marks[i]["300s"]["pnl_bp"] for i in (0, 2, 3))
    assert dec["by_horizon"]["300s"]["adverse_selection_usd"] == pytest.approx(expected_300 * cfg.dollars_per_bp_per_lot)
    expected_next = marks[0][ne]["pnl_bp"] + marks[2][ne]["pnl_bp"]
    assert dec["by_horizon"][ne]["adverse_selection_usd"] == pytest.approx(expected_next * cfg.dollars_per_bp_per_lot)


@pytest.mark.parametrize("cfg,seed,bot,quit_at", SESSIONS)
def test_client_table_adds_up_to_the_decomposition(cfg, seed, bot, quit_at):
    """The per-client mark-outs sum, at 60 s, 300 s and the next event, to
    the decomposition's adverse selection at that horizon, with the same
    fills excluded; your net lots per client -- your side, positive when you
    bought -- sum to your net client fills. A client table on its own
    horizon definition, or with the client's side instead of yours, reads
    plausibly row by row and contradicts the decomposition above it."""
    w, s = played(cfg, seed, bot, quit_at)
    dec = decompose(cfg, w, s.book.trades)
    clients = scoring.diagnostics(w, s.log, s.book.trades)["clients"].values()
    for key in ("60s", "300s", scoring.NEXT_EVENT):
        h = dec["by_horizon"][key]
        assert sum(c["markouts_usd"][key] for c in clients) == pytest.approx(h["adverse_selection_usd"], abs=0.005), key
        assert sum(c["markouts_excluded"][key] for c in clients) == h["fills_excluded"], key
    fills = [t for t in s.book.trades if t.source is Source.CLIENT]
    assert fills and sum(c["net_lots"] for c in clients) == sum(t.quantity for t in fills)
    assert sum(c["lots"] for c in clients) == sum(abs(t.quantity) for t in fills)


@pytest.mark.parametrize("cfg,seed,bot,quit_at", SESSIONS)
def test_client_markouts_split_by_ticket_not_by_client_type(cfg, seed, bot, quit_at):
    """Each client's mark-outs split into its informed fills and the rest by
    the RFQ, not the client: informed + other is the client's total at every
    horizon; summed over clients, the informed part is the decomposition's
    informed-print plus informed-move lines and the rest its uninformed line.
    Its informed RFQs are those inside its information window, counted among
    the RFQs you were shown, and its informed fills are the ones you filled.
    An informed client's uninformed RFQs land in "other". A split by client
    type would charge an informed client's column for flow that knew nothing
    -- the mistake that made one session look picked off with no informed
    fills at all."""
    w, s = played(cfg, seed, bot, quit_at)
    dec = decompose(cfg, w, s.book.trades)
    clients = scoring.diagnostics(w, s.log, s.book.trades)["clients"]
    rfqs = w.schedule.rfqs
    informed = lambda q: q.informed and q.informed_about is not None  # noqa: E731
    shown = [e.data["rfq"]["ordinal"] for e in s.log.of(Kind.RFQ_ARRIVED)]
    fills = [t for t in s.book.trades if t.source is Source.CLIENT]
    for cid, c in clients.items():
        mine = [o for o in shown if rfqs[o].client_id == int(cid)]
        assert c["informed_rfqs"] == sum(informed(rfqs[o]) for o in mine)
        assert c["informed_fills"] == sum(informed(rfqs[t.rfq_ordinal]) for t in fills if rfqs[t.rfq_ordinal].client_id == int(cid))
        for key in ("60s", "300s", scoring.NEXT_EVENT):
            split = c["markouts_by_fill_usd"]
            assert split["informed"][key] + split["other"][key] == pytest.approx(c["markouts_usd"][key], abs=1e-6), (cid, key)
            if c["informed_fills"] == 0:
                assert split["informed"][key] == 0.0
    for key in ("60s", "300s", scoring.NEXT_EVENT):
        by_type = dec["by_horizon"][key]["adverse_selection_by_type_usd"]
        got = sum(c["markouts_by_fill_usd"]["informed"][key] for c in clients.values())
        assert got == pytest.approx(by_type["informed_print"] + by_type["informed_move"], abs=0.005), key
        got = sum(c["markouts_by_fill_usd"]["other"][key] for c in clients.values())
        assert got == pytest.approx(by_type["uninformed"], abs=0.005), key
    # The case the split exists for: an informed client asking outside its window.
    outside = [t for t in fills if w.schedule.clients[rfqs[t.rfq_ordinal].client_id].informed_type and not informed(rfqs[t.rfq_ordinal])]
    if bot == "scripted" and quit_at is None:
        assert outside, "the full scripted session should include an informed client's uninformed fill"


# --------------------------------------------------------------------------
# The risk penalty
# --------------------------------------------------------------------------


def test_kappa_calibration_matches_the_carry_target():
    """At the default kappa a print costs about $0.5 * q^2 -- $1,250 for 50
    lots, $5,000 for 100 -- and the marginal penalty at 50 lots, about $50 a
    lot, matches the 1bp hedge cost: the best carry into a print is ~50.
    Catches a kappa or a units slip ($ vs bp-lots, q vs q^2) that would make
    the risk-adjusted score reward or punish the wrong carry."""
    cfg = GameConfig()
    pen = lambda q: scoring.print_penalty_usd(cfg, SCORING, q)
    assert pen(50) == pytest.approx(1250, rel=0.03)
    assert pen(100) == pytest.approx(5000, rel=0.03)
    assert pen(-100) == pen(100)
    marginal = pen(51) - pen(50)
    hedge_cost = cfg.hedge_cost_bp * cfg.dollars_per_bp_per_lot
    assert marginal == pytest.approx(hedge_cost, rel=0.05)
    assert scoring.print_variance_bp2(cfg) == pytest.approx(28 / 3)


@pytest.mark.parametrize("cfg,seed,bot,quit_at", SESSIONS)
def test_live_risk_meter_equals_the_penalty_rebuilt_from_the_blotter(cfg, seed, bot, quit_at):
    """The penalty shown during play and the one recomputed afterwards from
    the trades alone agree -- including after a quit, when the position
    keeps accruing through the remaining prints. If they disagreed, either
    the live gauge lies or re-scoring an old session gives a different
    number from the one the player saw."""
    w, s = played(cfg, seed, bot, quit_at)
    live = s.log.of(Kind.SESSION_END)[0].data["risk_penalty_usd"]
    rebuilt = scoring.ex_ante_penalty_usd(cfg, SCORING, w.path.event_steps, s.book.trades)
    assert rebuilt["total_usd"] == pytest.approx(live, rel=1e-9, abs=1e-6)
    if quit_at is not None and (w.path.event_steps > quit_at).any():
        assert rebuilt["prints_usd"] > 0


def test_print_penalty_charges_the_position_held_through_the_print():
    """The print lands on the increment into its step: a hedge one step
    before escapes the print's penalty, a hedge at the print step does not
    -- the screen already shows the print by then. Off by one step and the
    gauge would reward hedging the instant after the news."""
    cfg = GameConfig()
    w = world.generate(cfg, 3)
    e = int(w.path.event_steps[0])
    p = w.path.public.screen_at(0)

    def penalty(hedge_step):
        trades = [
            Trade(0, 0, Source.CLIENT, 100, p, 0, 0.0, 100),
            Trade(1, hedge_step, Source.HEDGE, -100, p, None, 1.0, 0),
        ]
        return scoring.ex_ante_penalty_usd(cfg, SCORING, [e], trades)

    before, at = penalty(e - 1), penalty(e)
    assert before["prints_usd"] == 0.0
    assert at["prints_usd"] == pytest.approx(scoring.print_penalty_usd(cfg, SCORING, 100))
    sigma2 = sigma_per_step(cfg) ** 2
    assert at["diffusion_usd"] == pytest.approx(
        SCORING.risk_kappa * (cfg.dollars_per_bp_per_lot * 100) ** 2 * sigma2 * e
    )


@pytest.mark.parametrize("offset,charged", [(0, True), (-1, False)])
def test_live_meter_charges_the_print_to_the_position_held_into_it(offset, charged):
    """In play: hold 100 lots, then hedge flat exactly at the print step
    (too late -- the print has landed) or one step before it (in time). The
    live penalty charges the print in the first case and not the second,
    and matches the penalty rebuilt from the blotter in both. Bots rarely
    trade on a print step, so a meter that charged the position after the
    step's actions would agree with the rebuild in every ordinary session."""
    cfg = GameConfig()
    w = world.generate(cfg, 3)
    e = int(w.path.event_steps[0])
    s = Session(w.public, w.resolve, settings=Settings(), scoring=SCORING)
    s.start()
    plan = {e // 2: 100, e + offset: -100}
    for target in sorted(plan) + [cfg.n_steps]:
        while s.step < target and not s.ended:
            s.handle(agents.advance(target))
            if s.pending is not None:
                s.handle(agents.pass_(s.pending))
        if target in plan and s.step == target:
            assert s.handle(agents.hedge(plan[target]))[-1]["book"]["position"] in (0, 100)
    live = s.log.of(Kind.SESSION_END)[0].data["risk_penalty_usd"]
    rebuilt = scoring.ex_ante_penalty_usd(cfg, SCORING, w.path.event_steps, s.book.trades)
    assert live == pytest.approx(rebuilt["total_usd"], rel=1e-12)
    assert (rebuilt["prints_usd"] > 0) is charged


def test_penalty_is_ex_ante_not_luck():
    """Same positions, prints reversed and the truth bent: identical
    penalty. It charges the risk taken, not what the prints did."""
    cfg = GameConfig()
    w, s = played(cfg, 8, "disciplined")
    other = dataclasses.replace(w.path, jump_sizes=-w.path.jump_sizes, values=w.path.values + 3.0)
    a = scoring.ex_ante_penalty_usd(cfg, SCORING, w.path.event_steps, s.book.trades)
    b = scoring.ex_ante_penalty_usd(cfg, SCORING, other.event_steps, s.book.trades)
    assert a == b


def test_rescoring_changes_the_penalty_and_nothing_else():
    """Re-score a recorded session at another kappa: the penalty scales
    with kappa exactly and P&L does not move. Scoring is a pure function of
    the stored world and the replay -- changing kappa later re-scores the
    whole history without regenerating anything."""
    cfg = GameConfig.for_session(60)
    w, s = played(cfg, 9, "scripted")
    rf = records.load_replay(records.replay_file(w, s, scoring=SCORING, benchmark=BENCH, flags=records.flags(s, False)))
    doubled = ScoringConfig(risk_kappa=2 * SCORING.risk_kappa)
    again, w2 = records.replay(rf, scoring=doubled)
    a, b = records.scores(w, s, SCORING), records.scores(w2, again, doubled)
    assert b["pnl_usd"] == a["pnl_usd"]
    assert b["risk_penalty_usd"] == pytest.approx(2 * a["risk_penalty_usd"])


def test_print_risk_gauge():
    """The live gauge names the next print, its countdown, and the penalty
    for carrying the current position through it -- from the calendar,
    which is public, and the position."""
    cfg = GameConfig()
    g = scoring.print_risk(cfg, SCORING, [1000, 5000], 400, 120)
    assert (g["print_index"], g["step"], g["in_s"]) == (0, 1000, 60.0)
    assert g["penalty_if_held_usd"] == pytest.approx(scoring.print_penalty_usd(cfg, SCORING, 120))
    assert scoring.print_risk(cfg, SCORING, [1000, 5000], 1000, 10)["print_index"] == 1
    assert scoring.print_risk(cfg, SCORING, [1000], 1000, 10) is None


# --------------------------------------------------------------------------
# Capture ratio
# --------------------------------------------------------------------------


def test_capture_ratio_and_its_guard():
    """(player - disciplined) / (oracle - disciplined), and n/a -- None --
    when the oracle's edge is under the threshold or negative. Without the
    guard a session with no informed flow divides noise by noise and posts
    a ratio of 40 to the track record."""
    assert scoring.capture_ratio(5000, 1000, 9000, 2000) == pytest.approx(0.5)
    assert scoring.capture_ratio(-3000, 1000, 9000, 2000) == pytest.approx(-0.5)
    assert scoring.capture_ratio(5000, 1000, 2500, 2000) is None
    assert scoring.capture_ratio(5000, 1000, 500, 2000) is None
