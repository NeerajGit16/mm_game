"""Tests for unannounced moves: their generation and the flow that foretells them.

A move is print-sized, lasts 30-60 seconds, and is known to informed clients a
few minutes ahead; nothing public announces it. The failures that matter are
silent: a move leaking onto the calendar or the gauge, a move draw shifting a
print or the screen, thinning that reshuffles other candidates, a move window
overlapping a print run-up so one inquiry is informed about two things, or a
rate the player never actually sees. Each docstring says which.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from mm_game import agents, fair_value, flow, world
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings
from mm_game.fair_value import PublicView, generate
from mm_game.rng import RngStreams, Stream
from mm_game.scoring import print_risk

CFG = GameConfig()  # the 120-minute preset: two moves per free hour


def moves_world(seed_from: int = 0, cfg: GameConfig = CFG, need: int = 1) -> world.World:
    return next(w for w in (world.generate(cfg, s) for s in range(seed_from, seed_from + 200)) if len(w.path.move_list) >= need)


def candidate_draws(cfg: GameConfig, seed: int, candidate: int) -> tuple[float, float, int, int]:
    """A candidate's own draws, straight from its substream in the documented
    order -- placement uniform, sign, magnitude, duration, lead -- as
    (sign, magnitude, duration_steps, lead_steps)."""
    gen = RngStreams(seed).substream(Stream.MOVES, candidate + 1)
    gen.random()
    sign = 1.0 if gen.random() < 0.5 else -1.0
    magnitude = float(gen.uniform(cfg.move_magnitude_min_bp, cfg.move_magnitude_max_bp))
    duration = max(1, round(float(gen.uniform(cfg.move_duration_min_s, cfg.move_duration_max_s)) / cfg.dt))
    lead = round(float(gen.uniform(cfg.move_lead_min_s, cfg.move_lead_max_s)) / cfg.dt)
    return sign, magnitude, duration, lead


# --------------------------------------------------------------------------
# Hidden: never on the public view, the calendar or the gauge
# --------------------------------------------------------------------------


def test_moves_never_appear_in_the_public_view_the_calendar_or_the_gauge():
    """The public view carries exactly the screen and the print calendar;
    its calendar is the prints only; the screen gets no forced observation
    at a move (nothing happens at a single step); the session's calendar
    message and the print-risk gauge name prints and nothing else, before,
    during and after every move. A move on the calendar or the gauge would
    hand the player the headline the game says is never shown."""
    assert {f.name for f in dataclasses.fields(PublicView)} == {
        "n_steps", "dt", "refresh_steps", "event_steps", "screen_steps", "screen"
    }
    w = moves_world(need=2)
    path, view = w.path, w.path.public
    # The prints, from a source the calendar is not built from: each print's
    # run-up ends at it, one run-up per jump.
    prints = sorted(win.end_step for win in path.drift_windows)
    assert len(prints) == len(path.jump_sizes)
    assert [int(e) for e in view.event_steps] == prints
    move_steps = {x for m in path.move_list for x in (m.inform_step, m.start_step, m.end_step)}
    assert not move_steps & set(prints)
    r = view.refresh_steps
    forced = {int(s) for s in view.screen_steps if s % r}
    assert forced <= set(prints)
    s = Session(w.public, w.resolve, settings=Settings(), scoring=ScoringConfig())
    start = s.start()[0]
    assert start["data"]["print_steps"] == prints
    for m in path.move_list:
        for step in (m.inform_step, m.start_step, (m.start_step + m.end_step) // 2, m.end_step):
            gauge = print_risk(w.config, ScoringConfig(), prints, step, 50)
            later = [e for e in prints if e > step]
            assert (gauge is None) == (not later)
            assert gauge is None or gauge["step"] == later[0]


# --------------------------------------------------------------------------
# Generation: deterministic thinning, no other stream moved
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_thinning_is_deterministic_and_never_moves_another_candidates_draws(seed):
    """Generating twice gives the same moves. Every kept move's sign,
    magnitude, duration and lead are exactly its own candidate's draws,
    whatever was kept before it. Raising the rate only ever adds candidates,
    and every candidate common to both rates keeps its draws. Drawn from one
    shared stream instead, dropping a candidate would shift every later
    move -- a calendar change would re-roll the whole session's moves."""
    a, b = generate(CFG, RngStreams(seed)), generate(CFG, RngStreams(seed))
    assert a.move_list == b.move_list
    # At ten times the rate free time saturates: later candidates find no
    # room and are dropped, and every kept one must still carry its own draws.
    for cfg in (CFG, dataclasses.replace(CFG, move_rate_per_free_hour=10 * CFG.move_rate_per_free_hour)):
        p = generate(cfg, RngStreams(seed))
        for m in p.move_list:
            sign, magnitude, duration, lead = candidate_draws(cfg, seed, m.candidate)
            assert m.total_bp == sign * magnitude
            assert (m.end_step - m.start_step, m.start_step - m.inform_step) == (duration, lead)
    small = {m.candidate for m in a.move_list}
    assert max(small, default=-1) <= max((m.candidate for m in p.move_list), default=-1)


def test_thinning_drops_candidates_mid_sequence_so_the_draw_check_has_teeth():
    """The check above only bites if some candidate is dropped while a later
    one is kept: keyed on kept count instead of candidate number, the two
    coincide whenever the kept candidates are 0, 1, 2, ... At ten times the
    rate, a long window that fits nowhere is followed by a shorter one that
    fits, in some of these seeds."""
    busy = dataclasses.replace(CFG, move_rate_per_free_hour=10 * CFG.move_rate_per_free_hour)
    gaps = 0
    for seed in range(8):
        kept = sorted(m.candidate for m in generate(busy, RngStreams(seed)).move_list)
        gaps += kept != list(range(len(kept)))
    assert gaps >= 1


def test_raising_the_rate_only_adds_candidates():
    """The candidate count is an inverse-CDF Poisson on one uniform, so for a
    fixed seed it is monotone in the rate."""
    for u in np.linspace(0.01, 0.99, 25):
        counts = [fair_value._poisson_count(lam, float(u)) for lam in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert counts == sorted(counts)


@pytest.mark.parametrize("seed", range(10))
def test_no_print_jump_drift_or_screen_draw_changes_when_moves_are_on(seed):
    """Turn moves on: the calendar, every jump, every drift window, the
    diffusion, the screen's observation steps and its error are bit for bit
    what they were, and the truth differs by exactly the moves array. A move
    drawn from a print or screen stream, or consuming a draw before them,
    would move every session's prints and screen with the move rate."""
    off = generate(dataclasses.replace(CFG, move_rate_per_free_hour=0.0), RngStreams(seed))
    on = generate(CFG, RngStreams(seed))
    for field in ("event_steps", "jump_sizes", "diffusion", "drift", "screen_error"):
        np.testing.assert_array_equal(getattr(on, field), getattr(off, field))
    np.testing.assert_array_equal(on.public.screen_steps, off.public.screen_steps)
    assert on.drift_windows == off.drift_windows
    assert not off.move_list and not off.moves.any()
    np.testing.assert_allclose(on.relative - off.relative, on.moves, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Placement: free time only, and the rate the player sees
# --------------------------------------------------------------------------


@pytest.mark.parametrize("minutes", [60, 120])
def test_every_informing_window_lies_entirely_in_free_time(minutes):
    """Each move's whole informing window [start - lead, end) sits inside
    free time -- clear of the first and last five minutes, of every print's
    run-up and the five minutes after it -- and at least the minimum gap from
    every other move's window. The ramp ends at the window's end, so the move
    happens entirely in free time too."""
    cfg = GameConfig.for_session(minutes)
    dt = cfg.dt
    gap = round(cfg.move_min_gap_s / dt)
    edge = round(cfg.move_edge_margin_s / dt)
    clearance = round(cfg.move_print_clearance_s / dt)
    checked = 0
    for seed in range(150):
        p = generate(cfg, RngStreams(seed))
        blocked = [(w.start_step, w.end_step + clearance) for w in p.drift_windows]  # run-up to print + clearance
        for i, m in enumerate(p.move_list):
            assert m.inform_step < m.start_step < m.end_step
            assert edge <= m.inform_step and m.end_step <= cfg.n_steps - edge
            for lo, hi in blocked:
                assert m.end_step <= lo or hi <= m.inform_step, (seed, m, lo, hi)
            for k in p.move_list[i + 1 :]:
                assert k.inform_step >= m.end_step + gap
            checked += 1
    assert checked > 100


@pytest.mark.parametrize("minutes,rate", [(60, 3.0), (120, 2.0), (120, 1.0)])
def test_realised_moves_track_the_rate_times_free_hours(minutes, rate):
    """Over 300 sessions the realised count is close to rate x free hours:
    a kept move takes its window and the gap out of free time for the next
    candidate rather than thinning it, so blocking does not eat the rate.
    Measured 85-101%. A placement that dropped colliding candidates instead
    lost a third of the rate, and a player would see far fewer moves than
    the config says."""
    cfg = GameConfig.for_session(minutes, move_rate_per_free_hour=rate)
    realised, expected = [], []
    for seed in range(300):
        p = generate(cfg, RngStreams(seed))
        free = fair_value.free_intervals(cfg, p.drift_windows, p.event_steps)
        realised.append(len(p.move_list))
        expected.append(rate * sum(b - a for a, b in free) * cfg.dt / 3600)
    ratio = np.mean(realised) / np.mean(expected)
    assert 0.8 < ratio < 1.08, ratio


def test_presets_set_the_rate_and_the_drill_has_no_moves():
    """Rate per preset: 3 per free hour at 60 minutes (about one move a
    session), 2 at 120, none in the 30-minute event drill, which stays a
    clean single-print drill."""
    assert [GameConfig.for_session(m).move_rate_per_free_hour for m in (30, 60, 120)] == [0.0, 3.0, 2.0]
    drill = GameConfig.for_session(30)
    assert all(not generate(drill, RngStreams(s)).move_list for s in range(30))
    n60 = np.mean([len(generate(GameConfig.for_session(60), RngStreams(s)).move_list) for s in range(200)])
    assert 0.75 < n60 < 1.25


# --------------------------------------------------------------------------
# Flow: informed about one thing at a time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("minutes", [60, 120])
def test_each_request_is_informed_about_at_most_one_thing(minutes):
    """No step is inside both a print's run-up and a move's informing window,
    so an RFQ can be informed about one thing at most; an informed-type
    client inside a move window is informed about that move, with the
    move's own index; inside a run-up, about that print; anywhere else, not
    at all. Overlapping windows would leave the fill's lesson ambiguous --
    which piece of news was it trading?"""
    cfg = GameConfig.for_session(minutes)
    move_informed = 0
    for seed in range(60):
        p = generate(cfg, RngStreams(seed))
        sched = flow.generate(cfg, RngStreams(seed), p)
        types = {c.client_id: c.informed_type for c in sched.clients}
        for q in sched.rfqs:
            in_print = [w for w in p.drift_windows if w.start_step <= q.step < w.end_step]
            in_move = [m for m in p.move_list if m.inform_step <= q.step < m.end_step]
            assert not (in_print and in_move)
            assert len(in_move) <= 1
            if in_move and types[q.client_id]:
                assert q.informed and q.informed_about.kind == "move" and q.informed_about.index == in_move[0].index
                move_informed += 1
            elif in_print and types[q.client_id]:
                assert q.informed and q.informed_about.kind == "print"
            else:
                assert not q.informed and q.informed_about is None
    assert move_informed > 30


def test_move_signals_are_one_view_per_client_and_move_and_right_eighty_percent():
    """Every inquiry a client sends about one move carries the same view,
    drawn from the move stream keyed on (client, candidate); the view's sign
    is right about 80% of the time, like a print's. Keyed on the kept-move
    index instead, an earlier candidate being dropped would change every
    later client's view."""
    cfg = GameConfig()
    views, right = {}, []
    for seed in range(150):
        p = generate(cfg, RngStreams(seed))
        sched = flow.generate(cfg, RngStreams(seed), p)
        for q in sched.rfqs:
            if q.informed and q.informed_about.kind == "move":
                key = (seed, q.client_id, q.informed_about.index)
                if key in views:
                    assert views[key] == q.signal_bp
                    continue
                views[key] = q.signal_bp
                m = p.move_list[q.informed_about.index]
                right.append((q.signal_bp > 0) == (m.total_bp > 0))
                gen = RngStreams(seed).substream(Stream.FLOW_INFORMED_MOVE, flow._pair_key(q.client_id, m.candidate))
                correct = gen.random() < cfg.informed_accuracy
                assert (q.signal_bp > 0) == ((m.total_bp > 0) == correct)
    assert len(right) > 300
    assert abs(np.mean(right) - cfg.informed_accuracy) < 0.06


def test_move_informed_clients_are_in_the_oracles_set():
    """The flow oracle is told an RFQ is informed whatever it is informed
    about -- and nothing more."""
    w = moves_world(need=2)
    move_ordinals = {q.ordinal for q in w.schedule.rfqs if q.informed and q.informed_about.kind == "move"}
    assert move_ordinals and move_ordinals <= w.informed_ordinals()
    bot = agents.make_bot("oracle", w, BenchmarkConfig())
    assert bot.informed == w.informed_ordinals()


def test_the_crowd_follows_a_move_only_once_the_screen_shows_it():
    """Uninformed clients tilt 60/40 with a move during its ramp -- the crowd
    follows what the screen shows -- and not at all during the lead, when
    only the informed know. A crowd that tilted during the lead would be
    informed by the back door, and the one-sided flow before a move would
    stop being a read on the informed clients. Measured at six times the
    usual RFQ rate so the short ramps hold enough inquiries."""
    cfg = GameConfig(rfq_rate_per_hour=600.0)
    ramp, lead = [], []
    for seed in range(120):
        p = generate(cfg, RngStreams(seed))
        sched = flow.generate(cfg, RngStreams(seed), p)
        for q in sched.rfqs:
            if q.informed:
                continue
            for m in p.move_list:
                with_move = (int(q.side) > 0) == (m.total_bp > 0)
                if m.start_step <= q.step < m.end_step:
                    ramp.append(with_move)
                elif m.inform_step <= q.step < m.start_step:
                    lead.append(with_move)
    assert len(ramp) > 1000 and len(lead) > 3000
    assert abs(np.mean(ramp) - cfg.crowd_drift_tilt) < 0.04
    assert abs(np.mean(lead) - 0.5) < 0.03
