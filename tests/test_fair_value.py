"""Tests for mm_game.fair_value.

The ones that matter most, in order: independence from flow (the path must not
move when the number of RFQs changes), jump-time stability under a changed
size distribution, the vol round-trip (the test that catches a conversion
wrong by sqrt(252) or by a factor of dt), mark-out truncation, and start_level
invariance. Each docstring says what silent failure it catches.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game.config import GameConfig
from mm_game.fair_value import (
    DriftWindow,
    EventMarkOut,
    FairValuePath,
    PublicView,
    _drift_increments,
    generate,
    session_sigma,
    sigma_per_step,
)
from mm_game.rng import RngStreams, Stream

SRC_DIR = Path(mm_game.__file__).resolve().parent

# A short session for the many-seed statistical tests. 6,000 steps instead of
# 72,000 keeps a 600-seed loop around a second; the event and drift machinery
# is exercised identically.
SHORT = GameConfig(session_length=600)
FLOW_STREAMS = (
    Stream.FLOW_ARRIVAL,
    Stream.FLOW_SIZE,
    Stream.FLOW_DIRECTION,
    Stream.FLOW_INFORMED,
    Stream.COMPETITION,
    Stream.AGENTS,
)


def consume_flow(rng: RngStreams, n: int = 10_000) -> None:
    """Draw as a busy session would: bulk from every flow stream and a few
    hundred per-RFQ substreams."""
    for name in FLOW_STREAMS:
        rng.stream(name).random(n)
        for i in range(200):
            rng.substream(name, i).random()


def assert_paths_identical(a: FairValuePath, b: FairValuePath) -> None:
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.diffusion, b.diffusion)
    np.testing.assert_array_equal(a.drift, b.drift)
    np.testing.assert_array_equal(a.event_steps, b.event_steps)
    np.testing.assert_array_equal(a.jump_sizes, b.jump_sizes)
    assert a.drift_windows == b.drift_windows
    np.testing.assert_array_equal(a.public.screen, b.public.screen)


def expected_session_sigma(cfg: GameConfig) -> float:
    """From first principles, not from the module, or the round-trip test
    would be tautological. Annual vol over sqrt(days), scaled by the sqrt of
    the fraction of a trading day the session covers."""
    daily = cfg.vol_annual_bp / math.sqrt(cfg.trading_days_per_year)
    return daily * math.sqrt(cfg.session_length / (cfg.trading_hours_per_day * 3600.0))


# --------------------------------------------------------------------------
# The ones that matter
# --------------------------------------------------------------------------


def test_independent_of_flow():
    """Generate after a busy session's worth of flow draws; identical to a
    quiet one. Also identical when generated twice on the same RngStreams.

    If fair value drew from a stream anything else touched, the path would
    shift with the number of RFQs, and "same market, vary the flow" -- the
    headline experiment -- would silently compare two different markets.
    """
    quiet = generate(GameConfig(), RngStreams(42))

    rng = RngStreams(42)
    consume_flow(rng)
    busy = generate(GameConfig(), rng)
    assert_paths_identical(quiet, busy)

    again = generate(GameConfig(), rng)
    assert_paths_identical(quiet, again)


def test_jump_times_and_signs_survive_a_size_distribution_change():
    """Same calendar, same signs, same drift; only the magnitudes move.

    This is the substream property: sizes come from per-event generators and
    the sign is the first draw from each. Without it "same prints, bigger
    prints" would re-roll the calendar and the counterfactual would be noise.
    """
    base = GameConfig(jump_size_min_bp=2.0, jump_size_max_bp=4.0)
    bigger = GameConfig(jump_size_min_bp=6.0, jump_size_max_bp=8.0)
    moved = 0
    for seed in range(40):
        a = generate(base, RngStreams(seed))
        b = generate(bigger, RngStreams(seed))
        np.testing.assert_array_equal(a.event_steps, b.event_steps)
        np.testing.assert_array_equal(np.sign(a.jump_sizes), np.sign(b.jump_sizes))
        np.testing.assert_array_equal(a.diffusion, b.diffusion)
        np.testing.assert_array_equal(a.drift, b.drift)
        assert a.drift_windows == b.drift_windows
        if a.event_steps.size:
            assert np.all(np.abs(b.jump_sizes) >= 6.0)
            moved += 1
    assert moved > 10, "too few sessions had events to prove anything"


def test_drift_parameters_never_move_a_jump_or_the_diffusion():
    """Dial the drift; the prints and the diffusion are bit-identical.

    Drift has its own stream, so the inventory consequences of a drift regime
    can be studied against an unchanged set of prints.
    """
    base = GameConfig()
    other = GameConfig(drift_magnitude_min_bp=3.0, drift_magnitude_max_bp=5.0, drift_prob_agree=0.9)
    for seed in range(20):
        a = generate(base, RngStreams(seed))
        b = generate(other, RngStreams(seed))
        np.testing.assert_array_equal(a.event_steps, b.event_steps)
        np.testing.assert_array_equal(a.jump_sizes, b.jump_sizes)
        np.testing.assert_array_equal(a.diffusion, b.diffusion)


def test_event_count_never_moves_the_diffusion():
    """Brief 01: dialling jump intensity must not shift the diffusion path
    underneath it. A zero-event session and a two-event session on the same
    seed share every diffusion increment."""
    quiet = generate(GameConfig(max_events=0), RngStreams(7))
    eventful = generate(GameConfig(max_events=2), RngStreams(7))
    assert quiet.event_steps.size == 0
    np.testing.assert_array_equal(quiet.diffusion, eventful.diffusion)
    np.testing.assert_array_equal(quiet.public.screen - quiet.values[::quiet.public.refresh_steps],
                                  eventful.public.screen - eventful.values[::eventful.public.refresh_steps])


@pytest.mark.parametrize(
    "hours,expected_bp",
    [
        (24.0, 1.637),  # the brief's own figure at 90 normal vol, 2h session
        (6.5, 3.145),  # and at a 6.5h day -- nearly double for the same input
    ],
)
def test_vol_round_trip(hours, expected_bp):
    """Annualised bp in, measured session stdev out, within 2%.

    Measured as the per-step stdev of the diffusion increments scaled by
    sqrt(n_steps), over a handful of seeds. With 72,000 increments the sample
    stdev has a relative standard error of ~0.26%, so 2% is ~8 SE: loose
    enough never to flake, tight enough to catch a factor of 1.05. The errors
    this exists for are sqrt(252) = 15.9x, sqrt(24) = 4.9x, and sqrt(dt) =
    3.2x, all of which miss by an order of magnitude. The expected values are
    the brief's numbers, computed independently in the test.
    """
    cfg = GameConfig(trading_hours_per_day=hours)
    assert math.isclose(expected_session_sigma(cfg), expected_bp, rel_tol=1e-3)
    for seed in (1, 2, 3):
        path = generate(cfg, RngStreams(seed))
        inc = np.diff(path.diffusion)
        measured = inc.std(ddof=1) * math.sqrt(cfg.n_steps)
        assert math.isclose(measured, expected_bp, rel_tol=0.02), (measured, expected_bp)
    assert math.isclose(session_sigma(cfg), expected_bp, rel_tol=1e-3)


def test_increment_variance_matches_sigma_squared_dt():
    """The brief's test 4, as a per-step variance check with a 3% tolerance.
    Relative SE of a sample variance over n=72,000 is sqrt(2/n) = 0.53%, so 3%
    is ~6 SE. Zero drift is checked alongside: the mean increment must be
    indistinguishable from zero, or the process has a free edge in it."""
    cfg = GameConfig()
    per_step = sigma_per_step(cfg)
    for seed in (10, 11):
        inc = np.diff(generate(cfg, RngStreams(seed)).diffusion)
        assert math.isclose(inc.var(ddof=1), per_step**2, rel_tol=0.03)
        assert abs(inc.mean()) < 4 * per_step / math.sqrt(cfg.n_steps)


def test_markout_truncation_is_flagged_not_silent():
    """A fill 30 s before the close gets a 30 s answer to a 5-minute question.

    Averaged in naively, late fills look harmless and measured adverse
    selection comes out flattering. The clamp is deliberate; the flag is what
    stops it being silent. Boundary: truncated iff step + horizon > n_steps,
    so a horizon ending exactly on the last step is a full measurement.
    """
    cfg = GameConfig()
    path = generate(cfg, RngStreams(3))
    n = path.n_steps
    horizons = cfg.markout_horizons_steps  # (100, 600, 3000)

    late = n - 300  # 30 s before the close: only the 10 s horizon fits
    m = path.markout(late, horizons)
    assert [x.truncated for x in m] == [False, True, True]
    assert m[0].change == path.relative[late + 100] - path.relative[late]
    assert m[1].change == path.relative[n] - path.relative[late]
    assert m[2].change == path.relative[n] - path.relative[late]

    exact = path.markout(n - 3000, (3000,))[0]
    assert exact.truncated is False and exact.change == path.relative[n] - path.relative[n - 3000]
    over = path.markout(n - 2999, (3000,))[0]
    assert over.truncated is True
    assert all(x.truncated for x in path.markout(n, horizons))
    assert all(x.change == 0.0 for x in path.markout(n, horizons))


def test_event_markout_is_undefined_when_no_print_follows():
    """Undefined, not zero. A zero would be averaged in as "no adverse
    selection" for every fill after the last print of the session."""
    cfg = GameConfig()
    seed = next(s for s in range(200) if generate(cfg, RngStreams(s)).event_steps.size == 1)
    path = generate(cfg, RngStreams(seed))
    e = int(path.event_steps[0])

    before = path.event_markout(e - 100)
    assert before == EventMarkOut(event_index=0, event_step=e, change=path.relative[e] - path.relative[e - 100])
    # It contains the jump: the mark is to just after the print.
    assert abs(before.change - path.jump_sizes[0]) < 1.0  # diffusion + drift over 10 s is far below 1bp

    at = path.event_markout(e)
    assert at.change is None and at.event_index is None and at.event_step is None
    assert path.event_markout(path.n_steps).change is None
    assert generate(GameConfig(max_events=0), RngStreams(1)).event_markout(0).change is None


def test_start_level_invariance():
    """Shift start_level by any amount: every price shifts by that constant and
    every P&L number is bit-identical.

    Differences are computed on ``relative``, never on ``values``, so the
    invariance of mark-outs and lookaheads is exact rather than within
    floating-point noise of a 9550 offset. ``values`` themselves match to a
    few ulp, which is all float64 can promise when adding a constant.
    """
    a = generate(GameConfig(start_level=9550.0), RngStreams(5))
    b = generate(GameConfig(start_level=100.0), RngStreams(5))
    c = generate(GameConfig(start_level=-3.25), RngStreams(5))
    for other, shift in ((b, 100.0 - 9550.0), (c, -3.25 - 9550.0)):
        np.testing.assert_allclose(other.values, a.values + shift, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(other.relative, a.relative)
        np.testing.assert_array_equal(other.diffusion, a.diffusion)
        np.testing.assert_array_equal(other.jump_sizes, a.jump_sizes)
        assert other.drift_windows == a.drift_windows
        np.testing.assert_allclose(other.public.screen, a.public.screen + shift, rtol=0, atol=1e-9)
        for step in (0, 1234, 40_000, a.n_steps):
            assert other.markout(step, (100, 600, 3000)) == a.markout(step, (100, 600, 3000))
            assert other.lookahead(step, 500) == a.lookahead(step, 500)
            assert other.event_markout(step) == a.event_markout(step)


# --------------------------------------------------------------------------
# Determinism and shape
# --------------------------------------------------------------------------


def test_deterministic():
    assert_paths_identical(generate(GameConfig(), RngStreams(2024)), generate(GameConfig(), RngStreams(2024)))


def test_different_seeds_differ():
    a, b = generate(GameConfig(), RngStreams(1)), generate(GameConfig(), RngStreams(2))
    assert not np.array_equal(a.diffusion, b.diffusion)


def test_shapes_and_bounds():
    cfg = GameConfig()
    path = generate(cfg, RngStreams(9))
    n = cfg.n_steps
    assert path.n_steps == n
    for arr in (path.values, path.relative, path.diffusion, path.drift):
        assert arr.shape == (n + 1,)
    assert path.diffusion[0] == 0.0 and path.drift[0] == 0.0 and path.relative[0] == 0.0
    assert path.values[0] == cfg.start_level
    assert path.public.screen.shape == (n // cfg.screen_refresh_steps + 1,)
    assert path.event_steps.dtype.kind == "i"
    assert np.all(np.diff(path.event_steps) > 0)  # sorted, unique
    assert np.all((path.event_steps >= 1) & (path.event_steps <= n))
    assert path.jump_sizes.shape == path.event_steps.shape
    assert len(path.drift_windows) == path.event_steps.size


def test_values_decompose_exactly():
    """values == start_level + diffusion + drift + jumps, by construction, and
    the jump lands on the increment into the event step."""
    path = generate(GameConfig(), RngStreams(3))
    jumps = path.relative - path.diffusion - path.drift
    for e, size in zip(path.event_steps, path.jump_sizes):
        assert math.isclose(jumps[e] - jumps[e - 1], size, rel_tol=0, abs_tol=1e-9)
    # Subtracting three cumsums leaves ~1e-13 residuals everywhere, so count
    # increments that are jumps rather than increments that are nonzero.
    steps_with_a_jump = np.flatnonzero(np.abs(np.diff(jumps)) > 1e-6) + 1
    np.testing.assert_array_equal(steps_with_a_jump, path.event_steps)
    np.testing.assert_array_equal(path.values, path.start_level + path.relative)


def test_arrays_are_read_only():
    """A frozen dataclass does not freeze array contents. An agent writing into
    the truth would corrupt every later mark-out, silently."""
    path = generate(GameConfig(), RngStreams(1))
    for arr in (path.values, path.relative, path.diffusion, path.drift, path.event_steps, path.jump_sizes, path.public.screen, path.public.event_steps):
        with pytest.raises(ValueError):
            arr[0] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        path.start_level = 0.0


# --------------------------------------------------------------------------
# Events and drift
# --------------------------------------------------------------------------


def test_event_count_is_uniform_on_zero_to_max():
    """Mean 1.0 and every count 0..2 present, over 600 short sessions. SE of
    the mean is sqrt(2/3)/sqrt(600) = 0.033, so 0.15 is ~4.5 SE."""
    counts = np.array([generate(SHORT, RngStreams(s)).event_steps.size for s in range(600)])
    assert set(counts) == {0, 1, 2}
    assert abs(counts.mean() - 1.0) < 0.15


def test_jump_sizes_are_symmetric_and_in_range():
    """Symmetry is the design invariant, not cosmetics: an asymmetric print
    distribution is a free edge. Over ~600 prints the sign mean has SE 0.04."""
    sizes = np.concatenate([generate(SHORT, RngStreams(s)).jump_sizes for s in range(600)])
    assert sizes.size > 400
    assert np.all((np.abs(sizes) >= 2.0) & (np.abs(sizes) <= 4.0))
    assert abs(np.sign(sizes).mean()) < 0.15


def test_drift_agrees_with_the_print_sixty_percent_of_the_time():
    """p = 0.60 is an agreement probability. Over ~600 prints SE is 0.02, so
    0.08 is 4 SE; it also separates 0.60 from 0.50 (a coin flip) and from 0.80
    (the informed client's q), which must stay distinct or "informed" means
    nothing. Direction mean is also checked: no net drift across sessions."""
    agree, directions = [], []
    for s in range(600):
        path = generate(SHORT, RngStreams(s))
        for w, size in zip(path.drift_windows, path.jump_sizes):
            agree.append(np.sign(w.total_bp) == np.sign(size))
            directions.append(np.sign(w.total_bp))
    assert len(agree) > 400
    assert abs(np.mean(agree) - 0.60) < 0.08
    assert abs(np.mean(directions)) < 0.15


def test_drift_windows_fit_before_their_print_and_shorten_when_early():
    """The window lies in [0, event_step]; a print too early for the drawn
    duration gets a window from step 0 at the same per-step rate.

    The drawn duration is not observable from outside, so the invariants are
    stated on what is: the per-step rate is always within the range the config
    implies, because shortening preserves it; a window that does not start at
    step 0 was never shortened, so its length and total are in range; and a
    window shorter than the minimum duration must be a shortened one, ending
    exactly at its print.
    """
    cfg = GameConfig()
    min_dur = round(cfg.drift_duration_min_s / cfg.dt)
    max_dur = round(cfg.drift_duration_max_s / cfg.dt)
    rate_lo = cfg.drift_magnitude_min_bp / max_dur
    rate_hi = cfg.drift_magnitude_max_bp / min_dur
    seen_full, seen_short = False, False
    for s in range(60):
        path = generate(cfg, RngStreams(s))
        for w, e in zip(path.drift_windows, path.event_steps):
            assert 0 <= w.start_step < w.end_step <= e
            length = w.end_step - w.start_step
            rate = abs(w.total_bp) / length
            assert rate_lo * (1 - 1e-9) <= rate <= rate_hi * (1 + 1e-9)
            assert abs(w.total_bp) <= cfg.drift_magnitude_max_bp
            if w.start_step > 0:
                seen_full = True
                assert min_dur <= length <= max_dur
                assert cfg.drift_magnitude_min_bp <= abs(w.total_bp) <= cfg.drift_magnitude_max_bp
            if length < min_dur:
                seen_short = True
                assert w.start_step == 0 and w.end_step == e
                assert abs(w.total_bp) < cfg.drift_magnitude_min_bp
    assert seen_full and seen_short


def test_drift_ramp_is_linear_and_overlaps_add():
    """Checked on the increment builder with synthetic windows, since
    overlapping draws cannot be forced from a seed."""
    a = DriftWindow(event_index=0, start_step=10, end_step=20, total_bp=1.0)
    b = DriftWindow(event_index=1, start_step=15, end_step=25, total_bp=-2.0)
    inc = _drift_increments(30, [a, b])
    assert inc[:10].sum() == 0 and inc[25:].sum() == 0
    np.testing.assert_allclose(inc[10:15], 0.1)
    np.testing.assert_allclose(inc[15:20], 0.1 - 0.2)
    np.testing.assert_allclose(inc[20:25], -0.2)
    assert math.isclose(inc.sum(), -1.0)


# --------------------------------------------------------------------------
# Lookahead
# --------------------------------------------------------------------------


def test_lookahead_clamps_and_zero_horizon_is_zero():
    path = generate(GameConfig(), RngStreams(4))
    n = path.n_steps
    assert path.lookahead(1000, 0) == 0.0
    assert path.lookahead(n, 0) == 0.0
    assert path.lookahead(n, 10_000) == 0.0
    assert path.lookahead(n - 5, 10_000) == path.relative[n] - path.relative[n - 5]
    assert path.lookahead(0, n) == path.relative[n]
    with pytest.raises(ValueError):
        path.lookahead(n + 1, 1)
    with pytest.raises(ValueError):
        path.lookahead(-1, 1)  # negative would silently index from the end
    with pytest.raises(ValueError):
        path.lookahead(0, -1)


def test_jumps_within_uses_a_half_open_window_and_clamps():
    cfg = GameConfig()
    seed = next(s for s in range(200) if generate(cfg, RngStreams(s)).event_steps.size == 2)
    path = generate(cfg, RngStreams(seed))
    e0, e1 = (int(x) for x in path.event_steps)
    np.testing.assert_array_equal(path.jumps_within(e0 - 1, 1), path.jump_sizes[:1])  # (e0-1, e0]
    assert path.jumps_within(e0, 0).size == 0  # (e0, e0] is empty
    assert path.jumps_within(e0, e1 - e0 - 1).size == 0
    np.testing.assert_array_equal(path.jumps_within(e0, e1 - e0), path.jump_sizes[1:])
    np.testing.assert_array_equal(path.jumps_within(0, 10**9), path.jump_sizes)


# --------------------------------------------------------------------------
# Screen
# --------------------------------------------------------------------------


def test_screen_is_truth_plus_noise_at_each_refresh():
    """Noise stdev matches config (SE over 1,441 readings is ~1.9%, so 10% is
    generous) and the screen is exactly the truth when noise is zero."""
    cfg = GameConfig()
    path = generate(cfg, RngStreams(6))
    r = cfg.screen_refresh_steps
    residual = path.public.screen - path.values[::r]
    assert math.isclose(residual.std(ddof=1), cfg.screen_noise_bp, rel_tol=0.10)
    assert abs(residual.mean()) < 0.1

    clean = generate(GameConfig(screen_noise_bp=0.0), RngStreams(6))
    np.testing.assert_array_equal(clean.public.screen, clean.values[::r])


def test_screen_reading_is_piecewise_constant_with_known_age():
    cfg = GameConfig()
    pub = generate(cfg, RngStreams(6)).public
    r = cfg.screen_refresh_steps
    assert pub.screen_at(0) == pub.screen[0]
    for step in (r - 1, r, r + 1, 2 * r + 7, cfg.n_steps):
        assert pub.screen_at(step) == pub.screen[step // r]
        assert math.isclose(pub.screen_age_s(step), (step % r) * cfg.dt)
    assert pub.screen_age_s(r) == 0.0
    assert math.isclose(pub.screen_age_s(r - 1), 4.9)


def test_screen_noise_never_moves_the_truth():
    """The screen has its own stream. Turning observation noise up must leave
    fair value, prints and drift bit-identical."""
    a = generate(GameConfig(screen_noise_bp=0.5), RngStreams(8))
    b = generate(GameConfig(screen_noise_bp=5.0), RngStreams(8))
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.jump_sizes, b.jump_sizes)
    assert a.drift_windows == b.drift_windows


def test_public_view_carries_no_truth():
    """The engine hands agents ``path.public``. It must be structurally unable
    to answer "what is fair value", not merely polite about it."""
    path = generate(GameConfig(), RngStreams(1))
    pub = path.public
    assert isinstance(pub, PublicView)
    fields = {f.name for f in dataclasses.fields(PublicView)}
    assert fields == {"n_steps", "dt", "refresh_steps", "event_steps", "screen"}
    assert not any(hasattr(pub, name) for name in ("values", "relative", "jump_sizes", "drift_windows", "diffusion", "drift"))
    assert not hasattr(pub, "lookahead")
    # Timing is public; outcome is not.
    np.testing.assert_array_equal(pub.event_steps, path.event_steps)


def test_countdown_to_next_event():
    cfg = GameConfig()
    seed = next(s for s in range(200) if generate(cfg, RngStreams(s)).event_steps.size == 2)
    pub = generate(cfg, RngStreams(seed)).public
    e0, e1 = (int(x) for x in pub.event_steps)
    assert pub.steps_to_next_event(0) == e0
    assert pub.steps_to_next_event(e0 - 1) == 1
    assert pub.steps_to_next_event(e0) == e1 - e0
    assert pub.steps_to_next_event(e1) is None
    assert pub.steps_to_next_event(cfg.n_steps) is None


# --------------------------------------------------------------------------
# Record and hygiene
# --------------------------------------------------------------------------


def test_records_are_plain_types_and_round_trip_json():
    path = generate(GameConfig(), RngStreams(12))
    for rec in (path.to_record(), path.public.to_record()):
        assert json.loads(json.dumps(rec)) == rec
    assert "jump_sizes_bp" in path.to_record()
    assert "jump_sizes_bp" not in path.public.to_record()


def test_sigma_conversion_is_written_out():
    """Pins the arithmetic at the defaults so a refactor that drops a sqrt
    fails here with a number, not three months later in a P&L."""
    cfg = GameConfig()
    assert math.isclose(sigma_per_step(cfg), 90 / math.sqrt(252) / math.sqrt(86_400) * math.sqrt(0.1))
    assert math.isclose(sigma_per_step(cfg), 0.006102, rel_tol=1e-3)
    assert math.isclose(session_sigma(cfg), 1.637, rel_tol=1e-3)
    assert sigma_per_step(GameConfig(vol_annual_bp=0.0)) == 0.0


def test_module_imports_only_rng_config_and_numpy():
    source = (SRC_DIR / "fair_value.py").read_text(encoding="utf-8")
    internal = set(re.findall(r"^\s*from\s+mm_game\.(\w+)\s+import", source, re.MULTILINE))
    assert internal == {"rng", "config"}
    assert not re.search(r"^\s*import\s+mm_game", source, re.MULTILINE)
