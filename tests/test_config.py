"""Tests for mm_game.config.

GameConfig is inert description -- a frozen dataclass you can serialise, diff and
put in a log line. These tests pin the defaults, prove the immutability is real,
and check each validation guard.

Validation raises ValueError explicitly. The ``-O`` subprocess test is the one
that keeps it that way: a regression back to bare ``assert`` passes every other
test here, because pytest never runs with ``-O``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mm_game
from mm_game.config import DEFAULT_CONFIG, GameConfig

SRC_DIR = Path(mm_game.__file__).resolve().parent


def test_defaults_match_the_instrument_spec():
    """One futures contract in basis-point units: a quarter of a bp per tick,
    $12.50 a tick, so $50 per bp per lot. 9550 displays as 95.50."""
    cfg = GameConfig()
    assert cfg.tick_size == 0.25
    assert cfg.tick_value == 12.5
    assert cfg.lot_size == 1
    assert cfg.bp_per_point == 100.0
    assert cfg.start_level == 9550.0
    assert cfg.session_length == 7200
    assert cfg.dt == 0.1
    assert cfg.vol_annual_bp == 90.0
    assert cfg.trading_hours_per_day == 24.0
    assert cfg.trading_days_per_year == 252
    assert cfg.event_count_probs == (0.2, 0.4, 0.4)
    assert cfg.max_events == 2
    assert (cfg.event_min_time_s, cfg.event_end_margin_s, cfg.event_min_gap_s) == (900.0, 600.0, 1800.0)
    assert (cfg.jump_size_min_bp, cfg.jump_size_max_bp) == (2.0, 4.0)
    assert cfg.drift_prob_agree == 0.60
    assert (cfg.drift_duration_min_s, cfg.drift_duration_max_s) == (600.0, 2400.0)
    assert (cfg.drift_magnitude_min_bp, cfg.drift_magnitude_max_bp) == (1.0, 2.0)
    assert cfg.screen_noise_bp == 0.5
    assert cfg.screen_noise_corr_s == 90.0
    assert cfg.screen_refresh_s == 5.0
    assert cfg.markout_horizons_s == (10.0, 60.0, 300.0)
    assert cfg.rfq_rate_per_hour == 100.0
    assert (cfg.n_hedge_funds, cfg.n_real_money, cfg.n_corporates) == (6, 6, 4)
    assert (cfg.informed_prob_hedge_fund, cfg.informed_prob_real_money, cfg.informed_prob_corporate) == (
        0.50,
        0.15,
        0.05,
    )
    assert cfg.informed_activity_multiplier == 1.0
    assert (cfg.informed_accuracy, cfg.informed_signal_noise) == (0.80, 0.25)
    assert (cfg.size_median_lots, cfg.size_log_sd, cfg.size_cap_lots) == (10.0, 0.9, 100)
    assert (cfg.informed_size_ref_bp, cfg.informed_size_factor_min, cfg.informed_size_factor_max) == (1.5, 1.0, 3.0)
    assert (cfg.crowd_drift_tilt, cfg.one_way_prob_uninformed, cfg.one_way_prob_informed) == (0.60, 0.05, 0.10)
    assert cfg.n_competitors == 3
    assert (cfg.competitor_half_spread_bp, cfg.competitor_size_widen_bp) == (0.5, 0.5)
    assert (cfg.competitor_print_widen, cfg.competitor_print_window_s) == (1.0, 120.0)
    assert (cfg.competitor_screen_weight, cfg.competitor_error_bp, cfg.competitor_error_corr_s) == (0.75, 0.15, 120.0)
    assert (cfg.client_tolerance_bp, cfg.client_tolerance_log_sd, cfg.client_tolerance_size_growth) == (0.75, 0.4, 0.5)
    assert cfg.informed_edge_given_up == 0.5
    assert (cfg.hedge_cost_bp, cfg.hedge_impact_bp, cfg.hedge_margin_bp) == (1.0, 0.5, 0.25)
    assert (cfg.hedge_print_widen, cfg.hedge_print_window_s) == (1.0, 120.0)
    assert (cfg.close_liquidation_multiplier, cfg.position_limit_lots) == (1.5, 200)


@pytest.mark.parametrize(
    "minutes,length,clients,counts,share",
    [
        (30, 1800, 6, (0.0, 1.0), 0.250),
        (60, 3600, 10, (0.2, 0.4, 0.4), 0.260),
        (120, 7200, 16, (0.2, 0.4, 0.4), 0.25625),
    ],
)
def test_session_presets_are_explicit(minutes, length, clients, counts, share):
    """30 / 60 / 120 minutes set 6 / 10 / 16 clients explicitly -- ~8-12 RFQs
    per client, which is what makes identity learnable -- and the print-count
    weights. Expected informed share stays ~25% on every preset. The
    120-minute preset is the default config, so the two can never drift
    apart."""
    cfg = GameConfig.for_session(minutes)
    assert (cfg.session_length, cfg.n_clients, cfg.event_count_probs) == (length, clients, counts)
    assert math.isclose(cfg.expected_informed_share, share)
    rfqs_per_client = cfg.rfq_rate_per_hour * minutes / 60 / cfg.n_clients
    assert 8 <= rfqs_per_client <= 13
    assert GameConfig.for_session(120) == GameConfig()
    assert GameConfig.for_session(60, rfq_rate_per_hour=50.0).rfq_rate_per_hour == 50.0
    with pytest.raises(ValueError, match="no preset"):
        GameConfig.for_session(45)


def test_event_drill_preset():
    """The 30-minute preset is an event drill: exactly one print, landing
    between minute 10 and minute 20, with a 5-15 minute drift. Validation has
    proved the one allowed count is placeable or construction would raise."""
    cfg = GameConfig.for_session(30)
    assert cfg.event_count_probs == (0.0, 1.0) and cfg.max_events == 1
    lo, hi = cfg.event_step_bounds
    assert (lo * cfg.dt, hi * cfg.dt) == (600.0, 1200.0)
    assert (cfg.drift_duration_min_s, cfg.drift_duration_max_s) == (300.0, 900.0)
    assert (cfg.n_hedge_funds, cfg.n_real_money, cfg.n_corporates) == (2, 3, 1)


def test_event_count_probs_round_trip_through_json():
    """Like the mark-out horizons: a list reloaded from JSON must give back a
    config equal to the one that generated the session."""
    cfg = GameConfig(event_count_probs=[0.25, 0.75])
    reloaded = GameConfig(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    assert reloaded == cfg and reloaded.event_count_probs == (0.25, 0.75)


def test_default_category_probabilities_are_strictly_between_zero_and_one():
    """The design rule: no category is ever certainly clean or
    certainly informed, so the label is a prior, not a classifier. Pinned on
    the defaults rather than enforced in validation, because the experiment
    independent streams exist for -- sweep the informed share from 0 to 1 on
    an otherwise identical session -- needs 0 and 1 to be expressible."""
    cfg = GameConfig()
    for p in (cfg.informed_prob_hedge_fund, cfg.informed_prob_real_money, cfg.informed_prob_corporate):
        assert 0 < p < 1
    assert GameConfig(informed_prob_hedge_fund=0.0, informed_prob_real_money=1.0).n_clients == 16
    # q must stay distinct from the crowd's agreement probability, or "informed" means nothing.
    assert cfg.informed_accuracy > cfg.drift_prob_agree + 0.1


def test_tick_is_a_quarter_basis_point_and_fifty_dollars_per_bp():
    """The old (0.005, 25.0) pair mixed a price increment from one instrument
    with a per-bp figure from another and made every P&L 2x hot. Pin the
    relation, not just the numbers, and pin that money goes through the one
    conversion property."""
    cfg = GameConfig()
    assert cfg.tick_value / cfg.tick_size == 50.0
    assert cfg.dollars_per_bp_per_lot == 50.0


def test_no_economic_parameter_is_denominated_in_ticks():
    """Every spread, tolerance, cost and margin is in bp; the tick is only the
    price grid. Catches a parameter reintroduced in ticks, which a tick change
    would then silently rescale. The field name is the unit contract, so it is
    checked structurally: the only tick-named fields are the grid's own."""
    names = {f.name for f in dataclasses.fields(GameConfig)}
    assert {n for n in names if "tick" in n} == {"tick_size", "tick_value"}


def test_economics_do_not_move_with_the_tick():
    """Halving the tick (keeping $50/bp/lot) leaves every economic function
    identical in bp: dealer half-spreads, hedge cost and dollars per bp. The
    one exception is the furthest a rounded dealer quote can sit from its
    mid, which carries half a tick of rounding by definition -- so it moves by
    exactly the change in half a tick. Catches a bp parameter multiplied by
    the tick somewhere in the spread or cost functions."""
    fine, coarse = GameConfig(), GameConfig(tick_size=0.5, tick_value=25.0)
    assert fine.dollars_per_bp_per_lot == coarse.dollars_per_bp_per_lot
    for size in (1, 10, 55, 100):
        for seconds in (0.0, 30.0, 119.0, 600.0, math.inf):
            assert fine.competitor_half_spread_bp_at(size, seconds) == coarse.competitor_half_spread_bp_at(size, seconds)
            assert fine.hedge_cost_bp_at(size, seconds) == coarse.hedge_cost_bp_at(size, seconds)
            assert math.isclose(
                coarse.competitor_max_quote_distance_bp(size, seconds)
                - fine.competitor_max_quote_distance_bp(size, seconds),
                (coarse.tick_size - fine.tick_size) / 2,
            )


def test_display_convention_is_pinned():
    """The model works in bp; a screen shows points. 9550bp is 95.50 and a
    quarter-bp tick is 0.0025, four decimals. Pinned so the conversion is
    never reinvented from memory at a display site -- the tick-size bug was a
    convention living in two places."""
    cfg = GameConfig()
    assert cfg.start_level / cfg.bp_per_point == 95.5
    assert cfg.tick_size / cfg.bp_per_point == 0.0025
    assert (cfg.start_level + 7 * cfg.tick_size) / cfg.bp_per_point == 95.5175


def test_default_grid_is_72000_steps():
    """7200 s at 0.1 s. The number fair_value.py sizes its arrays by."""
    assert GameConfig().n_steps == 72_000


def test_derived_step_counts():
    cfg = GameConfig()
    assert cfg.screen_refresh_steps == 50
    assert cfg.markout_horizons_steps == (100, 600, 3000)
    assert cfg.event_step_bounds == (9000, 66000)
    assert cfg.event_min_gap_steps == 18000


def test_event_calendar_must_be_placeable_for_every_seed():
    """The fit check is done in steps, exactly as the calendar draw computes
    it. A check in seconds that passed by a hair would let the draw fail on
    the seeds that happened to pick two prints, and only those."""
    window_s = 7200 - 900 - 600  # 5700
    assert GameConfig(event_min_gap_s=window_s - 0.1).event_min_gap_steps == 56_999
    with pytest.raises(ValueError, match="do not fit"):
        GameConfig(event_min_gap_s=window_s)  # room for two prints is zero steps
    assert GameConfig(event_count_probs=(0.5, 0.5), event_min_gap_s=100_000).max_events == 1
    assert GameConfig(event_count_probs=(1.0,), event_min_time_s=7000).max_events == 0
    # A trailing zero weight is not an allowed count, so it need not fit.
    assert GameConfig(event_count_probs=(0.5, 0.5, 0.0), event_min_gap_s=100_000).max_events == 1
    with pytest.raises(ValueError, match="empty"):
        GameConfig(event_min_time_s=7000)
    # A print can never sit on step 0 whatever the lower bound says.
    assert GameConfig(event_min_time_s=0).event_step_bounds[0] == 1


def test_default_config_is_a_plain_default_instance():
    assert DEFAULT_CONFIG == GameConfig()


def test_is_frozen():
    """Config that mutates mid-session is a bug factory."""
    cfg = GameConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.tick_size = 0.01


def test_overrides_are_supported():
    cfg = GameConfig(session_length=3600)
    assert cfg.session_length == 3600
    assert cfg.tick_size == GameConfig().tick_size


def test_is_comparable_and_serialisable():
    """It has to survive a log line and a diff."""
    assert GameConfig(dt=0.01) == GameConfig(dt=0.01)
    assert GameConfig(dt=0.01) != GameConfig(dt=0.02)
    assert dataclasses.asdict(GameConfig())["dt"] == 0.1


def test_round_trips_through_json_exactly():
    """The session record stores config as JSON. The tuple of horizons comes
    back as a list; without normalisation the reloaded config would compare
    unequal to the one that generated the session and replay would refuse."""
    cfg = GameConfig(markout_horizons_s=(10, 60))
    reloaded = GameConfig(**json.loads(json.dumps(dataclasses.asdict(cfg))))
    assert reloaded == cfg
    assert reloaded.markout_horizons_s == (10.0, 60.0)
    assert isinstance(reloaded.markout_horizons_s, tuple)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad",
    [
        ("tick_size", 0),
        ("tick_size", -0.5),
        ("tick_size", float("nan")),
        ("tick_value", 0),
        ("tick_value", -25.0),
        ("lot_size", 0),
        ("lot_size", -1),
        ("bp_per_point", 0),
        ("bp_per_point", -100.0),
        ("bp_per_point", float("nan")),
        ("start_level", float("nan")),
        ("start_level", float("inf")),
        ("session_length", 0),
        ("session_length", -60),
        ("dt", 0),
        ("dt", -0.1),
        ("dt", float("nan")),
        ("vol_annual_bp", -90.0),
        ("vol_annual_bp", float("nan")),
        ("trading_hours_per_day", 0),
        ("trading_hours_per_day", 25),
        ("trading_days_per_year", 0),
        ("event_count_probs", ()),
        ("event_count_probs", (0.5, 0.4)),  # sums to 0.9
        ("event_count_probs", (-0.1, 1.1)),
        ("event_count_probs", (float("nan"), 1.0)),
        ("event_min_time_s", -1.0),
        ("event_min_time_s", float("nan")),
        ("event_end_margin_s", -1.0),
        ("event_min_gap_s", -1.0),
        ("event_min_gap_s", float("inf")),
        ("jump_size_min_bp", -1.0),
        ("jump_size_min_bp", 5.0),  # above the default max of 4
        ("drift_prob_agree", -0.1),
        ("drift_prob_agree", 1.1),
        ("drift_prob_agree", float("nan")),
        ("drift_duration_min_s", 0),
        ("drift_duration_min_s", 3000.0),  # above the default max
        ("drift_magnitude_min_bp", -1.0),
        ("drift_magnitude_max_bp", 0.5),  # below the default min
        ("screen_noise_bp", -0.5),
        ("screen_noise_bp", float("nan")),
        ("screen_noise_corr_s", -1.0),
        ("screen_noise_corr_s", float("nan")),
        ("screen_refresh_s", 0),
        ("screen_refresh_s", 7201),
        ("screen_refresh_s", 0.13),  # not a whole number of 0.1 s steps
        ("markout_horizons_s", (0,)),
        ("markout_horizons_s", (7201,)),
        ("markout_horizons_s", (10, 0.13)),
        ("rfq_rate_per_hour", -1.0),
        ("rfq_rate_per_hour", float("nan")),
        ("rfq_rate_per_hour", 36_001.0),  # more than one RFQ per 0.1 s step
        ("n_hedge_funds", -1),
        ("informed_prob_hedge_fund", -0.1),
        ("informed_prob_real_money", 1.1),
        ("informed_prob_corporate", float("nan")),
        ("informed_activity_multiplier", 0.0),
        ("informed_activity_multiplier", float("inf")),
        ("informed_accuracy", 1.5),
        ("informed_signal_noise", -0.1),
        ("size_median_lots", 0.0),
        ("size_log_sd", -0.1),
        ("size_cap_lots", 0),
        ("informed_size_ref_bp", 0.0),
        ("informed_size_factor_min", 0.0),
        ("informed_size_factor_max", 0.5),  # below the default min of 1
        ("crowd_drift_tilt", 1.2),
        ("one_way_prob_uninformed", -0.01),
        ("one_way_prob_informed", float("nan")),
        ("n_competitors", 0),
        ("competitor_half_spread_bp", 0.1),  # under half a tick: a rounded bid could reach the mid
        ("competitor_size_widen_bp", -1.0),
        ("competitor_screen_weight", 1.5),
        ("competitor_error_bp", -0.1),
        ("competitor_error_corr_s", float("nan")),
        ("client_tolerance_bp", 0.0),
        ("client_tolerance_log_sd", -0.1),
        ("informed_edge_given_up", 1.5),
        ("hedge_cost_bp", 0.0),
        ("hedge_margin_bp", -0.1),
        ("hedge_cost_bp", 0.75),  # breaches the margin over the competitor quotes
        ("close_liquidation_multiplier", 0.9),  # holding to the bell would be cheaper than hedging
        ("position_limit_lots", 50),  # one ticket from flat could breach it
    ],
)
def test_bad_values_rejected(field, bad):
    """NaN is in the list on purpose: `x <= 0` lets it through, `not x > 0`
    does not. Negative vol is the silent one -- sigma * z with sigma < 0 is
    statistically identical to sigma > 0."""
    with pytest.raises(ValueError):
        GameConfig(**{field: bad})


def test_roster_must_not_be_empty():
    with pytest.raises(ValueError, match="at least one client"):
        GameConfig(n_hedge_funds=0, n_real_money=0, n_corporates=0)


def test_zero_vol_is_allowed():
    """A deterministic path plus prints is a coherent scenario, and useful for
    isolating the event machinery. It is also loud when eyeballed, unlike a
    negative vol, so there is no reason to refuse it."""
    assert GameConfig(vol_annual_bp=0.0).vol_annual_bp == 0.0


def test_dt_may_not_exceed_the_session():
    with pytest.raises(ValueError):
        GameConfig(session_length=10, dt=11)


def test_dt_may_equal_the_session():
    """Degenerate but coherent: a single-step session. The screen and mark-out
    horizons have to fit the session too."""
    cfg = GameConfig(
        session_length=10, dt=10, screen_refresh_s=10, markout_horizons_s=(10,), event_count_probs=(1.0,)
    )
    assert cfg.dt == 10
    assert cfg.n_steps == 1
    assert cfg.screen_refresh_steps == 1
    assert cfg.markout_horizons_steps == (1,)


@pytest.mark.parametrize(
    "session_length,dt",
    [
        (100, 0.7),
        (10, 3),
        (7200, 7),
        # A near miss: 72,000 steps of this dt end 7.2e-7 s past the session.
        # Pins the tolerance as tight -- loosening it to something like 1e-6
        # would start accepting dt values that do not actually hit the end.
        (7200, 0.10000000001),
    ],
)
def test_ragged_grid_rejected(session_length, dt):
    """session_length / dt must be a whole number of steps.

    A ragged grid would not raise downstream: np.arange(n_steps + 1) * dt just
    ends short of (or past) session_length, and the last RFQs snap off the end
    of the price path or index past it.
    """
    with pytest.raises(ValueError, match="integer number of steps"):
        GameConfig(session_length=session_length, dt=dt)


@pytest.mark.parametrize(
    "session_length,dt,n_steps,extra",
    [
        (7200, 0.1, 72_000, {}),
        (3600, 0.3, 12_000, dict(screen_refresh_s=3.0, markout_horizons_s=(9.0, 60.0, 300.0))),
        (1, 0.1, 10, dict(screen_refresh_s=0.1, markout_horizons_s=(0.5, 1.0), event_count_probs=(1.0,))),
        (7200, 0.001, 7_200_000, {}),
        # 21 / 0.7 is 30.000000000000004 in float: a naive `.is_integer()`
        # rejects it, and a naive `int()` truncates it to 30 by luck only.
        (21, 0.7, 30, dict(screen_refresh_s=7.0, markout_horizons_s=(7.0, 21.0), event_count_probs=(1.0,))),
    ],
)
def test_divisible_grid_accepted(session_length, dt, n_steps, extra):
    """Checked with a tolerance, not float.is_integer().

    0.7 (and 0.1, 0.3) are not exactly representable; the check must not reject
    a real grid because the float division landed a few ulp off an integer.
    The 21 / 0.7 case is the one that actually exercises this.
    """
    assert GameConfig(session_length=session_length, dt=dt, **extra).n_steps == n_steps


def test_screen_and_markouts_must_be_whole_steps():
    """Rounding a 10 s horizon at dt=0.3 to 33 steps makes it a 9.9 s horizon
    that the record still labels 10 s. Refuse instead."""
    with pytest.raises(ValueError, match="whole number of steps"):
        GameConfig(dt=0.3, session_length=3600, screen_refresh_s=5.0)
    with pytest.raises(ValueError, match="whole number of steps"):
        GameConfig(dt=0.3, session_length=3600, screen_refresh_s=3.0, markout_horizons_s=(10.0,))


def test_validation_survives_python_O():
    """`python -O` strips `assert`. Validation must not be made of it.

    Every other test in this file passes under a regression back to bare
    asserts, because pytest never runs with -O. This is the only one that would
    notice. The script also prints sys.flags.optimize so the test cannot pass
    trivially if the -O flag is ever dropped from the command line.
    """
    script = (
        "import sys\n"
        "from mm_game.config import GameConfig\n"
        "try:\n"
        "    GameConfig(dt=-1)\n"
        "except ValueError:\n"
        "    outcome = 'rejected'\n"
        "else:\n"
        "    outcome = 'constructed'\n"
        "print(sys.flags.optimize, outcome)\n"
    )
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR.parent), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    optimize, outcome = result.stdout.split()
    assert int(optimize) >= 1, "subprocess did not run under -O"
    assert outcome == "rejected", "validation evaporated under python -O"
