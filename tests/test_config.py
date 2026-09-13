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
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mm_game
from mm_game.config import DEFAULT_CONFIG, GameConfig

SRC_DIR = Path(mm_game.__file__).resolve().parent


def test_defaults_match_the_instrument_spec():
    """One futures contract in basis-point units: half a bp per tick, $25 a
    tick, so $50 per bp per lot. 9550 displays as 95.50."""
    cfg = GameConfig()
    assert cfg.tick_size == 0.5
    assert cfg.tick_value == 25.0
    assert cfg.lot_size == 1
    assert cfg.bp_per_point == 100.0
    assert cfg.start_level == 9550.0
    assert cfg.session_length == 7200
    assert cfg.dt == 0.1
    assert cfg.vol_annual_bp == 90.0
    assert cfg.trading_hours_per_day == 24.0
    assert cfg.trading_days_per_year == 252
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


def test_tick_is_half_a_basis_point_and_fifty_dollars_per_bp():
    """The old (0.005, 25.0) pair mixed a price increment from one instrument
    with a per-bp figure from another and made every P&L 2x hot. Pin the
    relation, not just the numbers."""
    cfg = GameConfig()
    dollars_per_bp_per_lot = cfg.tick_value / cfg.tick_size
    assert dollars_per_bp_per_lot == 50.0


def test_display_convention_is_pinned():
    """The model works in bp; a screen shows points. 9550bp is 95.50 and a
    half-bp tick is 0.005, three decimals. Pinned so the conversion is never
    reinvented from memory at a display site -- the tick-size bug was a
    convention living in two places."""
    cfg = GameConfig()
    assert cfg.start_level / cfg.bp_per_point == 95.5
    assert cfg.tick_size / cfg.bp_per_point == 0.005
    assert (cfg.start_level + 7 * cfg.tick_size) / cfg.bp_per_point == 95.535


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
    assert GameConfig(max_events=1, event_min_gap_s=100_000).max_events == 1
    assert GameConfig(max_events=0, event_min_time_s=7000).max_events == 0
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
        ("max_events", -1),
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
    ],
)
def test_bad_values_rejected(field, bad):
    """NaN is in the list on purpose: `x <= 0` lets it through, `not x > 0`
    does not. Negative vol is the silent one -- sigma * z with sigma < 0 is
    statistically identical to sigma > 0."""
    with pytest.raises(ValueError):
        GameConfig(**{field: bad})


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
        session_length=10, dt=10, screen_refresh_s=10, markout_horizons_s=(10,), max_events=0
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
        (1, 0.1, 10, dict(screen_refresh_s=0.1, markout_horizons_s=(0.5, 1.0), max_events=0)),
        (7200, 0.001, 7_200_000, {}),
        # 21 / 0.7 is 30.000000000000004 in float: a naive `.is_integer()`
        # rejects it, and a naive `int()` truncates it to 30 by luck only.
        (21, 0.7, 30, dict(screen_refresh_s=7.0, markout_horizons_s=(7.0, 21.0), max_events=0)),
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
