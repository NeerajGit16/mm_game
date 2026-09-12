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
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mm_game
from mm_game.config import DEFAULT_CONFIG, GameConfig

SRC_DIR = Path(mm_game.__file__).resolve().parent


def test_defaults_match_the_instrument_spec():
    """A generic rates future: price near 100, 0.005 tick, fixed value per lot."""
    cfg = GameConfig()
    assert cfg.tick_size == 0.005
    assert cfg.tick_value == 25.0
    assert cfg.lot_size == 1
    assert cfg.session_length == 7200
    assert cfg.dt == 0.1


def test_default_grid_is_72000_steps():
    """7200 s at 0.1 s. The number fair_value.py sizes its arrays by."""
    assert GameConfig().n_steps == 72_000


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


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,bad",
    [
        ("tick_size", 0),
        ("tick_size", -0.005),
        ("tick_size", float("nan")),
        ("tick_value", 0),
        ("tick_value", -25.0),
        ("lot_size", 0),
        ("lot_size", -1),
        ("session_length", 0),
        ("session_length", -60),
        ("dt", 0),
        ("dt", -0.1),
        ("dt", float("nan")),
    ],
)
def test_non_positive_values_rejected(field, bad):
    """NaN is in the list on purpose: `x <= 0` lets it through, `not x > 0` does not."""
    with pytest.raises(ValueError):
        GameConfig(**{field: bad})


def test_dt_may_not_exceed_the_session():
    with pytest.raises(ValueError):
        GameConfig(session_length=10, dt=11)


def test_dt_may_equal_the_session():
    """Degenerate but coherent: a single-step session."""
    cfg = GameConfig(session_length=10, dt=10)
    assert cfg.dt == 10
    assert cfg.n_steps == 1


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
    "session_length,dt,n_steps",
    [
        (7200, 0.1, 72_000),
        (3600, 0.3, 12_000),
        (1, 0.1, 10),
        (7200, 0.001, 7_200_000),
        # 21 / 0.7 is 30.000000000000004 in float: a naive `.is_integer()`
        # rejects it, and a naive `int()` truncates it to 30 by luck only.
        (21, 0.7, 30),
    ],
)
def test_divisible_grid_accepted(session_length, dt, n_steps):
    """Checked with a tolerance, not float.is_integer().

    0.7 (and 0.1, 0.3) are not exactly representable; the check must not reject
    a real grid because the float division landed a few ulp off an integer.
    The 21 / 0.7 case is the one that actually exercises this.
    """
    assert GameConfig(session_length=session_length, dt=dt).n_steps == n_steps


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
