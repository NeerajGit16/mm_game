"""Session configuration.

``GameConfig`` is inert description: a frozen dataclass you can serialise, diff
and put in a log line. It holds no randomness -- a session is ``(config, seed)``
and the seed lives with the engine, not here. Keeping the two apart is what
lets config equality mean "same scenario": running one scenario over 500 seeds
leaves this half byte-identical across all 500, so a diff of two run headers
shows the config was held fixed. A seed field would turn that sweep into 500
different configs and "same config, different seed" would stop being
expressible.

Validation raises ``ValueError`` explicitly rather than using ``assert``:
``python -O`` strips ``assert`` statements entirely, so an assert-based guard
lets ``GameConfig(dt=-1)`` construct silently under that flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Relative tolerance for the grid-divisibility check. Float division carries at
# most about 1 ulp (~1e-16) of error, so this is generous to rounding and still
# strict against a genuinely ragged grid (100 / 0.7 misses by ~1e-3).
_GRID_REL_TOL: float = 1e-12


@dataclass(frozen=True)
class GameConfig:
    tick_size: float = 0.005
    tick_value: float = 25.0
    lot_size: int = 1
    session_length: int = 60 * 60 * 2
    # 0.1 s: 72,000 grid points per session (~0.6 MB per float64 array) rather
    # than the 7.2M of the original 0.001. Nothing in an RFQ game consumes
    # millisecond fair-value resolution, and the smaller grid matters once
    # sessions run in parallel.
    dt: float = 0.1

    def __post_init__(self) -> None:
        # `not x > 0` rather than `x <= 0`: the former also rejects NaN.
        if not self.tick_size > 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if not self.tick_value > 0:
            raise ValueError(f"tick_value must be positive, got {self.tick_value}")
        if not self.lot_size >= 1:
            raise ValueError(f"lot_size must be at least 1, got {self.lot_size}")
        if not self.session_length > 0:
            raise ValueError(
                f"session_length must be positive, got {self.session_length}"
            )
        if not 0 < self.dt <= self.session_length:
            raise ValueError(
                f"dt must be in (0, session_length], got dt={self.dt} with "
                f"session_length={self.session_length}"
            )
        # The fair value grid is np.arange(n_steps + 1) * dt with events indexed
        # by integer step, so session_length must be a whole number of steps.
        # Checked with a tolerance: the ratio is a float division, and a naive
        # float.is_integer() would reject legitimate configs on rounding.
        n_steps = round(self.session_length / self.dt)
        if not math.isclose(
            n_steps * self.dt, self.session_length, rel_tol=_GRID_REL_TOL, abs_tol=0.0
        ):
            raise ValueError(
                "session_length / dt must be an integer number of steps, got "
                f"{self.session_length} / {self.dt} = {self.session_length / self.dt}"
            )

    @property
    def n_steps(self) -> int:
        """Number of fair value grid steps in the session.

        Read this rather than recomputing ``session_length / dt``: it is the
        rounded, validated count, so callers cannot disagree on the grid size
        through a different rounding rule.
        """
        return round(self.session_length / self.dt)


DEFAULT_CONFIG = GameConfig()
