"""Session configuration.

``GameConfig`` is inert description: a frozen dataclass you can serialise, diff
and put in a log line. It holds no randomness -- a session is ``(config, seed)``
and the seed lives with the engine, not here. Keeping the two apart is what
lets config equality mean "same scenario": running one scenario over 500 seeds
leaves this half byte-identical across all 500, so a diff of two run headers
shows the config was held fixed. A seed field would turn that sweep into 500
different configs and "same config, different seed" would stop being
expressible.

Every parameter that changes a generated number lives here, for the same
reason: the log header records ``config`` and ``seed``, and a session must be
reconstructible from those two alone. A jump-size range kept as a module
constant would be a third, unrecorded input.

Validation raises ``ValueError`` explicitly rather than using ``assert``:
``python -O`` strips ``assert`` statements entirely, so an assert-based guard
lets ``GameConfig(dt=-1)`` construct silently under that flag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Relative tolerance for the grid-divisibility checks. Float division carries at
# most about 1 ulp (~1e-16) of error, so this is generous to rounding and still
# strict against a genuinely ragged grid (100 / 0.7 misses by ~1e-3).
_GRID_REL_TOL: float = 1e-12


def _whole_steps(seconds: float, dt: float, what: str) -> int:
    """Convert a duration in seconds to grid steps, refusing a ragged result.

    Rounding silently would be the failure: a 10 s horizon at dt=0.3 becomes 33
    steps, which is 9.9 s, and every mark-out at that horizon would be
    measured over a different interval from the one the config names.
    """
    steps = round(seconds / dt)
    if steps < 1 or not math.isclose(steps * dt, seconds, rel_tol=_GRID_REL_TOL, abs_tol=0.0):
        raise ValueError(
            f"{what} must be a whole number of steps, got {seconds} / {dt} = {seconds / dt}"
        )
    return steps


@dataclass(frozen=True)
class GameConfig:
    # -- Instrument -----------------------------------------------------------
    # One futures contract priced in basis-point units: one price unit is 1bp
    # of yield equivalent, higher is better for a long. Half a basis point per
    # tick, $25 per tick, so $50 per basis point per lot.
    tick_size: float = 0.5
    tick_value: float = 25.0
    lot_size: int = 1
    # Cosmetic: nothing in the model depends on the level (arithmetic process,
    # no reversion, absolute tick size). 9550 displays as 95.50.
    start_level: float = 9550.0

    # -- Session grid ----------------------------------------------------------
    session_length: int = 60 * 60 * 2
    # 0.1 s: 72,000 grid points per session (~0.6 MB per float64 array) rather
    # than the 7.2M of the original 0.001. Nothing in an RFQ game consumes
    # millisecond fair-value resolution, and the smaller grid matters once
    # sessions run in parallel.
    dt: float = 0.1

    # -- Diffusion -------------------------------------------------------------
    # Annualised normal volatility in basis points: the number read off a
    # screen every day, so a value wrong by 10x is caught by eye. Converting to
    # a per-step sigma divides by sqrt(trading_days_per_year) and then by the
    # length of a trading day. The day length is an explicit field because it
    # is a judgement for a globally traded instrument and it moves the session
    # stdev by nearly 2x between 24h and 6.5h.
    vol_annual_bp: float = 90.0
    trading_hours_per_day: float = 24.0
    trading_days_per_year: int = 252

    # -- Scheduled events ------------------------------------------------------
    # 0..max_events data prints per session, count uniform, times uniform over
    # the session. Timing is public; the outcome is hidden.
    max_events: int = 2
    # Print outcome: sign symmetric, magnitude uniform on [min, max] bp. Below
    # ~2bp a jump is indistinguishable from diffusion noise.
    jump_size_min_bp: float = 2.0
    jump_size_max_bp: float = 4.0

    # -- Pre-event drift -------------------------------------------------------
    # P(drift direction == print sign). An agreement probability, not a
    # correlation coefficient: for +/-1 variables corr = 2p - 1, so 0.60 here
    # is a 0.20 correlation. Must stay meaningfully below 1 or "follow the
    # drift" becomes a free edge.
    drift_prob_agree: float = 0.60
    # Duration is drawn first, then the window is placed uniformly in whatever
    # room exists before the print. Magnitude is the total move over the
    # window; against ~0.65bp of diffusion noise over 20 minutes it is visible
    # but not free.
    drift_duration_min_s: float = 10 * 60.0
    drift_duration_max_s: float = 40 * 60.0
    drift_magnitude_min_bp: float = 1.0
    drift_magnitude_max_bp: float = 2.0

    # -- Screen ----------------------------------------------------------------
    # Fair value plus observation noise (stdev, bp -- about one tick), sampled
    # every screen_refresh_s. A coarse anchor, not something to quote off.
    screen_noise_bp: float = 0.5
    screen_refresh_s: float = 5.0

    # -- Mark-out horizons -----------------------------------------------------
    # 10s immediate toxicity, 60s a drift regime becomes perceptible, 300s
    # drift-driven adverse selection fully shows. Event-relative and
    # session-end marks need no config. Each must be a whole number of steps
    # and must not exceed the session.
    markout_horizons_s: tuple[float, ...] = (10.0, 60.0, 300.0)

    def __post_init__(self) -> None:
        # `not x > 0` rather than `x <= 0`: the former also rejects NaN.
        if not self.tick_size > 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if not self.tick_value > 0:
            raise ValueError(f"tick_value must be positive, got {self.tick_value}")
        if not self.lot_size >= 1:
            raise ValueError(f"lot_size must be at least 1, got {self.lot_size}")
        if not math.isfinite(self.start_level):
            raise ValueError(f"start_level must be finite, got {self.start_level}")
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

        # Diffusion. Zero vol is a coherent (if dull) scenario -- pure events --
        # and is loud when eyeballed. Negative vol is the silent one: sigma * z
        # with sigma < 0 is statistically identical to sigma > 0.
        if not (math.isfinite(self.vol_annual_bp) and self.vol_annual_bp >= 0):
            raise ValueError(
                f"vol_annual_bp must be finite and non-negative, got {self.vol_annual_bp}"
            )
        if not 0 < self.trading_hours_per_day <= 24:
            raise ValueError(
                f"trading_hours_per_day must be in (0, 24], got {self.trading_hours_per_day}"
            )
        if not self.trading_days_per_year > 0:
            raise ValueError(
                f"trading_days_per_year must be positive, got {self.trading_days_per_year}"
            )

        # Events.
        if not self.max_events >= 0:
            raise ValueError(f"max_events must be non-negative, got {self.max_events}")
        if not 0 <= self.jump_size_min_bp <= self.jump_size_max_bp:
            raise ValueError(
                "jump sizes must satisfy 0 <= min <= max, got "
                f"{self.jump_size_min_bp} and {self.jump_size_max_bp}"
            )

        # Drift.
        if not 0 <= self.drift_prob_agree <= 1:
            raise ValueError(
                f"drift_prob_agree must be a probability, got {self.drift_prob_agree}"
            )
        if not 0 < self.drift_duration_min_s <= self.drift_duration_max_s:
            raise ValueError(
                "drift durations must satisfy 0 < min <= max, got "
                f"{self.drift_duration_min_s} and {self.drift_duration_max_s}"
            )
        if not 0 <= self.drift_magnitude_min_bp <= self.drift_magnitude_max_bp:
            raise ValueError(
                "drift magnitudes must satisfy 0 <= min <= max, got "
                f"{self.drift_magnitude_min_bp} and {self.drift_magnitude_max_bp}"
            )

        # Screen.
        if not (math.isfinite(self.screen_noise_bp) and self.screen_noise_bp >= 0):
            raise ValueError(
                f"screen_noise_bp must be finite and non-negative, got {self.screen_noise_bp}"
            )
        if not 0 < self.screen_refresh_s <= self.session_length:
            raise ValueError(
                f"screen_refresh_s must be in (0, session_length], got {self.screen_refresh_s}"
            )
        _whole_steps(self.screen_refresh_s, self.dt, "screen_refresh_s")

        # Mark-outs. Normalised to a tuple of floats so a config that came back
        # through JSON (where the tuple became a list) compares equal.
        horizons = tuple(float(h) for h in self.markout_horizons_s)
        object.__setattr__(self, "markout_horizons_s", horizons)
        for h in horizons:
            if not 0 < h <= self.session_length:
                raise ValueError(
                    f"markout horizon {h} must be in (0, session_length]"
                )
            _whole_steps(h, self.dt, f"markout horizon {h}")

    @property
    def n_steps(self) -> int:
        """Number of fair value grid steps in the session.

        Read this rather than recomputing ``session_length / dt``: it is the
        rounded, validated count, so callers cannot disagree on the grid size
        through a different rounding rule.
        """
        return round(self.session_length / self.dt)

    @property
    def screen_refresh_steps(self) -> int:
        """Grid steps between screen refreshes, validated whole."""
        return round(self.screen_refresh_s / self.dt)

    @property
    def markout_horizons_steps(self) -> tuple[int, ...]:
        """Mark-out horizons in grid steps, validated whole, config order."""
        return tuple(round(h / self.dt) for h in self.markout_horizons_s)


DEFAULT_CONFIG = GameConfig()
