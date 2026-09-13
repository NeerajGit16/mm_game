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
    # Display convention: one price point is 100bp, so a level of 9553.5 shows
    # as 95.535 and a half-bp tick as 0.005. Recorded here so the conversion
    # lives in one place; the model itself never uses it. Display is UI work.
    bp_per_point: float = 100.0
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
    # 0..max_events data prints per session, count uniform. Timing is public;
    # the outcome is hidden. Times are uniform over the window
    # [event_min_time_s, session_length - event_end_margin_s] subject to a
    # minimum gap between prints: no print before the player has a baseline
    # and a full minimum-length drift can fit, none so late that there is no
    # aftermath to trade and the mark-outs are all truncated, and no two
    # prints closer than a real calendar puts them.
    max_events: int = 2
    event_min_time_s: float = 15 * 60.0
    event_end_margin_s: float = 10 * 60.0
    event_min_gap_s: float = 30 * 60.0
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
    # The window ends at the print and its duration carries the randomness, so
    # the start is somewhere 10-40 minutes out and there is no fixed lead time
    # to learn. Magnitude is the total move over the window; against ~0.65bp
    # of diffusion noise over 20 minutes it is visible but not free.
    drift_duration_min_s: float = 10 * 60.0
    drift_duration_max_s: float = 40 * 60.0
    drift_magnitude_min_bp: float = 1.0
    drift_magnitude_max_bp: float = 2.0

    # -- Screen ----------------------------------------------------------------
    # Fair value plus a persistent observation error, sampled every
    # screen_refresh_s. The error is AR(1) with stationary stdev
    # screen_noise_bp (about one tick) and correlation time
    # screen_noise_corr_s: locally unaverageable -- a minute of readings is
    # worth about one reading -- and globally bounded, since the stationary
    # stdev never grows. Independent per-refresh noise was tried first and a
    # minute of averaging pinned fair value to a third of a tick; 0 here
    # recovers that white-noise screen for comparison.
    screen_noise_bp: float = 0.5
    screen_noise_corr_s: float = 90.0
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
        if not (math.isfinite(self.bp_per_point) and self.bp_per_point > 0):
            raise ValueError(
                f"bp_per_point must be finite and positive, got {self.bp_per_point}"
            )
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
        if not (math.isfinite(self.event_min_time_s) and self.event_min_time_s >= 0):
            raise ValueError(
                f"event_min_time_s must be finite and non-negative, got {self.event_min_time_s}"
            )
        if not (math.isfinite(self.event_end_margin_s) and self.event_end_margin_s >= 0):
            raise ValueError(
                f"event_end_margin_s must be finite and non-negative, got {self.event_end_margin_s}"
            )
        if not (math.isfinite(self.event_min_gap_s) and self.event_min_gap_s >= 0):
            raise ValueError(
                f"event_min_gap_s must be finite and non-negative, got {self.event_min_gap_s}"
            )
        if self.max_events > 0:
            lo, hi = self.event_step_bounds
            if hi < lo:
                raise ValueError(
                    "event window is empty: session_length - event_end_margin_s - "
                    f"event_min_time_s leaves steps [{lo}, {hi}]"
                )
            # Every count up to max_events must be placeable with the minimum
            # gap, in steps, exactly as the calendar draw computes it -- or the
            # draw would fail for some seeds and not others.
            need = self.max_events - 1
            if hi - lo - need * self.event_min_gap_steps < need:
                raise ValueError(
                    f"{self.max_events} events with a {self.event_min_gap_s}s minimum "
                    f"gap do not fit in the event window [{self.event_min_time_s}s, "
                    f"{self.session_length - self.event_end_margin_s}s]"
                )
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
        if not (math.isfinite(self.screen_noise_corr_s) and self.screen_noise_corr_s >= 0):
            raise ValueError(
                f"screen_noise_corr_s must be finite and non-negative, got {self.screen_noise_corr_s}"
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

    @property
    def event_step_bounds(self) -> tuple[int, int]:
        """Inclusive step range a print may land on. Never step 0: the jump
        lands on the increment into its step, and step 0 has none."""
        lo = max(1, round(self.event_min_time_s / self.dt))
        hi = self.n_steps - round(self.event_end_margin_s / self.dt)
        return lo, hi

    @property
    def event_min_gap_steps(self) -> int:
        return round(self.event_min_gap_s / self.dt)


DEFAULT_CONFIG = GameConfig()
