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
    # of yield equivalent, higher is better for a long. A quarter of a basis
    # point per tick, as front SOFR futures trade; $12.50 a tick, so $50 per
    # basis point per lot.
    #
    # The tick is the price grid and nothing else. Every economic parameter
    # below -- spreads, tolerances, costs, margins -- is in bp, so a change of
    # tick moves where prices may sit, never what anything costs. Money is
    # dollars_per_bp_per_lot times bp-lots, never ticks times tick value.
    tick_size: float = 0.25
    tick_value: float = 12.5
    lot_size: int = 1
    # Display convention: one price point is 100bp, so a level of 9551.75
    # shows as 95.5175 and a quarter-bp tick as 0.0025. Recorded here so the
    # conversion lives in one place; the model never uses it. Display is UI work.
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
    # Number of data prints per session: event_count_probs[k] is P(k prints).
    # Timing is public; the outcome is hidden. Weighted toward one or two
    # prints so most sessions carry event risk; a print-free session is the
    # quiet baseline about one time in five. Times are uniform over the window
    # [event_min_time_s, session_length - event_end_margin_s] subject to a
    # minimum gap between prints: no print before the player has a baseline
    # and a full minimum-length drift can fit, none so late that there is no
    # aftermath to trade and the mark-outs are all truncated, and no two
    # prints closer than a real calendar puts them.
    event_count_probs: tuple[float, ...] = (0.2, 0.4, 0.4)
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
    # screen_noise_bp and correlation time screen_noise_corr_s: locally
    # unaverageable -- a minute of readings is worth about one reading -- and
    # globally bounded, since the stationary stdev never grows. Independent
    # per-refresh noise was tried first and a minute of averaging pinned fair
    # value to 0.14bp; 0 here recovers that white-noise screen for comparison.
    screen_noise_bp: float = 0.5
    screen_noise_corr_s: float = 90.0
    screen_refresh_s: float = 5.0

    # -- Mark-out horizons -----------------------------------------------------
    # 10s immediate toxicity, 60s a drift regime becomes perceptible, 300s
    # drift-driven adverse selection fully shows. Event-relative and
    # session-end marks need no config. Each must be a whole number of steps
    # and must not exceed the session.
    markout_horizons_s: tuple[float, ...] = (10.0, 60.0, 300.0)

    # -- Flow: arrivals --------------------------------------------------------
    # Homogeneous Poisson, as a per-step Bernoulli at rate * dt on the grid.
    # ~200 RFQs in a two-hour session. Volume does not rise into prints.
    rfq_rate_per_hour: float = 100.0

    # -- Flow: roster ----------------------------------------------------------
    # A named roster per session, with explicit counts per visible category;
    # see SESSION_PRESETS for the 30/60/120-minute values. Client ids run
    # hedge funds first, then real money, then corporates.
    n_hedge_funds: int = 6
    n_real_money: int = 6
    n_corporates: int = 4
    # P(informed type | category). One uniform per client is compared against
    # these, so raising one only adds informed clients. Defaults give an
    # expected informed share of ~25% on every preset, with no category at 0
    # or 1: the label sets a prior and flow confirms or refutes it. Validation
    # allows [0, 1] so "sweep the informed share from 0 to 1" stays
    # expressible as an experiment.
    informed_prob_hedge_fund: float = 0.50
    informed_prob_real_money: float = 0.15
    informed_prob_corporate: float = 0.05
    # Weight on informed-type clients in client assignment inside drift
    # windows. Built now and inert at 1.0: enabling it later is a parameter
    # change, but building it later would move which client each in-window
    # RFQ belongs to. Above 1.0 it also makes client assignment depend on
    # hidden types, which is the point of it.
    informed_activity_multiplier: float = 1.0

    # -- Flow: informed signal -------------------------------------------------
    # An informed client sees the print's direction correctly with
    # probability informed_accuracy (q), and its magnitude times a lognormal
    # error of log-sd informed_signal_noise -- they know roughly how much.
    # Kept distinct from drift_prob_agree: if q equalled the crowd's 0.60,
    # "informed" would mean nothing.
    informed_accuracy: float = 0.80
    informed_signal_noise: float = 0.25

    # -- Flow: size ------------------------------------------------------------
    # Uninformed: lognormal lots, median 10, log-sd 0.9 (90th pct ~32, 99th
    # ~80), integer, at least 1, capped. Informed: the same base draw times a
    # factor |signal| / informed_size_ref_bp clipped to [min, max] -- ~1x for
    # a 1.5bp signal, ~3x at 4.5bp, ~2x on average.
    size_median_lots: float = 10.0
    size_log_sd: float = 0.9
    size_cap_lots: int = 100
    informed_size_ref_bp: float = 1.5
    informed_size_factor_min: float = 1.0
    informed_size_factor_max: float = 3.0

    # -- Flow: direction -------------------------------------------------------
    # Uninformed side: a fair coin outside drift windows, this probability
    # toward the drift inside them. Separate from drift_prob_agree though both
    # default to 0.60. One-way RFQs come from both populations, more often
    # from informed, so a one-way inquiry is evidence, never proof.
    crowd_drift_tilt: float = 0.60
    one_way_prob_uninformed: float = 0.05
    one_way_prob_informed: float = 0.10

    # -- Fills: competing dealers ----------------------------------------------
    # Three dealers sharing one model. Each mid is the truth plus a share of
    # the screen's error (dealers watch the same screen, so when it reads
    # rich the street leans rich) plus the dealer's own persistent error. Both
    # components persist for minutes, so cover calibrates but cannot be
    # averaged down to the truth. Each quotes mid +/- a half-spread, rounded
    # to the nearest tick, widening for size and around prints -- public
    # information only. Dealers never adapt during a session.
    n_competitors: int = 3
    competitor_half_spread_bp: float = 0.5
    competitor_size_widen_bp: float = 0.5  # extra half-spread per 100 lots
    competitor_print_widen: float = 1.0  # half-spread x (1 + this) at a print
    competitor_print_window_s: float = 120.0  # widening tapers to 0 this far out
    competitor_screen_weight: float = 0.75
    competitor_error_bp: float = 0.15
    competitor_error_corr_s: float = 120.0

    # -- Fills: client limits --------------------------------------------------
    # Every RFQ carries the worst price the client accepts on their side.
    # Uninformed: the screen plus a tolerance, lognormal, growing mildly with
    # size: median 0.75bp, ~16% below 0.5bp, ~4% above 1.5bp. Informed: at
    # least that, and up to the screen plus a share of their expected move --
    # they give up roughly half their edge, so a wide fill is evidence, never
    # proof.
    client_tolerance_bp: float = 0.75
    client_tolerance_log_sd: float = 0.4
    client_tolerance_size_growth: float = 0.5  # tolerance x (1 + this * size / 100)
    informed_edge_given_up: float = 0.5

    # -- Hedge venue and closing -----------------------------------------------
    # Trade any size at the tick-rounded screen plus a per-lot cost that grows
    # with size and widens around prints. Priced off the screen, never the
    # truth. Validation proves the per-lot cost exceeds the furthest a
    # competitor quote can sit from its own mid -- half-spread plus half a
    # tick of rounding -- by hedge_margin_bp at every size up to the cap and
    # every distance from a print, so quote-fill-hedge is never riskless by
    # construction. The closing liquidation is the same cost times
    # close_liquidation_multiplier: the hedge you didn't do, at a worse time.
    hedge_cost_bp: float = 1.0
    hedge_impact_bp: float = 0.5  # extra per-lot cost per 100 lots
    hedge_print_widen: float = 1.0
    hedge_print_window_s: float = 120.0
    hedge_margin_bp: float = 0.25
    close_liquidation_multiplier: float = 1.5

    # -- Position limit --------------------------------------------------------
    # Hard limit either way, twice the ticket cap. A quote side whose fill
    # would breach it is blocked; hedges count toward it and are clipped.
    position_limit_lots: int = 200

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

        # Events. The count probabilities are normalised to a tuple of floats
        # so a config reloaded from JSON compares equal, and must already sum
        # to 1: renormalising silently would hide a typo in a weight.
        probs = tuple(float(p) for p in self.event_count_probs)
        object.__setattr__(self, "event_count_probs", probs)
        if not probs:
            raise ValueError("event_count_probs must not be empty")
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in probs):
            raise ValueError(f"event_count_probs must be probabilities, got {probs}")
        if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"event_count_probs must sum to 1, got {sum(probs)}")
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
            # Every count with non-zero probability must be placeable with the
            # minimum gap, in steps, exactly as the calendar draw computes it --
            # or the draw would fail for some seeds and not others. Fewer
            # prints need less room, so the largest allowed count decides it.
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

        # Flow: arrivals. The per-step Bernoulli probability must be a
        # probability; it should also be small for the Poisson reading to hold.
        if not (math.isfinite(self.rfq_rate_per_hour) and self.rfq_rate_per_hour >= 0):
            raise ValueError(
                f"rfq_rate_per_hour must be finite and non-negative, got {self.rfq_rate_per_hour}"
            )
        if self.rfq_arrival_prob > 1:
            raise ValueError(
                f"rfq_rate_per_hour * dt is {self.rfq_arrival_prob} per step; must be <= 1"
            )

        # Flow: roster.
        for name in ("n_hedge_funds", "n_real_money", "n_corporates"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not self.n_clients >= 1:
            raise ValueError("the roster must have at least one client")
        for name in ("informed_prob_hedge_fund", "informed_prob_real_money", "informed_prob_corporate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability, got {getattr(self, name)}")
        if not (math.isfinite(self.informed_activity_multiplier) and self.informed_activity_multiplier > 0):
            raise ValueError(
                "informed_activity_multiplier must be finite and positive, got "
                f"{self.informed_activity_multiplier}"
            )

        # Flow: informed signal.
        if not 0 <= self.informed_accuracy <= 1:
            raise ValueError(f"informed_accuracy must be a probability, got {self.informed_accuracy}")
        if not (math.isfinite(self.informed_signal_noise) and self.informed_signal_noise >= 0):
            raise ValueError(
                f"informed_signal_noise must be finite and non-negative, got {self.informed_signal_noise}"
            )

        # Flow: size.
        if not (math.isfinite(self.size_median_lots) and self.size_median_lots > 0):
            raise ValueError(f"size_median_lots must be finite and positive, got {self.size_median_lots}")
        if not (math.isfinite(self.size_log_sd) and self.size_log_sd >= 0):
            raise ValueError(f"size_log_sd must be finite and non-negative, got {self.size_log_sd}")
        if not self.size_cap_lots >= 1:
            raise ValueError(f"size_cap_lots must be at least 1, got {self.size_cap_lots}")
        if not (math.isfinite(self.informed_size_ref_bp) and self.informed_size_ref_bp > 0):
            raise ValueError(
                f"informed_size_ref_bp must be finite and positive, got {self.informed_size_ref_bp}"
            )
        if not 0 < self.informed_size_factor_min <= self.informed_size_factor_max < math.inf:
            raise ValueError(
                "informed size factors must satisfy 0 < min <= max < inf, got "
                f"{self.informed_size_factor_min} and {self.informed_size_factor_max}"
            )

        # Flow: direction.
        for name in ("crowd_drift_tilt", "one_way_prob_uninformed", "one_way_prob_informed"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability, got {getattr(self, name)}")

        # Fills: competitors.
        if not self.n_competitors >= 1:
            raise ValueError(f"n_competitors must be at least 1, got {self.n_competitors}")
        if not (
            math.isfinite(self.competitor_half_spread_bp)
            and self.competitor_half_spread_bp >= self.tick_size / 2
        ):
            # Below half a tick, rounding to the nearest tick could put a
            # dealer's bid at or above its own mid. The one place the grid
            # constrains an economic parameter, and it says so.
            raise ValueError(
                f"competitor_half_spread_bp must be at least half a tick ({self.tick_size / 2}bp), "
                f"got {self.competitor_half_spread_bp}"
            )
        for name in (
            "competitor_size_widen_bp",
            "competitor_print_widen",
            "competitor_print_window_s",
            "competitor_error_bp",
            "competitor_error_corr_s",
            "client_tolerance_log_sd",
            "client_tolerance_size_growth",
            "hedge_impact_bp",
            "hedge_print_widen",
            "hedge_print_window_s",
            "hedge_margin_bp",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not 0 <= self.competitor_screen_weight <= 1:
            raise ValueError(
                f"competitor_screen_weight must be in [0, 1], got {self.competitor_screen_weight}"
            )

        # Fills: client limits.
        if not (math.isfinite(self.client_tolerance_bp) and self.client_tolerance_bp > 0):
            raise ValueError(
                f"client_tolerance_bp must be finite and positive, got {self.client_tolerance_bp}"
            )
        if not 0 <= self.informed_edge_given_up <= 1:
            raise ValueError(
                f"informed_edge_given_up must be in [0, 1], got {self.informed_edge_given_up}"
            )

        # Hedge venue and closing.
        if not (math.isfinite(self.hedge_cost_bp) and self.hedge_cost_bp > 0):
            raise ValueError(f"hedge_cost_bp must be finite and positive, got {self.hedge_cost_bp}")
        if not (math.isfinite(self.close_liquidation_multiplier) and self.close_liquidation_multiplier >= 1):
            raise ValueError(
                "close_liquidation_multiplier must be at least 1, got "
                f"{self.close_liquidation_multiplier}"
            )
        shortfall = self.hedge_margin_shortfall()
        if shortfall is not None:
            size, seconds, gap = shortfall
            raise ValueError(
                "hedge cost must exceed the furthest competitor quote by "
                f"hedge_margin_bp={self.hedge_margin_bp} everywhere; at size {size} "
                f"and {seconds}s from a print the excess is {gap:.4f}bp"
            )

        # Position limit.
        if not self.position_limit_lots >= self.size_cap_lots:
            # Below the ticket cap, a single fill from flat could breach it.
            raise ValueError(
                f"position_limit_lots ({self.position_limit_lots}) must be at least "
                f"size_cap_lots ({self.size_cap_lots})"
            )

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

    @property
    def max_events(self) -> int:
        """The largest print count with non-zero probability: the one the
        placement check has to prove fits."""
        return max(k for k, p in enumerate(self.event_count_probs) if p > 0)

    # -- Spread and cost functions ---------------------------------------------
    # Defined here, on the parameters, so validation, the competitors and the
    # hedge venue all use one definition. Arguments are public: the ticket
    # size and the distance in seconds to the nearest print (inf if none).

    @staticmethod
    def _proximity(seconds_to_print: float, window_s: float) -> float:
        """1 at a print, tapering linearly to 0 at ``window_s`` either side."""
        if window_s <= 0 or not math.isfinite(seconds_to_print):
            return 0.0
        return max(0.0, 1.0 - abs(seconds_to_print) / window_s)

    def competitor_half_spread_bp_at(self, size: float, seconds_to_print: float) -> float:
        """A dealer's half-spread in bp before rounding: base plus size, times
        the widening around a print."""
        base = self.competitor_half_spread_bp + self.competitor_size_widen_bp * size / 100.0
        widen = 1.0 + self.competitor_print_widen * self._proximity(seconds_to_print, self.competitor_print_window_s)
        return base * widen

    def competitor_max_quote_distance_bp(self, size: float, seconds_to_print: float) -> float:
        """The furthest a dealer's rounded quote can sit from its own mid, bp:
        the half-spread plus half a tick of rounding to the nearest tick."""
        return self.competitor_half_spread_bp_at(size, seconds_to_print) + self.tick_size / 2

    def hedge_cost_bp_at(self, size: float, seconds_to_print: float) -> float:
        """Per-lot hedge cost in bp: base plus impact growing with size, times
        the widening around a print."""
        base = self.hedge_cost_bp + self.hedge_impact_bp * size / 100.0
        widen = 1.0 + self.hedge_print_widen * self._proximity(seconds_to_print, self.hedge_print_window_s)
        return base * widen

    def hedge_margin_shortfall(self) -> tuple[float, float, float] | None:
        """Where hedging is not dearer than the furthest competitor quote by
        ``hedge_margin_bp``, if anywhere: (size, seconds, excess in bp), else None.

        Exact, not sampled. For a fixed distance both functions are linear in
        size, so the excess is smallest at size 1 or at the cap; for a fixed
        size both are piecewise linear in distance with kinks only at the
        print and at each widening window's edge, so the excess is smallest at
        one of those points or far away. Checking that handful of corners
        covers every size and every distance.
        """
        distances = (0.0, self.competitor_print_window_s, self.hedge_print_window_s, math.inf)
        corners = [
            (
                size,
                seconds,
                self.hedge_cost_bp_at(size, seconds)
                - self.competitor_max_quote_distance_bp(size, seconds),
            )
            for size in (1.0, float(self.size_cap_lots))
            for seconds in distances
        ]
        worst = min(corners, key=lambda c: c[2])
        return worst if worst[2] < self.hedge_margin_bp - 1e-12 else None

    @property
    def dollars_per_bp_per_lot(self) -> float:
        """The one conversion from bp-lots to money. P&L is computed in bp-lots
        and converted here, so the tick never enters an economic number."""
        return self.tick_value / self.tick_size

    @property
    def rfq_arrival_prob(self) -> float:
        """Per-step Bernoulli probability of an RFQ: rate * dt."""
        return self.rfq_rate_per_hour * self.dt / 3600.0

    @property
    def n_clients(self) -> int:
        return self.n_hedge_funds + self.n_real_money + self.n_corporates

    @property
    def expected_informed_share(self) -> float:
        """Expected fraction of the roster that is informed type."""
        expected = (
            self.n_hedge_funds * self.informed_prob_hedge_fund
            + self.n_real_money * self.informed_prob_real_money
            + self.n_corporates * self.informed_prob_corporate
        )
        return expected / self.n_clients

    @classmethod
    def for_session(cls, minutes: int, **overrides: object) -> GameConfig:
        """The config for a 30, 60 or 120-minute session.

        Presets are explicit values, not a formula: each sets session length,
        the maximum number of prints the event bounds can fit, and the roster
        counts per category. Roster size holds ~8-12 RFQs per client, which is
        what makes identity learnable; a 16-client roster in 30 minutes would
        be ~3 per client and effectively anonymous. Any field can still be
        overridden by keyword.
        """
        if minutes not in SESSION_PRESETS:
            raise ValueError(f"no preset for {minutes} minutes; have {sorted(SESSION_PRESETS)}")
        return cls(**{**SESSION_PRESETS[minutes], **overrides})


# Session-length presets. Explicit per preset so every value is visible and
# diffable. Expected informed share at the default category probabilities:
# 30 min 25.0%, 60 min 26.0%, 120 min 25.6%.
#
# The 30-minute preset is an EVENT DRILL, not a short version of the full
# game: exactly one print, landing between minute 10 and minute 20, with a
# 5-15 minute drift running into it, so every drill session carries one
# event to manage and some aftermath to trade. The 60- and 120-minute
# presets fit up to two prints under the default event bounds.
SESSION_PRESETS: dict[int, dict[str, object]] = {
    30: {
        "session_length": 1800,
        "event_count_probs": (0.0, 1.0),
        "event_min_time_s": 10 * 60.0,
        "event_end_margin_s": 10 * 60.0,
        "drift_duration_min_s": 5 * 60.0,
        "drift_duration_max_s": 15 * 60.0,
        "n_hedge_funds": 2,
        "n_real_money": 3,
        "n_corporates": 1,
    },
    60: {
        "session_length": 3600,
        "event_count_probs": (0.2, 0.4, 0.4),
        "n_hedge_funds": 4,
        "n_real_money": 3,
        "n_corporates": 3,
    },
    120: {
        "session_length": 7200,
        "event_count_probs": (0.2, 0.4, 0.4),
        "n_hedge_funds": 6,
        "n_real_money": 6,
        "n_corporates": 4,
    },
}


DEFAULT_CONFIG = GameConfig()
