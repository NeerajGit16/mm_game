"""Fair value: the price nobody sees.

Everything else in the game is defined against it. A quote is good or bad
relative to it, a fill is attributed as edge captured against it, and informed
flow is informed *about* it. It is generated once, in full, before the first
RFQ, so that "informed" can be implemented as reading further along an array
that already exists -- no extra randomness, no coupling to how many inquiries
arrived.

The process is arithmetic Brownian motion in basis-point units with zero drift
and no reversion, plus 0..2 scheduled prints per session. Each print is a
symmetric jump the market pre-positions for: a drift regime of random duration
and magnitude runs into it, agreeing with the print's sign only 60% of the
time. The public screen is the truth plus a persistent observation error that
a minute of averaging cannot remove, rounded to the tick. Nothing here is
exploitable without reading flow: no reversion level to lean on, no constant
drift, no asymmetry in the jumps, no screen you can average your way to the
truth from.

This module is a producer of data, not a service. ``generate`` builds a
``FairValuePath`` once; the engine holds it and hands each participant the
part they are entitled to see. The public/private split is structural:
``FairValuePath.public`` is a separate object carrying only the screen and
the event calendar, and it is the only thing that should ever reach an agent.

Every draw comes from a substream, so ``generate`` is a pure function of
``(config, seed)``: calling it twice on one ``RngStreams`` gives the same path,
and nothing drawn elsewhere can move it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mm_game.config import GameConfig
from mm_game.rng import RngStreams, Stream

# Diffusion and screen noise are keyed on instrument ordinal; the event
# calendar on market ordinal. There is one of each today. Keyed now, rather
# than drawn from the plain stream, because adding a second instrument later
# would otherwise shift instrument zero's path and invalidate every recorded
# session. Decisions that change generated numbers are made early; the
# ``Instrument`` dataclass that changes none can wait.
INSTRUMENT_ORDINAL: int = 0
MARKET_ORDINAL: int = 0


def sigma_per_step(config: GameConfig) -> float:
    """Per-step diffusion stdev in bp, written out so the units are auditable.

    Annual normal vol -> daily by sqrt(trading days) -> per second by sqrt of
    the trading day in seconds -> per step by sqrt(dt). At the defaults
    (90bp, 252 days, 24h, dt 0.1) this is ~0.0061bp per step and ~1.64bp over
    a two-hour session.
    """
    sigma_daily = config.vol_annual_bp / math.sqrt(config.trading_days_per_year)
    seconds_per_day = config.trading_hours_per_day * 3600.0
    sigma_per_second = sigma_daily / math.sqrt(seconds_per_day)
    return sigma_per_second * math.sqrt(config.dt)


def session_sigma(config: GameConfig) -> float:
    """Diffusion stdev over the whole session, bp. For display and sanity."""
    return sigma_per_step(config) * math.sqrt(config.n_steps)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftWindow:
    """A pre-event drift regime: a linear ramp of ``total_bp`` over
    ``(start_step, end_step]``. Hidden."""

    event_index: int
    start_step: int
    end_step: int
    total_bp: float


@dataclass(frozen=True)
class Move:
    """An unannounced move: a linear ramp of ``total_bp`` over
    ``(start_step, end_step]``, foretold by flow from ``inform_step``.
    Hidden, never on the calendar. ``candidate`` is the candidate ordinal it
    was drawn as -- its identity for anything keyed on it, stable when
    thinning keeps or drops other candidates; ``index`` is its place among
    the moves kept."""

    index: int
    candidate: int
    inform_step: int
    start_step: int
    end_step: int
    total_bp: float


@dataclass(frozen=True)
class MarkOut:
    """Fair value change from a step over a fixed horizon.

    ``truncated`` is set when the horizon ran past the session end, in which
    case ``change`` is measured to the last step instead. Aggregates must
    exclude truncated values: averaging them in makes late fills look harmless
    and adverse selection come out systematically flattering.
    """

    horizon_steps: int
    change: float
    truncated: bool


@dataclass(frozen=True)
class NextEventMarkOut:
    """Fair value change from a step through the next event: the next print
    or the end of the next unannounced move, whichever comes first.

    Measured to the event step itself, where the path already includes the
    jump or the whole ramp. ``kind`` is "print" or "move" and ``index`` its
    index among the prints or the moves. Everything is ``None`` when neither
    follows the step: that mark-out is undefined, not zero, and must be
    recorded as such.
    """

    kind: str | None
    index: int | None
    event_step: int | None
    change: float | None


@dataclass(frozen=True, eq=False)
class PublicView:
    """What a participant may see: the screen and the event calendar.

    "Public" means public *at its step*: the engine reveals each reading when
    its step arrives. The calendar is the exception, public in advance, since
    the countdown is on the info monitor.

    Holds no reference to the truth. Screen readings are stored per
    observation, not per step: one at every regular refresh plus one forced at
    every print, so ``screen_steps`` is irregular and the age of the reading a
    participant is looking at is ``step - screen_steps[i]``. Every reading is
    a whole number of ticks from the session's start level: the screen quotes
    in ticks, as a screen does.
    """

    n_steps: int
    dt: float
    refresh_steps: int
    event_steps: np.ndarray  # public in advance
    screen_steps: np.ndarray  # sorted observation steps: 0, r, 2r, ... plus each print
    screen: np.ndarray  # one reading per observation step

    def _check_step(self, step: int) -> None:
        if not 0 <= step <= self.n_steps:
            raise ValueError(f"step must be in [0, {self.n_steps}], got {step}")

    def _showing(self, step: int) -> int:
        """Index of the latest observation at or before ``step``."""
        self._check_step(step)
        return int(np.searchsorted(self.screen_steps, step, side="right")) - 1

    def screen_at(self, step: int) -> float:
        """The screen mid showing at ``step``: the latest observation at or before it."""
        return float(self.screen[self._showing(step)])

    def screen_age_s(self, step: int) -> float:
        """Seconds since the reading showing at ``step`` was taken."""
        return (step - int(self.screen_steps[self._showing(step)])) * self.dt

    def seconds_to_nearest_print(self, step: int) -> float:
        """Seconds to the nearest print before or after ``step``; inf if the
        session has none. Public: the calendar is on the info monitor, and
        dealers and the hedge venue widen on it."""
        self._check_step(step)
        if self.event_steps.size == 0:
            return math.inf
        return float(np.min(np.abs(self.event_steps - step))) * self.dt

    def steps_to_next_event(self, step: int) -> int | None:
        """Steps until the next print strictly after ``step``; None if none."""
        self._check_step(step)
        later = self.event_steps[self.event_steps > step]
        return int(later[0] - step) if later.size else None

    def to_record(self) -> dict[str, object]:
        """Plain-type calendar for the public half of the session record."""
        return {
            "event_steps": [int(e) for e in self.event_steps],
            "screen_refresh_steps": self.refresh_steps,
        }


@dataclass(frozen=True, eq=False)
class FairValuePath:
    """A session's truth, generated in full at session start. Frozen data.

    ``values`` is the absolute price. ``relative`` is ``values - start_level``,
    built as the exact sum of ``diffusion``, ``drift`` and the cumulative
    jumps; every difference this object reports is computed on ``relative``,
    so P&L numbers are bit-identical across any change of ``start_level``.
    Arrays are read-only: a frozen dataclass does not protect their contents,
    and an agent that wrote into the truth would corrupt every later mark-out.

    Step semantics: ``values[k]`` is the price after the k-th increment. A
    print at ``event_steps[i]`` means ``values[e]`` already includes the jump
    and ``values[e - 1]`` does not.
    """

    start_level: float
    dt: float
    values: np.ndarray  # (n_steps + 1,) absolute, bp units
    relative: np.ndarray  # (n_steps + 1,) values - start_level, exact
    diffusion: np.ndarray  # (n_steps + 1,) cumulative diffusion, starts at 0
    drift: np.ndarray  # (n_steps + 1,) cumulative drift, starts at 0
    moves: np.ndarray  # (n_steps + 1,) cumulative unannounced moves, starts at 0. Hidden.
    event_steps: np.ndarray  # (n_events,) sorted, unique, in [1, n_steps]
    jump_sizes: np.ndarray  # (n_events,) signed bp. Hidden.
    drift_windows: tuple[DriftWindow, ...]
    move_list: tuple[Move, ...]  # hidden
    # Unrounded observation error at each public.screen_steps entry. Hidden:
    # with the truth it reconstructs the screen before rounding, and the
    # competitor pricing in fills shares this error.
    screen_error: np.ndarray
    public: PublicView

    @property
    def n_steps(self) -> int:
        return len(self.values) - 1

    def _check_step(self, step: int) -> None:
        if not 0 <= step <= self.n_steps:
            raise ValueError(f"step must be in [0, {self.n_steps}], got {step}")

    def at(self, step: int) -> float:
        """True fair value at ``step``."""
        self._check_step(step)
        return float(self.values[step])

    def lookahead(self, step: int, horizon_steps: int) -> float:
        """``fair(step + horizon) - fair(step)``, clamped at the session end.

        Clamping is the point: without it the last RFQs of a session would
        behave differently for reasons that have nothing to do with the model.
        ``lookahead(k, 0)`` is exactly 0.
        """
        self._check_step(step)
        if horizon_steps < 0:
            raise ValueError(f"horizon_steps must be non-negative, got {horizon_steps}")
        end = min(step + horizon_steps, self.n_steps)
        return float(self.relative[end] - self.relative[step])

    def jumps_within(self, step: int, horizon_steps: int) -> np.ndarray:
        """Signed sizes of prints landing in ``(step, step + horizon]``, clamped."""
        self._check_step(step)
        if horizon_steps < 0:
            raise ValueError(f"horizon_steps must be non-negative, got {horizon_steps}")
        end = min(step + horizon_steps, self.n_steps)
        mask = (self.event_steps > step) & (self.event_steps <= end)
        return self.jump_sizes[mask]

    def markout(self, step: int, horizons_steps: Sequence[int]) -> tuple[MarkOut, ...]:
        """Fair value change over each horizon, each carrying a truncation flag."""
        self._check_step(step)
        out = []
        for h in horizons_steps:
            if h < 0:
                raise ValueError(f"horizon must be non-negative, got {h}")
            truncated = step + h > self.n_steps
            end = min(step + h, self.n_steps)
            out.append(
                MarkOut(
                    horizon_steps=int(h),
                    change=float(self.relative[end] - self.relative[step]),
                    truncated=truncated,
                )
            )
        return tuple(out)

    def next_event_markout(self, step: int) -> NextEventMarkOut:
        """Change from ``step`` through the first print or move end strictly
        after it, or undefined when neither follows.

        A move's informing window never overlaps a print's run-up, so a fill
        informed about a print marks through that print and one informed
        about a move marks through the end of that move.
        """
        self._check_step(step)
        candidates = []
        later = np.flatnonzero(self.event_steps > step)
        if later.size:
            i = int(later[0])
            candidates.append((int(self.event_steps[i]), "print", i))
        ends = [(m.end_step, "move", m.index) for m in self.move_list if m.end_step > step]
        if ends:
            candidates.append(min(ends))
        if not candidates:
            return NextEventMarkOut(kind=None, index=None, event_step=None, change=None)
        e, kind, index = min(candidates)
        return NextEventMarkOut(
            kind=kind,
            index=index,
            event_step=e,
            change=float(self.relative[e] - self.relative[step]),
        )

    def to_record(self) -> dict[str, object]:
        """Plain-type summary including hidden fields. Post-session log only."""
        return {
            "start_level": self.start_level,
            "event_steps": [int(e) for e in self.event_steps],
            "jump_sizes_bp": [float(j) for j in self.jump_sizes],
            "drift_windows": [
                {
                    "event_index": w.event_index,
                    "start_step": w.start_step,
                    "end_step": w.end_step,
                    "total_bp": w.total_bp,
                }
                for w in self.drift_windows
            ],
            "moves": [
                {
                    "index": m.index,
                    "candidate": m.candidate,
                    "inform_step": m.inform_step,
                    "start_step": m.start_step,
                    "end_step": m.end_step,
                    "total_bp": m.total_bp,
                }
                for m in self.move_list
            ],
            "close": float(self.values[-1]),
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _draw_calendar(config: GameConfig, gen: np.random.Generator) -> np.ndarray:
    """How many prints and when.

    Count is drawn from ``event_count_probs`` by inverse CDF on one uniform,
    written out rather than left to ``Generator.choice`` so the mapping from
    bits to a count is ours and cannot change with numpy. A zero-weight count
    is never chosen. Times are uniform over the event window subject to the
    minimum gap: draw distinct steps from the window shrunk by the total gap,
    sort, then push each later print out by its share of the gaps. That is
    the standard construction and it is exactly uniform over the constrained
    set. Step 0 is never a print: the jump would land on the increment before
    it, which does not exist.
    """
    cdf = np.cumsum(config.event_count_probs)
    count = int(np.searchsorted(cdf, gen.random(), side="right"))
    # A cdf ending a few ulp below 1 could overrun; clamp to the largest count
    # that has weight, never to a trailing zero-weight one.
    count = min(count, config.max_events)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    lo, hi = config.event_step_bounds
    gap = config.event_min_gap_steps
    room = hi - lo - (count - 1) * gap
    if room + 1 < count:
        raise ValueError(
            f"cannot place {count} prints with a {gap}-step gap in steps [{lo}, {hi}]"
        )
    picks = np.sort(gen.choice(room + 1, size=count, replace=False))
    steps = lo + picks + np.arange(count) * gap
    return steps.astype(np.int64)


def _draw_jump(config: GameConfig, gen: np.random.Generator) -> tuple[float, float]:
    """One print's outcome: (sign, magnitude). Sign is drawn first and from a
    single uniform, so a change to the size distribution -- which consumes the
    draws after it -- leaves every sign, and hence every drift direction,
    exactly where it was."""
    sign = 1.0 if gen.random() < 0.5 else -1.0
    magnitude = float(gen.uniform(config.jump_size_min_bp, config.jump_size_max_bp))
    return sign, magnitude


def _draw_drift(
    config: GameConfig, gen: np.random.Generator, event_index: int, event_step: int, jump_sign: float
) -> DriftWindow:
    """The pre-positioning regime running into one print.

    The window ends at the print. A drift that finishes and flat-lines for ten
    minutes before the release is not pre-positioning. Duration carries the
    randomness, so the start is still somewhere 10-40 minutes out and there is
    no fixed lead time to learn. Draw order is direction, duration, magnitude.
    If the print is too early for the drawn duration the window is shortened
    to start at step 0, keeping the per-step rate rather than the total: a
    20-minute 1.5bp drift squeezed into 3 minutes at full magnitude would be a
    signal no market gives, and a print in the first seconds gets a negligible
    drift, which is the "skip" end of the rule.
    """
    agrees = gen.random() < config.drift_prob_agree
    direction = jump_sign if agrees else -jump_sign
    duration_s = float(gen.uniform(config.drift_duration_min_s, config.drift_duration_max_s))
    magnitude = float(gen.uniform(config.drift_magnitude_min_bp, config.drift_magnitude_max_bp))
    duration_steps = max(1, round(duration_s / config.dt))
    if duration_steps > event_step:
        magnitude *= event_step / duration_steps
        duration_steps = event_step
    return DriftWindow(
        event_index=event_index,
        start_step=event_step - duration_steps,
        end_step=event_step,
        total_bp=direction * magnitude,
    )


# Substream keys for MOVES: the candidate count on its own key, each
# candidate on the next ones, so no candidate's draws depend on the count or
# on any other candidate.
_MOVE_COUNT_KEY: int = 0


def _poisson_count(lam: float, u: float) -> int:
    """Inverse CDF of a Poisson(lam) at one uniform. Written out rather
    than left to ``Generator.poisson`` so raising the rate only ever adds
    candidates: for a fixed uniform the count is monotone in the rate."""
    k, p = 0, math.exp(-lam)
    cdf = p
    while u > cdf and k < 10_000:
        k += 1
        p *= lam / k
        cdf += p
    return k


def free_intervals(
    config: GameConfig, windows: Sequence[DriftWindow], event_steps: np.ndarray
) -> list[tuple[int, int]]:
    """Where a move's informing window may lie, as half-open step intervals:
    the session less its first and last edge margin, less each print's
    run-up and the clearance after the print."""
    n = config.n_steps
    edge = round(config.move_edge_margin_s / config.dt)
    clearance = round(config.move_print_clearance_s / config.dt)
    blocked = sorted((w.start_step, int(e) + clearance) for w, e in zip(windows, event_steps))
    free, cursor = [], edge
    for lo, hi in blocked:
        if lo > cursor:
            free.append((cursor, min(lo, n - edge)))
        cursor = max(cursor, hi)
    if cursor < n - edge:
        free.append((cursor, n - edge))
    return [(a, b) for a, b in free if b > a]


def _subtract(intervals: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    """Half-open intervals less [lo, hi)."""
    out = []
    for a, b in intervals:
        if hi <= a or b <= lo:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def _draw_moves(
    config: GameConfig, rng: RngStreams, windows: Sequence[DriftWindow], event_steps: np.ndarray
) -> tuple[Move, ...]:
    """Unannounced moves over free time.

    The count is Poisson at the rate per free hour times the free hours, by
    inverse CDF on its own key, so raising the rate only ever adds
    candidates. Every candidate draws a placement uniform, sign, magnitude,
    duration and lead, in that order, from its own substream whether or not
    it is kept -- so no candidate's draws depend on another's, and nothing
    here touches a print, jump, drift or screen stream. The placement puts
    the whole informing window uniformly over the positions where it fits in
    free time, and a kept move takes its window and the minimum gap either
    side out of free time for the candidates after it. Only a candidate that
    fits nowhere is dropped. Deterministic given the draws and the calendar.
    """
    free = free_intervals(config, windows, event_steps)
    free_hours = sum(b - a for a, b in free) * config.dt / 3600.0
    lam = config.move_rate_per_free_hour * free_hours
    if lam <= 0:
        return ()
    count = _poisson_count(lam, float(rng.substream(Stream.MOVES, _MOVE_COUNT_KEY).random()))
    dt = config.dt
    gap = round(config.move_min_gap_s / dt)
    kept: list[Move] = []
    for c in range(count):
        gen = rng.substream(Stream.MOVES, c + 1)
        u = float(gen.random())
        sign = 1.0 if gen.random() < 0.5 else -1.0
        magnitude = float(gen.uniform(config.move_magnitude_min_bp, config.move_magnitude_max_bp))
        duration = max(1, round(float(gen.uniform(config.move_duration_min_s, config.move_duration_max_s)) / dt))
        lead = round(float(gen.uniform(config.move_lead_min_s, config.move_lead_max_s)) / dt)
        width = lead + duration
        # Informing-window starts that fit: [a, b - width] in each free interval.
        room = [(a, b - width) for a, b in free if b - width > a]
        total = sum(hi - lo for lo, hi in room)
        if total <= 0:
            continue
        x = u * total
        for lo, hi in room:
            if x < hi - lo:
                inform = lo + int(x)
                break
            x -= hi - lo
        else:
            lo, hi = room[-1]
            inform = hi - 1
        start, end = inform + lead, inform + width
        kept.append(Move(index=len(kept), candidate=c, inform_step=inform, start_step=start, end_step=end,
                         total_bp=sign * magnitude))
        # A kept move blocks its own window and the gap either side of it for
        # every later candidate, so the gap rule thins nothing: it shapes
        # where later moves can go.
        free = _subtract(free, inform - gap, end + gap)
    kept.sort(key=lambda m: m.start_step)
    return tuple(Move(index=i, candidate=m.candidate, inform_step=m.inform_step, start_step=m.start_step,
                      end_step=m.end_step, total_bp=m.total_bp) for i, m in enumerate(kept))


def _move_increments(n_steps: int, moves: Sequence[Move]) -> np.ndarray:
    inc = np.zeros(n_steps)
    for m in moves:
        inc[m.start_step : m.end_step] += m.total_bp / (m.end_step - m.start_step)
    return inc


def _drift_increments(n_steps: int, windows: Sequence[DriftWindow]) -> np.ndarray:
    """Per-step drift increments. Overlapping windows add."""
    inc = np.zeros(n_steps)
    for w in windows:
        inc[w.start_step : w.end_step] += w.total_bp / (w.end_step - w.start_step)
    return inc


def _jump_increments(n_steps: int, event_steps: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Per-step jump increments: the gap lands on the increment into the
    event step, so ``values[e]`` includes it and ``values[e - 1]`` does not."""
    inc = np.zeros(n_steps)
    np.add.at(inc, event_steps - 1, sizes)
    return inc


def _screen_error(config: GameConfig, z: np.ndarray) -> np.ndarray:
    """Persistent observation error at each refresh: a stationary AR(1).

    ``e[0] ~ N(0, s^2)`` and ``e[k] = phi * e[k-1] + s * sqrt(1 - phi^2) * z[k]``
    with ``phi = exp(-refresh / corr_time)``, so every reading has stdev ``s``
    and readings ``t`` apart correlate as ``exp(-t / corr_time)``. Starting
    from the stationary distribution matters: starting at zero would make the
    first minutes of every session the most accurate ones.

    The point, against independent noise at the same stdev: the variance of
    an N-reading average is ``s^2/N * (1 + 2 * sum_k (1 - k/N) phi^k)``. At
    5 s refresh and a 90 s correlation time, a one-minute average of twelve
    readings has stdev 0.90 s rather than 0.29 s -- worth 1.2 independent
    readings, not twelve. Five minutes of averaging gets to 0.65 s, by which
    time the truth has moved further than that. Averaging does not get you
    the truth.
    """
    s = config.screen_noise_bp
    phi = _screen_corr(config, config.screen_refresh_s)
    innovation = s * math.sqrt(1.0 - phi * phi)
    err = np.empty(len(z))
    err[0] = s * z[0]
    for k in range(1, len(z)):
        err[k] = phi * err[k - 1] + innovation * z[k]
    return err


def _screen_corr(config: GameConfig, seconds: float) -> float:
    """Correlation of the screen error across ``seconds``: exp(-t / tau).
    A correlation time of 0 is the white screen: zero correlation at any gap."""
    tau = config.screen_noise_corr_s
    return math.exp(-seconds / tau) if tau > 0 else 0.0


def _bridge_error(
    config: GameConfig, left_err: float, t_left: float, right_err: float | None, t_right: float, z: float
) -> float:
    """The screen error at an instant between two known observations.

    The error is a stationary Ornstein-Uhlenbeck process sampled at the
    observation times, and that process is Markov, so the value at an
    intermediate instant given everything else depends only on its two
    neighbours. With ``rho1 = exp(-t_left / tau)`` and
    ``rho2 = exp(-t_right / tau)``, the conditional law is Gaussian with

        mean = (a * rho1 * (1 - rho2^2) + b * rho2 * (1 - rho1^2)) / (1 - rho1^2 rho2^2)
        var  = s^2 * (1 - rho1^2) * (1 - rho2^2) / (1 - rho1^2 rho2^2)

    and with no right neighbour it is the forward step N(rho1 * a,
    s^2 (1 - rho1^2)). Sampling this way gives the observations jointly the
    law of the OU at their actual times -- so consecutive observations are
    AR(1) with coefficient exp(-gap / tau) for each actual gap -- while the
    regular refreshes, drawn first, are untouched by the insertion.
    """
    s = config.screen_noise_bp
    rho1 = _screen_corr(config, t_left)
    if right_err is None:
        return rho1 * left_err + s * math.sqrt(1.0 - rho1 * rho1) * z
    rho2 = _screen_corr(config, t_right)
    denom = 1.0 - rho1 * rho1 * rho2 * rho2
    mean = (left_err * rho1 * (1.0 - rho2 * rho2) + right_err * rho2 * (1.0 - rho1 * rho1)) / denom
    var = s * s * (1.0 - rho1 * rho1) * (1.0 - rho2 * rho2) / denom
    return mean + math.sqrt(var) * z


def _screen_observations(
    config: GameConfig, rng: RngStreams, event_steps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Observation steps and the persistent screen error at each.

    Regular refreshes at 0, r, 2r, ... are drawn exactly as before, from the
    instrument's ``SCREEN`` substream, one normal per refresh. Then every
    print that does not land on a refresh gets a forced observation at its
    own step, so the screen shows the print the moment it happens: without
    it, a print between refreshes leaves the pre-print level on the screen for
    up to ``screen_refresh_s``, and a hedge venue priced off the screen would
    fill at that stale level -- free money from the generator.

    The forced observation is bridged between its neighbours rather than
    folded into a sequential recursion, for two reasons. A sequential AR(1)
    over the merged steps would make the number of ``SCREEN`` draws depend on
    the random calendar, and inserting a print would shift the error at every
    regular refresh after it. Bridging leaves every regular refresh
    bit-identical whatever the calendar, keeps the ``SCREEN`` draw count a
    function of config alone, and takes each forced observation's one normal
    from ``SCREEN_PRINT`` keyed on event ordinal -- the same per-event pattern
    as ``JUMPS`` and ``DRIFT``. The joint law is identical to the sequential
    one: the OU sampled at the merged times.
    """
    r = config.screen_refresh_steps
    grid = np.arange(config.n_steps // r + 1) * r
    z = rng.substream(Stream.SCREEN, INSTRUMENT_ORDINAL).standard_normal(len(grid))
    grid_err = _screen_error(config, z)

    forced_steps: list[int] = []
    forced_err: list[float] = []
    for i, e in enumerate(int(x) for x in event_steps):
        if e % r == 0:
            continue  # the print lands on a regular refresh, already observed
        g = e // r
        left_step, left_err = int(grid[g]), float(grid_err[g])
        if forced_steps and forced_steps[-1] > left_step:
            # An earlier print in the same refresh interval is the nearer
            # left neighbour; conditioning on it keeps the chain Markov.
            left_step, left_err = forced_steps[-1], forced_err[-1]
        if g + 1 < len(grid):
            right_step, right_err = int(grid[g + 1]), float(grid_err[g + 1])
        else:
            right_step, right_err = e, None  # print after the last refresh
        zi = float(rng.substream(Stream.SCREEN_PRINT, i).standard_normal())
        forced_steps.append(e)
        forced_err.append(
            _bridge_error(
                config,
                left_err,
                (e - left_step) * config.dt,
                right_err,
                (right_step - e) * config.dt,
                zi,
            )
        )

    steps = np.concatenate((grid, np.array(forced_steps, dtype=np.int64)))
    errs = np.concatenate((grid_err, np.array(forced_err)))
    order = np.argsort(steps, kind="stable")
    return steps[order].astype(np.int64), errs[order]


def _cumulative(increments: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(increments)))


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def generate(config: GameConfig, rng: RngStreams) -> FairValuePath:
    """Build a session's fair value path. Called once, by the engine.

    Every draw is a function of ``config`` alone -- never of anything
    downstream -- and every draw comes from a substream, so nothing drawn
    elsewhere can move this and calling it twice gives the same path.
    """
    n = config.n_steps

    # Diffusion: one bulk vectorised draw from the instrument's own generator.
    z = rng.substream(Stream.FAIR_VALUE, INSTRUMENT_ORDINAL).standard_normal(n)
    diffusion = _cumulative(sigma_per_step(config) * z)

    # Calendar (public timing), then per-event outcome and drift (hidden).
    event_steps = _draw_calendar(config, rng.substream(Stream.EVENTS, MARKET_ORDINAL))
    signs = np.empty(len(event_steps))
    magnitudes = np.empty(len(event_steps))
    windows = []
    for i, e in enumerate(event_steps):
        signs[i], magnitudes[i] = _draw_jump(config, rng.substream(Stream.JUMPS, i))
        windows.append(
            _draw_drift(config, rng.substream(Stream.DRIFT, i), i, int(e), signs[i])
        )
    jump_sizes = signs * magnitudes
    drift = _cumulative(_drift_increments(n, windows))
    jumps = _cumulative(_jump_increments(n, event_steps, jump_sizes))
    # Unannounced moves: drawn after, and independently of, everything above.
    move_list = _draw_moves(config, rng, windows, event_steps)
    moves = _cumulative(_move_increments(n, move_list))

    relative = diffusion + drift + jumps + moves
    values = config.start_level + relative

    # Screen: the truth at each observation step plus a persistent
    # observation error, then rounded to the tick. Observations are every
    # regular refresh plus a forced one at each print, so there is no stale
    # window after a print. Rounding happens after the error and in
    # level-relative space, so every reading is a whole number of ticks from
    # start_level whatever the level is, and the invariance to start_level
    # stays exact. The persistent error underneath is what makes the rounded
    # screen sit on one tick for a while and then flip, rather than jitter.
    obs_steps, screen_err = _screen_observations(config, rng, event_steps)
    observed = relative[obs_steps] + screen_err
    screen = config.start_level + np.round(observed / config.tick_size) * config.tick_size

    event_steps = _freeze(event_steps)
    public = PublicView(
        n_steps=n,
        dt=config.dt,
        refresh_steps=config.screen_refresh_steps,
        event_steps=event_steps,
        screen_steps=_freeze(obs_steps),
        screen=_freeze(screen),
    )
    return FairValuePath(
        start_level=config.start_level,
        dt=config.dt,
        values=_freeze(values),
        relative=_freeze(relative),
        diffusion=_freeze(diffusion),
        drift=_freeze(drift),
        moves=_freeze(moves),
        event_steps=event_steps,
        jump_sizes=_freeze(jump_sizes),
        drift_windows=tuple(windows),
        move_list=move_list,
        screen_error=_freeze(screen_err),
        public=public,
    )
