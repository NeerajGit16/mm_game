"""Fills: where your quote meets the world.

For each RFQ this decides whether you trade, at what price, and what you are
told afterwards. Everything before this module is pre-generated and
indifferent to you; this is the first that runs against your decisions, which
makes its governing rule the one that matters most:

    Pre-generate everything random, per RFQ. The only live computation is a
    deterministic comparison against your quote.

``pregenerate`` builds a ``MarketSide`` for every RFQ before the session starts
-- the three competing dealers' quotes, the client's limit, the tie-break --
and takes no quote, no position and nothing else the player does. ``resolve``
is a pure function of an RFQ, a quote and its ``MarketSide``. So quote
differently at 10:15 and RFQ #212's outcome may change, but RFQ #213's
competitors, limit and tie-break are bit-identical: every later difference is
caused by your position, never by a reshuffled market.

The competitors
    Three dealers share one model. A dealer's mid is the truth, plus a share
    of the screen's error -- dealers watch the same screen, so when it reads
    rich the street leans rich -- plus the dealer's own persistent error.
    Both error components persist for minutes, so cover calibrates you but
    cannot be averaged down to the truth. Each quotes its mid plus or minus a
    half-spread, rounded to the nearest tick, widening for size and around
    prints. That is public information only: a dealer never knows who is
    informed, and never adapts during a session, because whether a dealer was
    picked off depends on whether you won the trade.

The client
    Every RFQ carries a limit: the worst price the client accepts on their
    side. Uninformed limits sit on the screen plus a tolerance of about
    three-quarters of a bp, occasionally three times that. Informed limits
    reflect the client's view of the
    print: they pay up to the screen plus about half their expected move. The
    ranges overlap, so a fill at a wide price is evidence, never proof. The
    client trades with the best price inside the limit, or walks.

What you are told
    WON: the cover, the best competing price. DONE_AWAY: the side the client
    traded and the level it traded at. NOT_DONE: only that the client walked.
    PASSED: nothing at all -- passing has an information cost as well as a
    spread cost, which is why dealers keep quoting when they don't want the
    trade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from mm_game.config import GameConfig
from mm_game.fair_value import FairValuePath
from mm_game.flow import RFQ, FlowSchedule, Side
from mm_game.rng import RngStreams, Stream


class Outcome(StrEnum):
    WON = "won"  # you traded
    DONE_AWAY = "done_away"  # the client traded with a competitor
    NOT_DONE = "not_done"  # the client walked: no price inside their limit
    PASSED = "passed"  # you declined to quote


# --------------------------------------------------------------------------
# Prices on the tick grid
# --------------------------------------------------------------------------


def to_ticks(price: float, config: GameConfig) -> int:
    """A price as a whole number of ticks from the start level.

    Quotes are compared in integer ticks so a tie is a tie: two prices built
    by different arithmetic can differ in the last bit and silently turn a
    tie into a win. A price off the grid is refused rather than rounded.
    """
    k = (price - config.start_level) / config.tick_size
    nearest = round(k)
    if not math.isclose(k, nearest, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"price {price} is not on the tick grid")
    return int(nearest)


def from_ticks(k: int, config: GameConfig) -> float:
    return config.start_level + k * config.tick_size


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """Your price on an RFQ. ``None`` on a side means no price on that side;
    both ``None`` is a pass. Size is the RFQ's size; no partial fills."""

    rfq_ordinal: int
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        for name in ("bid", "ask"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.bid is not None and self.ask is not None and not self.bid < self.ask:
            raise ValueError(f"bid {self.bid} must be below ask {self.ask}")

    @property
    def is_pass(self) -> bool:
        return self.bid is None and self.ask is None


@dataclass(frozen=True)
class MarketSide:
    """Everything random about one RFQ's market, drawn before the session.
    Hidden during play; written to the coaching record afterwards."""

    rfq_ordinal: int
    competitor_bid_ticks: tuple[int, ...]
    competitor_ask_ticks: tuple[int, ...]
    competitor_mids: tuple[float, ...]  # diagnostics only
    client_limit: float  # worst price the client accepts on their side
    tie_break_u: float

    @property
    def best_bid_ticks(self) -> int:
        return max(self.competitor_bid_ticks)

    @property
    def best_ask_ticks(self) -> int:
        return min(self.competitor_ask_ticks)


@dataclass(frozen=True)
class FillReport:
    """What you are told about one RFQ."""

    rfq_ordinal: int
    outcome: Outcome
    side: Side | None  # the client's side, where revealed
    price: float | None  # your fill price, if WON
    cover: float | None  # WON: best competing price; DONE_AWAY: the traded level
    quantity: int  # signed change in your position: 0 unless WON

    def to_record(self) -> dict[str, object]:
        return {
            "rfq_ordinal": self.rfq_ordinal,
            "outcome": self.outcome.value,
            "side": None if self.side is None else int(self.side),
            "price": self.price,
            "cover": self.cover,
            "quantity": self.quantity,
        }


# --------------------------------------------------------------------------
# Pre-generation
# --------------------------------------------------------------------------


def _persistent_error(sd: float, corr_s: float, refresh_s: float, z: np.ndarray) -> np.ndarray:
    """A stationary AR(1) on the refresh grid: stdev ``sd`` at every point,
    correlation exp(-t / corr_s) across t seconds, started from the stationary
    distribution. A correlation time of 0 gives independent draws."""
    phi = math.exp(-refresh_s / corr_s) if corr_s > 0 else 0.0
    innovation = sd * math.sqrt(1.0 - phi * phi)
    err = np.empty(len(z))
    err[0] = sd * z[0]
    for k in range(1, len(z)):
        err[k] = phi * err[k - 1] + innovation * z[k]
    return err


def _client_limit(
    config: GameConfig, rfq: RFQ, screen_mid: float, z_tolerance: float
) -> float:
    """The worst price the client accepts, on their side.

    Uninformed: the screen plus a lognormal tolerance around
    ``client_tolerance_bp``, growing mildly with size. Informed: at least
    that, and up to the screen plus ``informed_edge_given_up`` of the move
    they expect from their view -- (2q - 1) * |signal| for a view right with
    probability q. The tolerance draw is shared by both, so the two ranges
    overlap and turning a client informed only ever widens their limit.
    """
    distance = (
        config.client_tolerance_bp
        * math.exp(config.client_tolerance_log_sd * z_tolerance)
        * (1.0 + config.client_tolerance_size_growth * rfq.size / 100.0)
    )
    if rfq.informed and rfq.signal_bp is not None:
        expected_move = (2.0 * config.informed_accuracy - 1.0) * abs(rfq.signal_bp)
        distance = max(distance, config.informed_edge_given_up * expected_move)
    return screen_mid + int(rfq.side) * distance


def pregenerate(
    config: GameConfig, rng: RngStreams, path: FairValuePath, schedule: FlowSchedule
) -> tuple[MarketSide, ...]:
    """Every RFQ's market, drawn before the session starts.

    Takes no quote and no position: nothing here may depend on anything the
    player does. Dealer errors come from ``COMPETITION`` keyed on dealer
    ordinal, one normal per screen refresh; each RFQ's limit and tie-break
    from ``ACCEPTANCE`` keyed on RFQ ordinal, tie-break first. All draws are
    substreams, so this is pure.
    """
    if path.n_steps != config.n_steps or path.dt != config.dt:
        raise ValueError("path grid does not match config")

    tick = config.tick_size
    r = config.screen_refresh_steps
    n_grid = config.n_steps // r + 1
    dealer_err = np.array(
        [
            _persistent_error(
                config.competitor_error_bp,
                config.competitor_error_corr_s,
                config.screen_refresh_s,
                rng.substream(Stream.COMPETITION, i).standard_normal(n_grid),
            )
            for i in range(config.n_competitors)
        ]
    )
    public = path.public

    sides = []
    for rfq in schedule.rfqs:
        t = rfq.step
        showing = int(np.searchsorted(public.screen_steps, t, side="right")) - 1
        screen_mid = float(public.screen[showing])
        shared = config.competitor_screen_weight * float(path.screen_error[showing])
        seconds = public.seconds_to_nearest_print(t)
        half_spread = config.competitor_half_spread_bp_at(rfq.size, seconds)

        mids = path.values[t] + shared + dealer_err[:, t // r]
        bids = np.rint((mids - half_spread - config.start_level) / tick).astype(int)
        asks = np.rint((mids + half_spread - config.start_level) / tick).astype(int)
        asks = np.maximum(asks, bids + 1)  # a dealer never shows a locked market

        acceptance = rng.substream(Stream.ACCEPTANCE, rfq.ordinal)
        tie_break_u = float(acceptance.random())
        z_tolerance = float(acceptance.standard_normal())

        sides.append(
            MarketSide(
                rfq_ordinal=rfq.ordinal,
                competitor_bid_ticks=tuple(int(b) for b in bids),
                competitor_ask_ticks=tuple(int(a) for a in asks),
                competitor_mids=tuple(float(m) for m in mids),
                client_limit=_client_limit(config, rfq, screen_mid, z_tolerance),
                tie_break_u=tie_break_u,
            )
        )
    return tuple(sides)


# --------------------------------------------------------------------------
# Position limit
# --------------------------------------------------------------------------


def blocked_sides(size: int, position: int, config: GameConfig) -> tuple[str, ...]:
    """Quote sides whose fill would take the position through the limit.

    A filled bid buys ``size``; a filled ask sells it. The blocked side is
    always the risk-increasing one, so a dealer at the limit can still show
    a one-way price that reduces risk.
    """
    blocked = []
    if position + size > config.position_limit_lots:
        blocked.append("bid")
    if position - size < -config.position_limit_lots:
        blocked.append("ask")
    return tuple(blocked)


def apply_position_limit(
    quote: Quote, size: int, position: int, config: GameConfig
) -> tuple[Quote, tuple[str, ...]]:
    """The quote with any limit-breaching side removed, and the sides that
    were blocked. The engine records every block for the session review."""
    blocked = blocked_sides(size, position, config)
    limited = replace(
        quote,
        bid=None if "bid" in blocked else quote.bid,
        ask=None if "ask" in blocked else quote.ask,
    )
    return limited, blocked


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def resolve(rfq: RFQ, quote: Quote | None, market: MarketSide, config: GameConfig) -> FillReport:
    """Pure: the outcome of one RFQ given your quote and its pre-drawn market.

    The client compares your price on their side with the best competitor's.
    The better one trades if it is inside the client's limit; otherwise the
    client walks. A tie with k dealers at the best price goes to you with
    probability 1 / (k + 1), decided by the pre-drawn uniform -- never by a
    rule that always favours one side.
    """
    if market.rfq_ordinal != rfq.ordinal:
        raise ValueError(f"market side {market.rfq_ordinal} is not for RFQ {rfq.ordinal}")
    if quote is not None and quote.rfq_ordinal != rfq.ordinal:
        raise ValueError(f"quote {quote.rfq_ordinal} is not for RFQ {rfq.ordinal}")
    if quote is None or quote.is_pass:
        return FillReport(rfq.ordinal, Outcome.PASSED, None, None, None, 0)

    # Work in "ticks as the client sees them": lower is better for the client
    # on either side, so one comparison serves both.
    sign = int(rfq.side)  # BUY +1: client pays your ask; SELL -1: client hits your bid
    if rfq.side is Side.BUY:
        yours = None if quote.ask is None else to_ticks(quote.ask, config)
        theirs = market.best_ask_ticks
        at_best = market.competitor_ask_ticks.count(theirs)
    else:
        yours = None if quote.bid is None else to_ticks(quote.bid, config)
        theirs = market.best_bid_ticks
        at_best = market.competitor_bid_ticks.count(theirs)

    def acceptable(k: int) -> bool:
        return sign * (from_ticks(k, config) - market.client_limit) <= 1e-12

    you_best = yours is not None and sign * yours < sign * theirs
    tied = yours is not None and yours == theirs
    best = yours if you_best else theirs
    if not acceptable(best):
        return FillReport(rfq.ordinal, Outcome.NOT_DONE, None, None, None, 0)

    won = you_best or (tied and market.tie_break_u < 1.0 / (at_best + 1))
    if won:
        return FillReport(
            rfq_ordinal=rfq.ordinal,
            outcome=Outcome.WON,
            side=rfq.side,
            price=from_ticks(yours, config),
            cover=from_ticks(theirs, config),
            quantity=-sign * rfq.size,  # client buys: you sell
        )
    return FillReport(
        rfq_ordinal=rfq.ordinal,
        outcome=Outcome.DONE_AWAY,
        side=rfq.side,
        price=None,
        cover=from_ticks(theirs, config),
        quantity=0,
    )
