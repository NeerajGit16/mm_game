"""Flow: who asks, when, for how much, and what they know.

This is where adverse selection is manufactured. Fair value built the thing
there is to know -- a print with a hidden size at a public time. This module
decides who knows it, how well, and what their inquiries look like from the
outside, which is the only thing a market maker ever sees.

The schedule is a pure function of ``(config, seed, path)`` and is fixed
before the session starts. It never responds to the player: quoting tight
brings no extra inquiries and getting picked off does not bring a client back.
That is what makes "same session, quote differently at 10:15" a clean
comparison. Each RFQ carries the client's latent intent -- a side and a size
they came with -- and whether they trade is decided later, in ``fills``.

Model, in one pass over the arrivals:

- **When.** A per-step Bernoulli at ``rate * dt``: homogeneous Poisson on the
  grid, one bulk draw, arrivals on distinct steps by construction.
- **Who.** A named roster with a visible category per client and a hidden
  type drawn fresh every session. Each RFQ is attributed to a client by one
  uniform keyed on RFQ ordinal.
- **Informed?** An informed-type client arriving inside a drift window is
  informed about that window's print -- the nearer upcoming one if windows
  overlap. Informed flow *replaces* volume, it never adds it, so raising the
  informed share leaves every RFQ's ordinal, step and client where it was.
- **Signal.** One view per client per print: direction right with
  probability q, magnitude the print's times a lognormal error. Every informed
  inquiry a client sends about a print carries that same view, so an informed
  client never buys and sells the same print. Drawn from ``FLOW_INFORMED``
  keyed on the (client, print) pair, never from a fair-value stream.
- **Side.** Informed: the signal's sign. Uninformed: a fair coin outside drift
  windows, tilted toward the drift inside them.
- **Size.** One lognormal base draw per RFQ whether or not it is informed;
  informed RFQs scale it by signal strength. Bigger on average, overlapping.
- **One-way.** A minority of RFQs from both populations show their side on
  arrival. Drawn after the side, so tuning the rate never moves a side.
- **Competition.** Each RFQ goes to each of the dealers independently, one
  uniform per dealer. How many are in competition is public on arrival, as a
  platform shows it; which ones is hidden. Informed clients can be made to
  ask fewer dealers, a lever built and off by default.

Every draw comes from a substream -- arrivals on market ordinal, client types
on client id, signals on the (client, print) pair, everything else on RFQ
ordinal -- so ``generate`` is pure and these invariances hold by
construction:

- the size model never moves step, client, side, one-way or informed status;
- one-way rates never move a side;
- informed probabilities never move client assignment, base size, or any
  uninformed RFQ (while the activity multiplier is at its inert 1.0);
- only the arrival rate and the grid move arrivals;
- competition is drawn last, from its own stream, so it moves nothing else,
  and raising the participation probability only ever adds dealers;
- nothing here touches a fair-value stream: the path is an input.

The public/private split mirrors ``FairValuePath``: ``FlowSchedule.public``
carries what a participant may see, each RFQ at its own step. Informed status,
signal, the print it concerns, and the side of a two-way RFQ are hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

from mm_game.config import GameConfig
from mm_game.fair_value import MARKET_ORDINAL, DriftWindow, FairValuePath
from mm_game.rng import RngStreams, Stream


class Side(IntEnum):
    """The client's side. BUY means the client buys from you: you go short."""

    BUY = 1
    SELL = -1


class Category(StrEnum):
    """A client's visible category. Correlates with informed type, imperfectly."""

    HEDGE_FUND = "hedge_fund"
    REAL_MONEY = "real_money"
    CORPORATE = "corporate"


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicClient:
    client_id: int
    category: Category


@dataclass(frozen=True)
class Client:
    client_id: int
    category: Category
    informed_type: bool  # hidden; redrawn every session

    @property
    def public(self) -> PublicClient:
        return PublicClient(client_id=self.client_id, category=self.category)


@dataclass(frozen=True)
class PublicRFQ:
    """An inquiry as the player sees it on arrival. ``side`` is None for a
    two-way RFQ: the client says the size, not the direction. ``in_comp`` is
    how many other dealers were asked -- not which."""

    ordinal: int
    step: int
    client_id: int
    size: int
    one_way: bool
    side: Side | None
    in_comp: int


@dataclass(frozen=True)
class RFQ:
    """An inquiry with its hidden half. Latent intent, not an execution."""

    ordinal: int
    step: int
    client_id: int
    size: int  # lots, the client's intent
    one_way: bool
    side: Side  # public only if one_way
    informed: bool  # hidden
    signal_bp: float | None  # hidden: the noisy print estimate, signed
    event_index: int | None  # hidden: the print they are informed about
    base_size: float  # hidden: the lognormal draw before scaling and rounding
    dealers_in_comp: tuple[bool, ...]  # hidden: which dealers were asked

    @property
    def in_comp(self) -> int:
        """How many dealers are in competition. Public."""
        return sum(self.dealers_in_comp)

    @property
    def public(self) -> PublicRFQ:
        return PublicRFQ(
            ordinal=self.ordinal,
            step=self.step,
            client_id=self.client_id,
            size=self.size,
            one_way=self.one_way,
            side=self.side if self.one_way else None,
            in_comp=self.in_comp,
        )


@dataclass(frozen=True)
class PublicFlow:
    """What a participant may see. The roster is public at the start; each RFQ
    is public from its own step, and the engine reveals it then."""

    clients: tuple[PublicClient, ...]
    rfqs: tuple[PublicRFQ, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "clients": [{"client_id": c.client_id, "category": c.category.value} for c in self.clients],
            "rfqs": [
                {
                    "ordinal": q.ordinal,
                    "step": q.step,
                    "client_id": q.client_id,
                    "size": q.size,
                    "one_way": q.one_way,
                    "side": None if q.side is None else int(q.side),
                    "in_comp": q.in_comp,
                }
                for q in self.rfqs
            ],
        }


@dataclass(frozen=True)
class FlowSchedule:
    """A session's RFQ schedule, generated in full before the session starts."""

    clients: tuple[Client, ...]
    rfqs: tuple[RFQ, ...]  # sorted by step; ordinal == index

    @property
    def public(self) -> PublicFlow:
        return PublicFlow(
            clients=tuple(c.public for c in self.clients),
            rfqs=tuple(q.public for q in self.rfqs),
        )

    def to_record(self) -> dict[str, object]:
        """Plain-type schedule including hidden fields. Post-session log only."""
        return {
            "clients": [
                {"client_id": c.client_id, "category": c.category.value, "informed_type": c.informed_type}
                for c in self.clients
            ],
            "rfqs": [
                {
                    "ordinal": q.ordinal,
                    "step": q.step,
                    "client_id": q.client_id,
                    "size": q.size,
                    "one_way": q.one_way,
                    "side": int(q.side),
                    "informed": q.informed,
                    "signal_bp": q.signal_bp,
                    "event_index": q.event_index,
                    "base_size": q.base_size,
                    "dealers_in_comp": list(q.dealers_in_comp),
                }
                for q in self.rfqs
            ],
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _roster_categories(config: GameConfig) -> list[Category]:
    """Categories by client id: hedge funds, then real money, then corporates.
    Fixed rather than shuffled: the category is public, so the ordering hides
    nothing, and a shuffle added later would change which client is informed."""
    return (
        [Category.HEDGE_FUND] * config.n_hedge_funds
        + [Category.REAL_MONEY] * config.n_real_money
        + [Category.CORPORATE] * config.n_corporates
    )


def _informed_prob(config: GameConfig, category: Category) -> float:
    return {
        Category.HEDGE_FUND: config.informed_prob_hedge_fund,
        Category.REAL_MONEY: config.informed_prob_real_money,
        Category.CORPORATE: config.informed_prob_corporate,
    }[category]


def _draw_roster(config: GameConfig, rng: RngStreams) -> tuple[Client, ...]:
    """One uniform per client, compared with its category's probability.

    Keyed on client id, so raising a probability only ever adds informed
    clients -- a client informed at p stays informed at any p' > p -- and the
    count of informed clients is random, not fixed: a known count would let
    the player stop suspecting everyone else once they had caught that many.
    """
    clients = []
    for client_id, category in enumerate(_roster_categories(config)):
        u = rng.substream(Stream.FLOW_CLIENT_TYPE, client_id).random()
        clients.append(
            Client(
                client_id=client_id,
                category=category,
                informed_type=bool(u < _informed_prob(config, category)),
            )
        )
    return tuple(clients)


def _draw_arrivals(config: GameConfig, rng: RngStreams) -> np.ndarray:
    """RFQ steps: a Bernoulli trial on each of steps 0..n-1, one bulk draw.

    Stamped at the start of each grid interval, so no RFQ lands on the closing
    step -- one there could only be filled and immediately liquidated. The
    expected count is exactly rate * session_length.
    """
    u = rng.substream(Stream.FLOW_ARRIVAL, MARKET_ORDINAL).random(config.n_steps)
    return np.flatnonzero(u < config.rfq_arrival_prob).astype(np.int64)


def _informing_window(windows: tuple[DriftWindow, ...], step: int) -> DriftWindow | None:
    """The drift window an informed client at ``step`` is informed about.

    A window covers ``[start_step, end_step)``: from the first drift increment
    up to, not including, the print step, where the print is already public.
    If windows overlap, the one whose print comes first.
    """
    covering = [w for w in windows if w.start_step <= step < w.end_step]
    return min(covering, key=lambda w: w.end_step) if covering else None


def _assign_client(clients: tuple[Client, ...], in_window: bool, multiplier: float, u: float) -> Client:
    """Inverse-CDF assignment over activity weights.

    Every client has weight 1, and informed-type clients have weight
    ``multiplier`` inside drift windows. At the default 1.0 the weights ignore
    types entirely, so client assignment is independent of the informed
    probabilities. Built now so enabling the multiplier is a parameter change.
    """
    weights = np.array(
        [multiplier if (in_window and c.informed_type) else 1.0 for c in clients]
    )
    cdf = np.cumsum(weights)
    index = int(np.searchsorted(cdf, u * cdf[-1], side="right"))
    return clients[min(index, len(clients) - 1)]


def _pair_key(client_id: int, event_index: int) -> int:
    """A stable non-negative integer for a (client, print) pair: the Cantor
    pairing. Independent of roster size and print count, so adding a client
    or a print never moves another pair's signal."""
    s = client_id + event_index
    return s * (s + 1) // 2 + event_index


def _draw_signal(config: GameConfig, gen: np.random.Generator, jump_bp: float) -> float:
    """An informed client's estimate of one print, signed. One per (client,
    print): drawn per RFQ instead, the same client re-drew its view on every
    inquiry and half of informed clients with several inquiries both bought
    and sold the same print -- incoherent, and fatal to the persistence read.

    Direction is right with probability ``informed_accuracy``; magnitude is the
    print's times ``exp(noise * z)``, so they know roughly how much. Two draws
    in that order, so tuning the magnitude noise never flips a direction.

    Chosen over additive noise on the signed print. Additive noise calibrated
    to 80% direction accuracy on 2-4bp prints needs a ~3.6bp stdev, which
    would leave the client knowing almost nothing about magnitude, when an
    informed client is meant to know roughly how much.
    """
    correct = gen.random() < config.informed_accuracy
    direction = math.copysign(1.0, jump_bp) * (1.0 if correct else -1.0)
    magnitude = abs(jump_bp) * math.exp(config.informed_signal_noise * gen.standard_normal())
    return direction * magnitude


def _size_factor(config: GameConfig, signal_bp: float) -> float:
    """Informed size multiplier: |signal| / ref, clipped. ~1x at 1.5bp, ~3x at
    4.5bp; ~2x on average over 2-4bp prints."""
    raw = abs(signal_bp) / config.informed_size_ref_bp
    return min(config.informed_size_factor_max, max(config.informed_size_factor_min, raw))


def _dealers_in_comp(config: GameConfig, gen: np.random.Generator, informed: bool) -> tuple[bool, ...]:
    """Which dealers an RFQ goes to: one uniform per dealer against the
    participation probability, so raising it only ever adds dealers and a
    dealer added to the street never changes whether the others are in. On
    informed RFQs the probability is scaled by the informed multiplier; at
    1.0 the comparison is exactly the uninformed one."""
    u = gen.random(config.n_competitors)
    p = config.competitor_participation_prob
    if informed:
        p = min(1.0, p * config.informed_participation_multiplier)
    return tuple(bool(x < p) for x in u)


def _lots(config: GameConfig, size: float) -> int:
    """Whole lots, at least 1, at most the cap. No single inquiry decides a session."""
    return int(max(1, min(config.size_cap_lots, round(size))))


def generate(config: GameConfig, rng: RngStreams, path: FairValuePath) -> FlowSchedule:
    """Build a session's RFQ schedule. Called once, by the engine, after
    ``fair_value.generate`` and before the session starts.

    Takes no player input and never will: the schedule must not respond to
    anything the player does.
    """
    if path.n_steps != config.n_steps or path.dt != config.dt:
        raise ValueError(
            f"path grid ({path.n_steps} steps of {path.dt}s) does not match config "
            f"({config.n_steps} steps of {config.dt}s)"
        )

    clients = _draw_roster(config, rng)
    steps = _draw_arrivals(config, rng)
    # The drift rate at each step, for the crowd's tilt: the net of any
    # overlapping windows, which is what the screen shows drifting.
    drift_rate = np.diff(path.drift)
    # Each informed client's view of each print, drawn on first use. The draw
    # is keyed on the pair, so caching changes nothing but speed.
    views: dict[tuple[int, int], float] = {}

    rfqs = []
    for ordinal, step in enumerate(int(s) for s in steps):
        window = _informing_window(path.drift_windows, step)
        client = _assign_client(
            clients,
            in_window=window is not None,
            multiplier=config.informed_activity_multiplier,
            u=rng.substream(Stream.FLOW_CLIENT, ordinal).random(),
        )
        base = config.size_median_lots * math.exp(
            config.size_log_sd * rng.substream(Stream.FLOW_SIZE, ordinal).standard_normal()
        )

        # Side, then one-way flag, always both and always in that order, so
        # the one-way uniform sits in the same place whatever the side model.
        direction = rng.substream(Stream.FLOW_DIRECTION, ordinal)
        u_side, u_one_way = direction.random(), direction.random()

        informed = window is not None and client.informed_type
        if window is not None and client.informed_type:
            pair = (client.client_id, window.event_index)
            if pair not in views:
                views[pair] = _draw_signal(
                    config,
                    rng.substream(Stream.FLOW_INFORMED, _pair_key(*pair)),
                    float(path.jump_sizes[window.event_index]),
                )
            signal = views[pair]
            side = Side.BUY if signal > 0 else Side.SELL
            size = _lots(config, base * _size_factor(config, signal))
            one_way = u_one_way < config.one_way_prob_informed
            event_index: int | None = window.event_index
        else:
            signal = None
            rate = drift_rate[step]
            if rate > 0:
                p_buy = config.crowd_drift_tilt
            elif rate < 0:
                p_buy = 1.0 - config.crowd_drift_tilt
            else:
                p_buy = 0.5
            side = Side.BUY if u_side < p_buy else Side.SELL
            size = _lots(config, base)
            one_way = u_one_way < config.one_way_prob_uninformed
            event_index = None

        rfqs.append(
            RFQ(
                ordinal=ordinal,
                step=step,
                client_id=client.client_id,
                size=size,
                one_way=bool(one_way),
                side=side,
                informed=informed,
                signal_bp=signal,
                event_index=event_index,
                base_size=base,
                dealers_in_comp=_dealers_in_comp(
                    config, rng.substream(Stream.FLOW_PARTICIPATION, ordinal), informed
                ),
            )
        )

    return FlowSchedule(clients=clients, rfqs=tuple(rfqs))
