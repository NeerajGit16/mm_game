"""The world: everything random about a session, generated before it starts.

A session is ``(config, seed)``. This module turns that pair into the price
path, the RFQ schedule and every RFQ's market, and it is the only place a
session's ``RngStreams`` is constructed. Everything downstream -- the engine,
the book, scoring, the bots -- receives the world and draws nothing.

The world splits the way its parts do:

- ``World.public`` is what a participant may see, each item at its own step:
  the screen, the print calendar, the roster and the RFQs as they arrive. It
  holds the whole session, future included; the engine decides when each
  item is shown.
- ``World.resolve`` is the one door from a live session into the hidden half:
  given an RFQ ordinal and a quote it returns what the player is told. The
  engine is handed this callable and the public half, never the world, so the
  code that runs during play has no reference to the truth to read.
- The rest -- the truth, informed flags, signals, competitors, limits -- is
  for the post-session record and the flow oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mm_game import fair_value, fills, flow
from mm_game.config import CONFIG_SCHEMA_VERSION, GameConfig
from mm_game.fair_value import FairValuePath, PublicView
from mm_game.fills import FillReport, MarketSide, Quote
from mm_game.flow import FlowSchedule, PublicFlow
from mm_game.rng import RngStreams

Resolver = Callable[[int, Quote | None], FillReport]


@dataclass(frozen=True, eq=False)
class PublicWorld:
    """What a participant may see, each item from its own step on. The print
    calendar is the exception: its times are public from the start."""

    config: GameConfig
    view: PublicView  # screen observations and the calendar
    flow: PublicFlow  # roster, and every RFQ as it will look on arrival


@dataclass(frozen=True, eq=False)
class World:
    """A session's pre-generated world. Frozen data; hidden in play."""

    config: GameConfig
    seed: int
    header: dict[str, Any]  # provenance: written here, at generation, never by a caller
    path: FairValuePath
    schedule: FlowSchedule
    sides: tuple[MarketSide, ...]

    @property
    def public(self) -> PublicWorld:
        return PublicWorld(config=self.config, view=self.path.public, flow=self.schedule.public)

    def resolve(self, rfq_ordinal: int, quote: Quote | None) -> FillReport:
        """The outcome of one RFQ under a quote: the only hidden lookup a live
        session makes, and its output is exactly what the player is told."""
        return fills.resolve(self.schedule.rfqs[rfq_ordinal], quote, self.sides[rfq_ordinal], self.config)

    def informed_ordinals(self) -> frozenset[int]:
        """Which RFQs were informed. Post-session, and for the flow oracle."""
        return frozenset(q.ordinal for q in self.schedule.rfqs if q.informed)


def header(config: GameConfig, rng: RngStreams) -> dict[str, Any]:
    """The provenance header every record starts with: the scheme, seed,
    numpy version and canary from the streams, plus the config and the
    version of its layout, so a loader can refuse what it cannot read."""
    return {
        **rng.provenance(),
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "config": config.to_record(),
    }


def generate(config: GameConfig, seed: int) -> World:
    """Build a session's world from ``(config, seed)``. Pure: the same pair
    always gives the same world, bit for bit."""
    rng = RngStreams(seed)
    path = fair_value.generate(config, rng)
    schedule = flow.generate(config, rng, path)
    sides = fills.pregenerate(config, rng, path, schedule)
    return World(
        config=config,
        seed=rng.seed,
        header=header(config, rng),
        path=path,
        schedule=schedule,
        sides=sides,
    )
