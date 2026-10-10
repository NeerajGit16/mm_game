"""The book: position, cash, P&L and the blotter.

Pure bookkeeping. It draws nothing and never sees the truth: the only price it
marks to is one it is handed, and the engine hands it the screen. A live P&L
marked to true fair value would let the player read the truth off their own
P&L line, which defeats the screen.

Prices are kept relative to the session's start level. Money is accumulated
in bp-lots and converted to dollars once, at the edge, through
``dollars_per_bp_per_lot``; relative prices keep the sums small, so a P&L
built from hundreds of trades at ~9550 does not lose cents to rounding.

Realised P&L uses average cost: a trade that reduces the position realises
the difference between its price and the average entry of what it closes.
Realised plus unrealised always equals cash plus the position at the mark, so
the split is presentation and the total is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mm_game.config import GameConfig


class Source(StrEnum):
    CLIENT = "client"  # a client fill on an RFQ
    HEDGE = "hedge"  # the hedge venue
    CLOSE = "close"  # the closing liquidation


@dataclass(frozen=True)
class Trade:
    """One blotter line. ``quantity`` is signed: + bought, - sold."""

    seq: int
    step: int
    source: Source
    quantity: int
    price: float  # absolute
    rfq_ordinal: int | None  # client fills only
    cost_bp_per_lot: float  # venue and close: the charge against the screen
    position_after: int

    def to_record(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "step": self.step,
            "source": self.source.value,
            "quantity": self.quantity,
            "price": self.price,
            "rfq_ordinal": self.rfq_ordinal,
            "cost_bp_per_lot": self.cost_bp_per_lot,
            "position_after": self.position_after,
        }


@dataclass(frozen=True)
class Block:
    """A quote side removed because its fill would breach the limit."""

    step: int
    rfq_ordinal: int
    sides: tuple[str, ...]
    position: int

    def to_record(self) -> dict[str, object]:
        return {"step": self.step, "rfq_ordinal": self.rfq_ordinal, "sides": list(self.sides), "position": self.position}


class Book:
    """Position, cash and P&L for one session. Starts flat."""

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._level = config.start_level
        self._position = 0
        self._cash = 0.0  # bp-lots, prices relative to the start level
        self._avg = 0.0  # average entry of the open position, relative
        self._realised = 0.0  # bp-lots
        self._trades: list[Trade] = []
        self._blocks: list[Block] = []

    # -- state ------------------------------------------------------------------

    @property
    def position(self) -> int:
        return self._position

    @property
    def trades(self) -> tuple[Trade, ...]:
        return tuple(self._trades)

    @property
    def blocks(self) -> tuple[Block, ...]:
        return tuple(self._blocks)

    def room(self) -> tuple[int, int]:
        """Lots that can still be bought and sold before the limit."""
        limit = self._config.position_limit_lots
        return limit - self._position, limit + self._position

    # -- money, in bp-lots and in dollars ---------------------------------------

    def realised_bp(self) -> float:
        return self._realised

    def unrealised_bp(self, mark: float) -> float:
        return self._position * ((mark - self._level) - self._avg)

    def total_bp(self, mark: float) -> float:
        """Cash plus the position at ``mark``. Equal to realised plus
        unrealised up to rounding; computed from cash so the final score does
        not inherit the average-cost arithmetic."""
        return self._cash + self._position * (mark - self._level)

    def cash_bp(self) -> float:
        """Cash in bp-lots. Once flat, this is the realised P&L."""
        return self._cash

    def usd(self, bp_lots: float) -> float:
        return bp_lots * self._config.dollars_per_bp_per_lot

    # -- recording --------------------------------------------------------------

    def trade(
        self,
        *,
        step: int,
        source: Source,
        quantity: int,
        price: float,
        rfq_ordinal: int | None = None,
        cost_bp_per_lot: float = 0.0,
    ) -> Trade:
        if quantity == 0:
            raise ValueError("a trade must have a non-zero quantity")
        limit = self._config.position_limit_lots
        if abs(self._position + quantity) > limit and source is not Source.CLOSE:
            # Fills and hedges are limited before they reach the book; a
            # breach here is an engine bug, not a market event.
            raise ValueError(f"trade of {quantity} from {self._position} breaches the {limit}-lot limit")
        p = price - self._level
        self._cash -= quantity * p
        q, pos = quantity, self._position
        if pos != 0 and (pos > 0) != (q > 0):
            closing = -pos if abs(q) >= abs(pos) else q  # signed like q
            self._realised += -closing * (p - self._avg)
            pos += closing
            q -= closing
            if pos == 0:
                self._avg = 0.0
        if q != 0:
            self._avg = (self._avg * pos + p * q) / (pos + q)
            pos += q
        self._position = pos
        entry = Trade(
            seq=len(self._trades),
            step=step,
            source=source,
            quantity=quantity,
            price=price,
            rfq_ordinal=rfq_ordinal,
            cost_bp_per_lot=cost_bp_per_lot,
            position_after=pos,
        )
        self._trades.append(entry)
        return entry

    def record_block(self, step: int, rfq_ordinal: int, sides: tuple[str, ...]) -> Block:
        block = Block(step=step, rfq_ordinal=rfq_ordinal, sides=sides, position=self._position)
        self._blocks.append(block)
        return block
