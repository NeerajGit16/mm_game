"""The hedge venue and the closing liquidation.

You can lay off inventory at any time, not only by waiting for a client to
come the other way. Two constraints decide whether the game survives it.

It prices off the tick-rounded screen, never the truth. A venue that filled at
true fair value plus a cost would leak the truth through its fill prices and
defeat the screen's noise. Priced off the screen, hedging when the screen is
wrong costs you -- which is correct. Every print step is a screen observation,
so there is no stale moment after a print to hedge at the old level.

It is dearer than the competitive client spread everywhere. If the per-lot
cost were below what a client fill typically earns, the dominant strategy
would be quote, fill, hedge instantly -- riskless and mechanical, flat before
every print, adverse selection gone. Config validation proves the cost
exceeds the furthest a competitor quote can sit from its own mid by a margin,
at every ticket size and every distance from a print. Hedging is then a
decision: pay a basis point to get out of a fill you think was informed,
ahead of a print that could cost you three.

Both functions take only the public view. They are structurally unable to read
the truth, and they draw no randomness: the same request at the same step
always fills at the same price. Hedges do not move the price.
"""

from __future__ import annotations

from dataclasses import dataclass

from mm_game.config import GameConfig
from mm_game.fair_value import PublicView


@dataclass(frozen=True)
class HedgeFill:
    """A venue execution. ``quantity`` is signed: + bought, - sold."""

    step: int
    requested: int
    quantity: int
    price: float | None  # average price per lot; None if nothing filled
    cost_bp_per_lot: float
    clipped: bool  # the position limit cut the request

    @property
    def cost_bp(self) -> float:
        """Total cost against the screen, in bp-lots. Times
        ``dollars_per_bp_per_lot`` for dollars."""
        return abs(self.quantity) * self.cost_bp_per_lot

    def to_record(self) -> dict[str, object]:
        return {
            "step": self.step,
            "requested": self.requested,
            "quantity": self.quantity,
            "price": self.price,
            "cost_bp_per_lot": self.cost_bp_per_lot,
            "clipped": self.clipped,
        }


def _execute(
    step: int, requested: int, quantity: int, public: PublicView, config: GameConfig, multiplier: float
) -> HedgeFill:
    if quantity == 0:
        return HedgeFill(step, requested, 0, None, 0.0, requested != 0)
    seconds = public.seconds_to_nearest_print(step)
    cost_bp = config.hedge_cost_bp_at(abs(quantity), seconds) * multiplier
    direction = 1 if quantity > 0 else -1
    price = public.screen_at(step) + direction * cost_bp
    return HedgeFill(step, requested, quantity, price, cost_bp, quantity != requested)


def hedge(step: int, quantity: int, position: int, public: PublicView, config: GameConfig) -> HedgeFill:
    """Buy (``quantity`` > 0) or sell at the screen showing at ``step`` plus
    the per-lot cost for that size and distance from a print.

    Hedges count toward the position limit and can never take you through
    it: the request is clipped to the room left, and the fill says so.
    """
    limit = config.position_limit_lots
    allowed = max(-limit - position, min(limit - position, quantity))
    return _execute(step, quantity, allowed, public, config, multiplier=1.0)


def liquidate(position: int, step: int, public: PublicView, config: GameConfig) -> HedgeFill:
    """Flatten the residual position at the scheduled close: the hedge-cost
    function at the final screen, times ``close_liquidation_multiplier`` --
    the hedge you didn't do, at a worse time. Without the multiplier, holding
    to the bell would cost exactly what hedging earlier would have, and a
    P&L-only score would give no reason to get flat. A quit session is
    liquidated the same way, at the same scheduled close."""
    return _execute(step, -position, -position, public, config, config.close_liquidation_multiplier)
