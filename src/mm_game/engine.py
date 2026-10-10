"""The engine: walks a player through a pre-generated world.

    The engine decides only when you see things and when your actions are
    stamped. It never draws randomness and never changes the world.

Everything random was drawn before the session started (``world``). This
module sequences it: advance the clock, reveal each public item at its own
step, accept a quote or a hedge, stamp it, resolve it, book it, log it. Two
sessions on the same seed can differ only in what the player did and when.

The session is handed the public half of the world and a resolver for
quotes -- never the world -- so nothing here can read the truth. "Public" is
not "visible now": the public screen and RFQ list hold the whole session,
including the future, and the engine shows each item only when the clock
reaches its step. The print calendar's times are the one exception: the
countdown is public from the start.

Time is step-driven. The front end asks for the clock to move; the clock
stops when an RFQ arrives and stays stopped until it is answered, so every
quote is stamped at its RFQ's step however long the player took. RFQs that
arrive close together queue naturally: the next one is shown only once the
clock, released by answering the current one, reaches its step. Hedges are
stamped at the current step, whether the clock is running or stopped on an
RFQ. Speed, the manual pause and the optional pricing timer are front-end
inputs: they are logged, so a review can see them, and they change nothing
in the world.

Every session starts flat. On quit, trading stops, the position rides the
rest of the path -- prints included -- and is liquidated at the scheduled
close, and the record says where it ended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mm_game import fills, market, protocol, world as world_module
from mm_game.book import Book, Source
from mm_game.config import GameConfig, ScoringConfig
from mm_game.events import EventLog, Kind
from mm_game.fills import Outcome, Quote
from mm_game.protocol import Action, ActionType, ProtocolError
from mm_game.scoring import RiskMeter, print_risk
from mm_game.world import PublicWorld, Resolver, World


@dataclass(frozen=True)
class Settings:
    """Per-session choices that are not the scenario. Recorded, replayed."""

    preset: int | None = None  # 30, 60 or 120 minutes, if from a preset
    pricing_timer_s: float | None = None  # wall-clock limit per RFQ; None is off

    def __post_init__(self) -> None:
        if self.pricing_timer_s is not None and not self.pricing_timer_s > 0:
            raise ValueError(f"pricing_timer_s must be positive or None, got {self.pricing_timer_s}")

    def to_record(self) -> dict[str, Any]:
        return {"preset": self.preset, "pricing_timer_s": self.pricing_timer_s}


class Session:
    """One live session. Speaks only the protocol: ``start()`` once, then
    ``handle(message)`` for every input; both return the outbound messages."""

    def __init__(
        self,
        public: PublicWorld,
        resolve: Resolver,
        *,
        settings: Settings,
        scoring: ScoringConfig,
        replay_of_played_seed: bool = False,
    ) -> None:
        self._public = public
        self._config: GameConfig = public.config
        self._view = public.view
        self._flow = public.flow
        self._resolve = resolve
        self.settings = settings
        self._scoring = scoring
        self._replay_of_played_seed = replay_of_played_seed

        self.log = EventLog()
        self.book = Book(self._config)
        self.actions: list[dict[str, Any]] = []  # accepted inputs with steps: the replay file
        self._print_steps = [int(e) for e in self._view.event_steps]
        self._risk = RiskMeter(self._config, scoring, self._print_steps)
        self._category = {c.client_id: c.category.value for c in self._flow.clients}
        self._n = self._config.n_steps

        self.step = 0
        self.pending: int | None = None  # RFQ ordinal on the ticket
        self.speed = 1
        self.paused = False
        self.quit_step: int | None = None
        self.ended = False
        self._started = False
        self._next_obs = 0
        self._next_print = 0
        self._next_rfq = 0

    @property
    def config(self) -> GameConfig:
        return self._config

    @property
    def n_steps(self) -> int:
        return self._n

    # -- protocol -----------------------------------------------------------

    def start(self) -> list[dict[str, Any]]:
        if self._started:
            raise RuntimeError("session already started")
        self._started = True
        self.log.append(
            0,
            Kind.SESSION_START,
            protocol_version=protocol.PROTOCOL_VERSION,
            config=self._config.to_record(),
            settings=self.settings.to_record(),
            print_steps=list(self._print_steps),
            roster=[{"client_id": c.client_id, "category": c.category.value} for c in self._flow.clients],
            replay_of_played_seed=self._replay_of_played_seed,
        )
        self._reveal(0)
        return self._outbound(0)

    def handle(self, message: Any) -> list[dict[str, Any]]:
        """Apply one input. A refused input changes nothing and answers with
        an error message and the unchanged state."""
        if not self._started:
            raise RuntimeError("call start() first")
        mark = len(self.log)
        try:
            action = protocol.parse_action(message)
            if self.ended:
                raise ProtocolError("the session has ended")
            self._apply(action)
        except ProtocolError as error:
            return [protocol.error_message(self.step, error), protocol.state_message(self.state())]
        return self._outbound(mark)

    def _outbound(self, mark: int) -> list[dict[str, Any]]:
        out = [protocol.event_message(e) for e in list(self.log)[mark:]]
        out.append(protocol.state_message(self.state()))
        return out

    # -- dispatch -----------------------------------------------------------

    def _apply(self, action: Action) -> None:
        p = action.params
        t = action.type
        if t is ActionType.ADVANCE:
            self._advance(p["to_step"])
            return  # implied by the steps of everything else: not a replay action
        if t is ActionType.QUOTE:
            self._answer(p["rfq_ordinal"], p["bid"], p["ask"], by="player")
        elif t is ActionType.PASS:
            self._answer(p["rfq_ordinal"], None, None, by="player")
        elif t is ActionType.TIMER_EXPIRED:
            if self.settings.pricing_timer_s is None:
                raise ProtocolError("the pricing timer is off")
            self._require_pending(p["rfq_ordinal"])
            self.log.append(self.step, Kind.TIMER_EXPIRED, rfq_ordinal=p["rfq_ordinal"])
            self._answer(p["rfq_ordinal"], None, None, by="timer")
        elif t is ActionType.HEDGE:
            self._hedge(p["quantity"])
        elif t is ActionType.QUIT:
            self._record(action)
            self._quit()
            return
        elif t is ActionType.SET_SPEED:
            self.speed = p["speed"]
            self.log.append(self.step, Kind.SPEED_CHANGED, speed=self.speed)
        elif t is ActionType.PAUSE:
            if self.paused:
                raise ProtocolError("already paused")
            self.paused = True
            self.log.append(self.step, Kind.PAUSED)
        elif t is ActionType.RESUME:
            if not self.paused:
                raise ProtocolError("not paused")
            self.paused = False
            self.log.append(self.step, Kind.RESUMED)
        self._record(action)

    def _record(self, action: Action) -> None:
        self.actions.append({"seq": len(self.actions), "step": self.step, "type": action.type.value, **action.params})

    # -- the clock ------------------------------------------------------------

    def _advance(self, to_step: int) -> None:
        if self.paused:
            raise ProtocolError("the clock is paused")
        if self.pending is not None:
            raise ProtocolError(f"RFQ {self.pending} is waiting for a quote")
        if not self.step <= to_step <= self._n:
            raise ProtocolError(f"cannot advance from step {self.step} to {to_step}")
        while self.step < to_step and self.pending is None:
            nxt = min(self._next_item_step(), to_step)
            self._move_clock(nxt)
            self._reveal(nxt)
        if self.step == self._n and self.pending is None:
            self._close()

    def _next_item_step(self) -> int:
        candidates = [self._n]
        if self._next_obs < len(self._view.screen_steps):
            candidates.append(int(self._view.screen_steps[self._next_obs]))
        if self._next_print < len(self._print_steps):
            candidates.append(self._print_steps[self._next_print])
        if self._next_rfq < len(self._flow.rfqs):
            candidates.append(self._flow.rfqs[self._next_rfq].step)
        return min(candidates)

    def _move_clock(self, to_step: int) -> None:
        self._risk.advance(to_step, self.book.position)
        self.step = to_step

    def _reveal(self, step: int) -> None:
        """Show everything public at ``step``: a print, the screen reading
        taken there, then an RFQ -- priced off the post-print screen."""
        while self._next_print < len(self._print_steps) and self._print_steps[self._next_print] == step:
            self.log.append(step, Kind.PRINT_RELEASED, print_index=self._next_print)
            self._next_print += 1
        if self._next_obs < len(self._view.screen_steps) and int(self._view.screen_steps[self._next_obs]) == step:
            self.log.append(step, Kind.SCREEN_UPDATE, mid=float(self._view.screen[self._next_obs]))
            self._next_obs += 1
        if self._next_rfq < len(self._flow.rfqs) and self._flow.rfqs[self._next_rfq].step == step:
            rfq = self._flow.rfqs[self._next_rfq]
            self.log.append(step, Kind.RFQ_ARRIVED, rfq=self._rfq_public(rfq))
            self.pending = rfq.ordinal
            self._next_rfq += 1

    def _rfq_public(self, rfq) -> dict[str, Any]:
        return {
            "ordinal": rfq.ordinal,
            "client_id": rfq.client_id,
            "category": self._category[rfq.client_id],
            "size": rfq.size,
            "one_way": rfq.one_way,
            "side": None if rfq.side is None else int(rfq.side),
            "in_comp": rfq.in_comp,
        }

    # -- quoting --------------------------------------------------------------

    def _require_pending(self, ordinal: int) -> None:
        if self.pending is None:
            raise ProtocolError("no RFQ is waiting for a quote")
        if ordinal != self.pending:
            raise ProtocolError(f"RFQ {self.pending} is on the ticket, not {ordinal}")

    def _answer(self, ordinal: int, bid: float | None, ask: float | None, *, by: str) -> None:
        self._require_pending(ordinal)
        rfq = self._flow.rfqs[ordinal]
        # Validate fully before anything is logged: a refused quote changes nothing.
        try:
            requested = Quote(ordinal, bid, ask)
            for price in (bid, ask):
                if price is not None:
                    fills.to_ticks(price, self._config)
        except ValueError as error:
            raise ProtocolError(str(error)) from None
        sent, blocked = fills.apply_position_limit(requested, rfq.size, self.book.position, self._config)
        if blocked and not requested.is_pass:
            self.book.record_block(self.step, ordinal, blocked)
            self.log.append(
                self.step, Kind.LIMIT_BLOCKED, rfq_ordinal=ordinal, sides=list(blocked), position=self.book.position
            )
        self.log.append(
            self.step,
            Kind.QUOTE_SENT,
            rfq_ordinal=ordinal,
            bid=sent.bid,
            ask=sent.ask,
            requested_bid=bid,
            requested_ask=ask,
            by=by,
        )
        report = self._resolve(ordinal, None if sent.is_pass else sent)
        if report.outcome is Outcome.WON:
            self.book.trade(
                step=self.step,
                source=Source.CLIENT,
                quantity=report.quantity,
                price=report.price,
                rfq_ordinal=ordinal,
            )
        self.log.append(
            self.step,
            Kind.RFQ_RESOLVED,
            rfq_ordinal=ordinal,
            outcome=report.outcome.value,
            side=None if report.side is None else int(report.side),
            price=report.price,
            cover=report.cover,
            quantity=report.quantity,
            position_after=self.book.position,
        )
        self.pending = None

    # -- hedging, quitting, closing -------------------------------------------

    def _hedge(self, quantity: int) -> None:
        if self.quit_step is not None:
            raise ProtocolError("trading has stopped")
        fill = market.hedge(self.step, quantity, self.book.position, self._view, self._config)
        if fill.quantity:
            self.book.trade(
                step=self.step,
                source=Source.HEDGE,
                quantity=fill.quantity,
                price=fill.price,
                cost_bp_per_lot=fill.cost_bp_per_lot,
            )
        self.log.append(
            self.step,
            Kind.HEDGE_EXECUTED,
            requested=fill.requested,
            quantity=fill.quantity,
            price=fill.price,
            cost_bp_per_lot=fill.cost_bp_per_lot,
            clipped=fill.clipped,
            position_after=self.book.position,
        )

    def _quit(self) -> None:
        """Trading stops here. An RFQ on the ticket is passed. The position
        rides the rest of the path, prints included, accruing risk, and is
        liquidated at the scheduled close like any other."""
        if self.pending is not None:
            self._answer(self.pending, None, None, by="quit")
        self.quit_step = self.step
        self.log.append(self.step, Kind.SESSION_QUIT, position=self.book.position)
        self.paused = False
        self._move_clock(self._n)
        self._close()

    def _close(self) -> None:
        fill = market.liquidate(self.book.position, self._n, self._view, self._config)
        if fill.quantity:
            self.book.trade(
                step=self._n,
                source=Source.CLOSE,
                quantity=fill.quantity,
                price=fill.price,
                cost_bp_per_lot=fill.cost_bp_per_lot,
            )
        self.log.append(
            self._n,
            Kind.CLOSE_LIQUIDATION,
            quantity=fill.quantity,
            price=fill.price,
            cost_bp_per_lot=fill.cost_bp_per_lot,
        )
        pnl = self.book.usd(self.book.cash_bp())
        penalty = self._risk.penalty_usd
        self.log.append(
            self._n,
            Kind.SESSION_END,
            pnl_usd=pnl,
            risk_penalty_usd=penalty,
            risk_adjusted_usd=pnl - penalty,
            ended_early_at_step=self.quit_step,
        )
        self.ended = True

    # -- the snapshot ---------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Everything a front end shows, from public state only. The book is
        marked at the screen; nothing here can see the truth."""
        mark = self._view.screen_at(self.step)
        position = self.book.position
        to_buy, to_sell = self.book.room()
        ticket = None
        if self.pending is not None:
            rfq = self._flow.rfqs[self.pending]
            ticket = {
                **self._rfq_public(rfq),
                "blocked_sides": list(fills.blocked_sides(rfq.size, position, self._config)),
            }
        return {
            "step": self.step,
            "time_s": self.step * self._config.dt,
            "status": "ended" if self.ended else ("quit" if self.quit_step is not None else "live"),
            "clock": {"speed": self.speed, "paused": self.paused, "awaiting_quote": self.pending},
            "screen": {"mid": mark, "age_s": self._view.screen_age_s(self.step)},
            "book": {
                "position": position,
                "realised_usd": self.book.usd(self.book.realised_bp()),
                "unrealised_usd": self.book.usd(self.book.unrealised_bp(mark)),
                "total_usd": self.book.usd(self.book.total_bp(mark)),
                "limit": self._config.position_limit_lots,
                "room_to_buy": to_buy,
                "room_to_sell": to_sell,
            },
            "risk": {
                "penalty_so_far_usd": self._risk.penalty_usd,
                "next_print": print_risk(self._config, self._scoring, self._print_steps, self.step, position),
            },
            "ticket": ticket,
        }


def open_session(
    config: GameConfig,
    seed: int,
    *,
    settings: Settings,
    scoring: ScoringConfig,
    replay_of_played_seed: bool = False,
) -> tuple[Session, World]:
    """Generate the world for ``(config, seed)`` and open a session on it.
    The header is written by the generation, never supplied by a caller. The
    session gets the public half and the resolver; the world is returned for
    the post-session record."""
    w = world_module.generate(config, seed)
    session = Session(
        w.public, w.resolve, settings=settings, scoring=scoring, replay_of_played_seed=replay_of_played_seed
    )
    return session, w
