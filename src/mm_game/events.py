"""The event log: the spine of a session.

An ordered, append-only list of typed events, each stamped with the step it
happened at and carrying its public payload. Every kind declares its payload
fields up front and an event with any other field is refused, so the set of
things that can ever be said during play is written down here, in one place,
and a hidden field cannot slip into the log by being added to a dict.

The hidden layer -- true fair value, informed flags, signals, competitors,
limits -- is not in this log. It is attached after the session from the
world, keyed by the steps and ordinals recorded here (see ``records``), so
nothing that runs during play ever writes or reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Kind(StrEnum):
    SESSION_START = "session_start"
    SCREEN_UPDATE = "screen_update"
    PRINT_RELEASED = "print_released"
    RFQ_ARRIVED = "rfq_arrived"
    QUOTE_SENT = "quote_sent"
    LIMIT_BLOCKED = "limit_blocked"
    RFQ_RESOLVED = "rfq_resolved"
    HEDGE_EXECUTED = "hedge_executed"
    SPEED_CHANGED = "speed_changed"
    PAUSED = "paused"
    RESUMED = "resumed"
    TIMER_EXPIRED = "timer_expired"
    SESSION_QUIT = "session_quit"
    CLOSE_LIQUIDATION = "close_liquidation"
    SESSION_END = "session_end"


# Every field each kind may carry. Public, all of them: this is the in-session
# vocabulary.
FIELDS: dict[Kind, frozenset[str]] = {
    Kind.SESSION_START: frozenset(
        {"protocol_version", "config", "settings", "print_steps", "roster", "replay_of_played_seed"}
    ),
    Kind.SCREEN_UPDATE: frozenset({"mid"}),
    Kind.PRINT_RELEASED: frozenset({"print_index"}),
    Kind.RFQ_ARRIVED: frozenset({"rfq"}),
    Kind.QUOTE_SENT: frozenset({"rfq_ordinal", "bid", "ask", "requested_bid", "requested_ask", "by"}),
    Kind.LIMIT_BLOCKED: frozenset({"rfq_ordinal", "sides", "position"}),
    Kind.RFQ_RESOLVED: frozenset({"rfq_ordinal", "outcome", "side", "price", "cover", "quantity", "position_after"}),
    Kind.HEDGE_EXECUTED: frozenset(
        {"requested", "quantity", "price", "cost_bp_per_lot", "clipped", "position_after"}
    ),
    Kind.SPEED_CHANGED: frozenset({"speed"}),
    Kind.PAUSED: frozenset(),
    Kind.RESUMED: frozenset(),
    Kind.TIMER_EXPIRED: frozenset({"rfq_ordinal"}),
    Kind.SESSION_QUIT: frozenset({"position"}),
    Kind.CLOSE_LIQUIDATION: frozenset({"quantity", "price", "cost_bp_per_lot"}),
    Kind.SESSION_END: frozenset(
        {"pnl_usd", "risk_penalty_usd", "risk_adjusted_usd", "ended_early_at_step"}
    ),
}


@dataclass(frozen=True)
class Event:
    seq: int
    step: int
    kind: Kind
    data: dict[str, Any]

    def __post_init__(self) -> None:
        allowed = FIELDS[self.kind]
        if set(self.data) != allowed:
            raise ValueError(
                f"{self.kind.value} carries {sorted(self.data)}, declared {sorted(allowed)}"
            )

    def to_record(self) -> dict[str, Any]:
        return {"seq": self.seq, "step": self.step, "kind": self.kind.value, "data": self.data}


class EventLog:
    """Append-only. Steps never go backwards."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def append(self, step: int, kind: Kind, **data: Any) -> Event:
        if self._events and step < self._events[-1].step:
            raise ValueError(f"event at step {step} after one at step {self._events[-1].step}")
        event = Event(seq=len(self._events), step=step, kind=kind, data=data)
        self._events.append(event)
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __getitem__(self, i: int) -> Event:
        return self._events[i]

    def of(self, *kinds: Kind) -> list[Event]:
        return [e for e in self._events if e.kind in kinds]

    def to_record(self) -> list[dict[str, Any]]:
        return [e.to_record() for e in self._events]
