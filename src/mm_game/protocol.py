"""The message protocol between the engine and any front end.

Versioned, JSON-serialisable, and the only way a front end talks to a session,
so the same front end works whether the engine is a local server, a hosted one
or running in the page.

Out, during play: ``event`` messages (one per logged event, its public
payload only), a ``state`` snapshot after every input, and ``error`` for a
refused input. Out, after the session: one ``reveal`` message carrying the
coaching record. Hidden fields are never part of the in-session vocabulary --
event payloads are declared in ``events.FIELDS`` and snapshots are built here
from public state only -- and the reveal is a separate message type, sent
once the session has ended.

In: player and front-end actions. Advancing the clock is an input like any
other: the front end turns wall time and speed into steps and asks for them.
The engine never reads a clock.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mm_game.events import Event

PROTOCOL_VERSION: int = 2  # 2: the snapshot carries the venue's current cost

# Speeds a front end may run the clock at. Presentation only: the world is
# fixed in steps, and speed changes are logged so a review can say "you were
# at 4x when the print hit".
SPEEDS: frozenset[int] = frozenset({1, 2, 4})

# Hedge sizes the snapshot prices the venue at, so a front end can show what
# its hedge buttons would cost now without reimplementing the cost model.
VENUE_QUOTE_SIZES: tuple[int, ...] = (10, 25, 50)


class ActionType(StrEnum):
    ADVANCE = "advance"  # move the clock; stops at the next RFQ
    QUOTE = "quote"  # answer the RFQ on the ticket
    PASS = "pass"  # decline it
    HEDGE = "hedge"  # trade at the venue, stamped at the current step
    QUIT = "quit"  # stop trading; the position rides to the close
    SET_SPEED = "set_speed"
    PAUSE = "pause"
    RESUME = "resume"
    TIMER_EXPIRED = "timer_expired"  # the optional pricing timer ran out: a pass


# Required parameters per action, beyond "v" and "type".
_PARAMS: dict[ActionType, frozenset[str]] = {
    ActionType.ADVANCE: frozenset({"to_step"}),
    ActionType.QUOTE: frozenset({"rfq_ordinal", "bid", "ask"}),
    ActionType.PASS: frozenset({"rfq_ordinal"}),
    ActionType.HEDGE: frozenset({"quantity"}),
    ActionType.QUIT: frozenset(),
    ActionType.SET_SPEED: frozenset({"speed"}),
    ActionType.PAUSE: frozenset(),
    ActionType.RESUME: frozenset(),
    ActionType.TIMER_EXPIRED: frozenset({"rfq_ordinal"}),
}


class ProtocolError(ValueError):
    """A message the engine cannot accept. Raised before any state changes."""


@dataclass(frozen=True)
class Action:
    type: ActionType
    params: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        return {"v": PROTOCOL_VERSION, "type": self.type.value, **self.params}


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer, got {value!r}")
    return value


def _price(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"{name} must be a finite number or null, got {value!r}")
    return float(value)


def parse_action(message: Any) -> Action:
    """Validate an inbound message into an ``Action``. Strict: an unknown
    version, type or parameter is refused rather than guessed at."""
    if not isinstance(message, dict):
        raise ProtocolError("a message must be a JSON object")
    if message.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {message.get('v')!r}; this engine speaks {PROTOCOL_VERSION}")
    try:
        kind = ActionType(message.get("type"))
    except ValueError:
        raise ProtocolError(f"unknown action type {message.get('type')!r}") from None
    params = {k: v for k, v in message.items() if k not in ("v", "type")}
    if set(params) != _PARAMS[kind]:
        raise ProtocolError(f"{kind.value} takes {sorted(_PARAMS[kind])}, got {sorted(params)}")
    if "rfq_ordinal" in params:
        _int(params["rfq_ordinal"], "rfq_ordinal")
    if kind is ActionType.ADVANCE:
        _int(params["to_step"], "to_step")
    if kind is ActionType.QUOTE:
        params = {**params, "bid": _price(params["bid"], "bid"), "ask": _price(params["ask"], "ask")}
    if kind is ActionType.HEDGE and _int(params["quantity"], "quantity") == 0:
        raise ProtocolError("hedge quantity must be non-zero")
    if kind is ActionType.SET_SPEED and params["speed"] not in SPEEDS:
        raise ProtocolError(f"speed must be one of {sorted(SPEEDS)}, got {params['speed']!r}")
    return Action(kind, params)


# -- outbound -------------------------------------------------------------------


def event_message(event: Event) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION,
        "type": "event",
        "seq": event.seq,
        "step": event.step,
        "kind": event.kind.value,
        "data": event.data,
    }


def state_message(state: dict[str, Any]) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "state", **state}


def error_message(step: int, error: Exception) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "error", "step": step, "detail": str(error)}


def reveal_message(record: dict[str, Any]) -> dict[str, Any]:
    """The post-session reveal: the coaching record, hidden layer and all.
    Never sent while a session is live."""
    return {"v": PROTOCOL_VERSION, "type": "reveal", "record": record}


def dumps(message: dict[str, Any]) -> str:
    """Serialise for the wire. NaN and infinity are refused: they are not
    JSON, and a front end in another language would choke on them."""
    return json.dumps(message, sort_keys=True, allow_nan=False)
