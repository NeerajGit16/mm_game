"""Hosting one session: the clock, the desk and whoever is watching.

What the engine deliberately does not own, and the front end must not:

- **The clock.** The engine never reads one. ``Host.tick`` turns wall time
  and the session's speed into ``advance`` messages. The clock runs only
  once begun, while the session is live and unpaused and no RFQ is on the
  ticket, and restarts from wherever the engine stopped: time spent pricing
  an RFQ is never caught up.
- **The desk** (``desk.py``): the player's own records, derived from the
  messages already sent.
- **The watchers.** Every frame is a JSON array of protocol messages: one
  batch per input or clock tick. A page that connects mid-session is sent
  the public history so far and rebuilds from it.

What leaves during play is exactly what the engine sends -- events,
snapshots, errors -- plus the desk, the handbook and a small
``host`` status message, none of them built from anything hidden. The
coaching record leaves once, in the ``reveal`` message after the session
has ended, followed by the markdown summary and the track record.

Inbound, actions pass through to the engine except ``advance``: the clock
is the host's. One message of its own, ``begin``, starts the clock the
first time.

Pure: no I/O, no threads, no event loop. A transport hands it raw messages
and a send callback -- ``server.py`` over a websocket, or a page running the
engine itself.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Protocol

from mm_game import handbook, protocol
from mm_game.desk import Desk
from mm_game.engine import Session
from mm_game.game import Game
from mm_game.protocol import ActionType, ProtocolError

V = protocol.PROTOCOL_VERSION


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------


class Clock(Protocol):
    def target(self, step: int, speed: int, running: bool) -> int:
        """The step the session should be at now."""


class WallClock:
    """Wall time to steps.

    The target is the step the session was at when the clock last started,
    plus the wall time since then times the speed. Anything that stops the
    clock -- not begun, paused, an RFQ on the ticket -- drops that anchor,
    so the next start is from the session's own step and the stopped time
    is never caught up. A speed change re-anchors, so only the time after it
    runs at the new speed. ``now`` is injectable: tests run the clock fast.
    """

    def __init__(self, dt: float, now: Callable[[], float] = time.monotonic) -> None:
        self._dt = dt
        self._now = now
        self._anchor: tuple[float, int, int] | None = None  # wall time, step, speed

    def target(self, step: int, speed: int, running: bool) -> int:
        if not running:
            self._anchor = None
            return step
        now = self._now()
        if self._anchor is None or self._anchor[2] != speed:
            self._anchor = (now, step, speed)
            return step
        t0, s0, _ = self._anchor
        return max(step, s0 + math.floor((now - t0) * speed / self._dt))


# --------------------------------------------------------------------------
# One hosted session
# --------------------------------------------------------------------------


def frame(messages: list[dict[str, Any]]) -> str:
    """One websocket frame: a JSON array of protocol messages."""
    return "[" + ",".join(protocol.dumps(m) for m in messages) + "]"


class Host:
    """One session, its clock, the desk, and whoever is watching."""

    def __init__(self, game: Game, clock: Clock, *, records_dir: str) -> None:
        self.game = game
        self.clock = clock
        self.records_dir = records_dir
        self.desk = Desk()
        self.begun = False
        self._sockets: list[Callable[[str], None]] = []
        self._events: list[dict[str, Any]] = []  # every event sent: the public history
        self._latest: dict[str, dict[str, Any]] = {}  # last state, desk, reveal, summary
        self._desk_sent: str | None = None
        self._handbook = handbook.message(
            game.session.config, game.scoring, scored_sessions=game.track.scored_sessions()
        )
        self._publish(game.start())

    @property
    def session(self) -> Session:
        return self.game.session

    def running(self) -> bool:
        s = self.session
        return self.begun and not s.ended and not s.paused and s.pending is None

    # -- the clock ------------------------------------------------------------

    def tick(self) -> bool:
        """Move the session to where the clock says. True if it moved."""
        s = self.session
        running = self.running()
        target = min(self.clock.target(s.step, s.speed, running), s.n_steps)
        if not running or target <= s.step:
            return False
        self._publish(self.game.handle({"v": V, "type": ActionType.ADVANCE.value, "to_step": target}))
        return True

    # -- inbound --------------------------------------------------------------

    def receive(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            message = raw  # the engine refuses anything that is not an object
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "begin" and message.get("v") == V:
            self.begun = True
            self._publish([self._host_message()])
        elif kind == ActionType.ADVANCE.value:
            refused = ProtocolError("the server runs the clock; advance is not accepted from the front end")
            self._publish([protocol.error_message(self.session.step, refused), protocol.state_message(self.session.state())])
        else:
            self._publish(self.game.handle(message))

    # -- outbound -------------------------------------------------------------

    def _host_message(self) -> dict[str, Any]:
        return {"v": V, "type": "host", "begun": self.begun}

    def _publish(self, messages: list[dict[str, Any]]) -> None:
        batch = list(messages)
        for m in messages:
            self.desk.feed(m)
            if m["type"] == "event":
                self._events.append(m)
            elif m["type"] in ("state", "reveal"):
                self._latest[m["type"]] = m
        desk = self.desk.message()
        encoded = protocol.dumps(desk)
        if encoded != self._desk_sent:
            self._desk_sent = encoded
            self._latest["desk"] = desk
            batch.append(desk)
        if any(m["type"] == "reveal" for m in messages):
            summary = {
                "v": V,
                "type": "summary",
                "session_id": self.game.session_id,
                "saved_to": f"{self.records_dir}/sessions/{self.game.session_id}/",
                "markdown": self.game.summary,
            }
            track = self.game.track.message()
            self._latest["summary"] = summary
            self._latest["track"] = track
            batch += [summary, track]
        text = frame(batch)
        for send in list(self._sockets):
            send(text)

    def connect(self, send: Callable[[str], None]) -> None:
        """A page connected: send it everything public so far."""
        self._sockets.append(send)
        latest = [self._latest[k] for k in ("state", "desk") if k in self._latest]
        after = [self._latest[k] for k in ("reveal", "summary", "track") if k in self._latest]
        send(frame([*self._events, *latest, self._host_message(), self._handbook, *after]))

    def disconnect(self, send: Callable[[str], None]) -> None:
        if send in self._sockets:
            self._sockets.remove(send)
