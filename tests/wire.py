"""Test helpers, not tests: a hosted session served on a background event
loop, and a synchronous websocket client to it.

The server runs on its own thread's event loop, exactly as in play; the
test thread talks to it only through the websocket, and reaches into the
host only through ``LoopThread.call``, which runs on the server's thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import tornado.websocket

from mm_game import host as host_module, protocol, server

V = protocol.PROTOCOL_VERSION
ENGINE_TYPES = {"event", "state", "error", "reveal"}


class ManualClock:
    """The test says where the clock is: ``to`` is the step it targets."""

    def __init__(self) -> None:
        self.to = 0

    def target(self, step: int, speed: int, running: bool) -> int:
        return max(step, self.to) if running else step


class LoopThread:
    """An asyncio event loop on a daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, fn, *args):
        async def on_loop():
            return fn(*args)

        return self.run(on_loop())

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


def serve(lt: LoopThread, host: host_module.Host, *, tick_ms: int | None) -> int:
    async def start():
        _, port, _ = server.listen(host, tick_ms=tick_ms)
        return port

    return lt.run(start())


class Wire:
    """A websocket client to a hosted session, driven synchronously. Every
    frame received is kept, in order."""

    def __init__(self, lt: LoopThread, host: host_module.Host, port: int) -> None:
        self.lt = lt
        self.host = host
        self.frames: list[list[dict[str, Any]]] = []

        async def connect():
            return await tornado.websocket.websocket_connect(f"ws://127.0.0.1:{port}/ws")

        self.conn = lt.run(connect())
        self.first = self.read()  # the history sent on connect

    def read(self) -> list[dict[str, Any]]:
        async def read():
            return await self.conn.read_message()

        raw = self.lt.run(read())
        if raw is None:
            raise ConnectionError("the server closed the websocket")
        batch = json.loads(raw)
        self.frames.append(batch)
        return batch

    def send(self, message: Any) -> list[dict[str, Any]]:
        async def write():
            await self.conn.write_message(json.dumps(message))

        self.lt.run(write())
        return self.read()

    def advance(self, to_step: int) -> list[dict[str, Any]]:
        """Move the server's clock to ``to_step`` and run one tick. The
        engine stops early at an RFQ, as always."""
        self.host.clock.to = to_step
        moved = self.lt.call(self.host.tick)
        return self.read() if moved else []

    def messages(self) -> list[dict[str, Any]]:
        return [m for f in self.frames for m in f]

    def close(self) -> None:
        self.lt.call(self.conn.close)


class WireSession:
    """The interface ``agents.play`` drives (``start``, ``handle``,
    ``ended``), over the websocket. Returns only the engine's own messages,
    so a strategy sees exactly what it would see headless."""

    def __init__(self, wire: Wire) -> None:
        self.wire = wire
        self.ended = False

    def _engine(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = [m for m in batch if m["type"] in ENGINE_TYPES]
        if any(m.get("kind") == "session_end" for m in out):
            self.ended = True
        return out

    def start(self) -> list[dict[str, Any]]:
        self.wire.send({"v": V, "type": "begin"})
        return self._engine(self.wire.first)

    def handle(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        if message["type"] == "advance":
            return self._engine(self.wire.advance(message["to_step"]))
        return self._engine(self.wire.send(message))
