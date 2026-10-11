"""The local server: the front end's static files, and one hosted session
over a websocket on localhost.

The transport only. Everything about the session -- the clock, the desk,
what may be sent and when -- is in ``host.py``, which has no I/O, so the
same front end can later drive a session hosted elsewhere or in the page.
This module is the one place in the package with an event loop and file
reads of its own: it serves ``web/``, which is the front end's code, not a
record (records still go through ``storage``).
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import time
import webbrowser
from pathlib import Path
from typing import Callable

import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import tornado.websocket

from mm_game.config import GameConfig
from mm_game.engine import Settings
from mm_game.game import Game, fresh_seed
from mm_game.host import Host, WallClock
from mm_game.records import TrackRecord
from mm_game.storage import DirectoryStorage

WEB_DIR = Path(__file__).resolve().parent / "web"
TICK_MS = 50


# --------------------------------------------------------------------------
# The web app
# --------------------------------------------------------------------------


class _Socket(tornado.websocket.WebSocketHandler):
    def initialize(self, host: Host) -> None:
        self.host = host

    def open(self) -> None:
        self.host.connect(self._send)

    def _send(self, text: str) -> None:
        try:
            self.write_message(text)
        except tornado.websocket.WebSocketClosedError:
            self.host.disconnect(self._send)

    def on_message(self, message: str | bytes) -> None:
        self.host.receive(message if isinstance(message, str) else message.decode("utf-8"))

    def on_close(self) -> None:
        self.host.disconnect(self._send)


class _Static(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path: str) -> None:
        self.set_header("Cache-Control", "no-store")


def make_app(host: Host) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/ws", _Socket, {"host": host}),
            (r"/(.*)", _Static, {"path": str(WEB_DIR), "default_filename": "index.html"}),
        ]
    )


def listen(host: Host, *, port: int = 0, tick_ms: int | None = TICK_MS):
    """Serve ``host`` on localhost. Call from inside the event loop. Returns
    the HTTP server, the port, and the clock's ticker (None without one)."""
    sockets = tornado.netutil.bind_sockets(port, "127.0.0.1")
    server = tornado.httpserver.HTTPServer(make_app(host))
    server.add_sockets(sockets)
    ticker = None
    if tick_ms:
        ticker = tornado.ioloop.PeriodicCallback(host.tick, tick_ms)
        ticker.start()
    return server, sockets[0].getsockname()[1], ticker


def new_host(
    preset: int,
    seed: int | None,
    *,
    records_dir: str,
    now: Callable[[], float] = time.monotonic,
    draw: Callable[[], int] = lambda: secrets.randbelow(2**31),
) -> Host:
    """A hosted session on ``seed``, or on a fresh one -- never played on
    this config -- when ``seed`` is None."""
    config = GameConfig.for_session(preset)
    storage = DirectoryStorage(records_dir)
    if seed is None:
        seed = fresh_seed(config, TrackRecord.load(storage), draw)
    game = Game(config, seed, storage=storage, settings=Settings(preset=preset))
    return Host(game, WallClock(config.dt, now), records_dir=records_dir)


def main(argv: list[str] | None = None) -> None:
    """``uv run play.py --preset 60 [--seed N]``: one session in the browser."""
    parser = argparse.ArgumentParser(prog="play.py", description="Play an mm_game session in the browser.")
    parser.add_argument("--preset", type=int, choices=(30, 60, 120), required=True)
    parser.add_argument("--seed", type=int, default=None, help="play this seed; a new one if omitted")
    parser.add_argument("--records", default="records", help="where records are written (default: records)")
    parser.add_argument("--port", type=int, default=0, help="localhost port (default: any free one)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)
    # Without --seed, a fresh one is drawn (never one already played on this
    # config) and not shown: everything hidden regenerates from it.
    host = new_host(args.preset, args.seed, records_dir=args.records)

    async def serve() -> None:
        _, port, _ = listen(host, port=args.port)
        url = f"http://127.0.0.1:{port}/"
        print(f"mm_game: {args.preset}-minute session at {url}  (Ctrl+C to stop the server)", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        await asyncio.Event().wait()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
