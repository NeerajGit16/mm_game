"""A game: one session, start to reveal.

The front end talks to a ``Game`` through the protocol and nothing else:
``start()`` once, ``handle(message)`` for every input. While the session is
live these return exactly what the engine says. When it ends -- at the close,
or at the close after a quit -- the game runs the three benchmark bots on the
identical world, builds the coaching record, writes the replay file, the
record, its markdown summary and the track-record line through the storage
layer, and appends one ``reveal`` message carrying the record. That message
is the first time anything hidden leaves the engine.

Whether the seed has been played before is decided here, from the track
record, before the session starts: a replay is flagged in both files and
excluded from the track record's aggregates.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from mm_game import agents, protocol, records
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings, open_session
from mm_game.storage import DirectoryStorage, Storage
from mm_game.world import World


class Game:
    def __init__(
        self,
        config: GameConfig,
        seed: int,
        *,
        storage: Storage,
        settings: Settings = Settings(),
        scoring: ScoringConfig = ScoringConfig(),
        benchmark: BenchmarkConfig = BenchmarkConfig(),
    ) -> None:
        self.storage = storage
        self.scoring = scoring
        self.benchmark = benchmark
        self.track = records.TrackRecord.load(storage)
        self.replay_of_played_seed = self.track.played(config, seed)
        self.session, self._world = open_session(
            config, seed, settings=settings, scoring=scoring, replay_of_played_seed=self.replay_of_played_seed
        )
        self.session_id: str | None = None
        self.record: dict[str, Any] | None = None

    def start(self) -> list[dict[str, Any]]:
        return self.session.start()

    def handle(self, message: Any) -> list[dict[str, Any]]:
        out = self.session.handle(message)
        if self.session.ended and self.record is None:
            out.append(protocol.reveal_message(self._finish()))
        return out

    def _finish(self) -> dict[str, Any]:
        world: World = self._world
        bots = {
            name: agents.run_bot(
                name, world, settings=self.session.settings, scoring=self.scoring, benchmark=self.benchmark
            )
            for name in agents.BOTS
        }
        flags = records.flags(self.session, self.replay_of_played_seed)
        self.record = records.coaching_record(
            world, self.session, scoring=self.scoring, benchmark=self.benchmark, bots=bots, flags=flags
        )
        replay = records.replay_file(
            world, self.session, scoring=self.scoring, benchmark=self.benchmark, flags=flags
        )
        self.session_id = self.track.next_id(self.session.settings, world.seed)
        base = f"sessions/{self.session_id}"
        self.storage.write_text(f"{base}/replay.json", records.dumps(replay))
        self.storage.write_text(f"{base}/coaching_record.json", records.dumps(self.record))
        self.storage.write_text(f"{base}/summary.md", records.markdown_summary(self.record))
        self.track.add(self.session_id, world.config, world.seed, self.record)
        self.track.save(self.storage)
        return self.record


def run_headless(game: Game, strategy: agents.Strategy) -> Game:
    """Play a whole game with a scripted strategy standing in for the player,
    through the same protocol a front end uses."""

    class _Bridge:
        # The bots' driver speaks to a Session; route it through the Game so
        # the post-session reveal and the files happen exactly as in play.
        ended = property(lambda self: game.session.ended)

        def start(self):
            return game.start()

        def handle(self, message):
            return game.handle(message)

    agents.play(_Bridge(), strategy)  # type: ignore[arg-type]
    return game


def _scripted(quit_at: int | None) -> Callable[[GameConfig], agents.Strategy]:
    return lambda cfg: agents.ScriptedPlayer(cfg, BenchmarkConfig(), quit_at_step=quit_at, let_timer_expire_every=25)


def main(argv: list[str] | None = None) -> None:
    """``python -m mm_game.game --preset 30 --seed 7 --out records/``: one
    headless session with the scripted player, both files written."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--preset", type=int, choices=(30, 60, 120), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quit-at-minute", type=float, default=None)
    args = parser.parse_args(argv)
    config = GameConfig.for_session(args.preset)
    quit_at = None if args.quit_at_minute is None else round(args.quit_at_minute * 60 / config.dt)
    game = Game(
        config,
        args.seed,
        storage=DirectoryStorage(args.out),
        settings=Settings(preset=args.preset, pricing_timer_s=15.0),
    )
    run_headless(game, _scripted(quit_at)(config))
    s = game.record["scores"]
    print(f"{game.session_id}: P&L ${s['pnl_usd']:,.0f}, risk-adjusted ${s['risk_adjusted_usd']:,.0f}")
    print(f"written under {args.out}/sessions/{game.session_id}/")


if __name__ == "__main__":
    main()
