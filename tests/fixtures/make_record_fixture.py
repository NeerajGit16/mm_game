"""Save one session under the current formats, as a fixture for
``tests/test_saved_records.py``.

    uv run python tests/fixtures/make_record_fixture.py

Run it once after any bump of the protocol, the record format or the config
schema, and commit the new directory. It never overwrites an existing
fixture: those are sessions saved under earlier versions, and the test
loads every one of them.

The fixture is a records directory exactly as play writes it -- the track
record and one session's replay, coaching record and summary -- from the
scripted player on the 30-minute drill: quotes, passes, hedges, a pause,
timer expiries, a speed change and a quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mm_game import agents, game
from mm_game.config import CONFIG_SCHEMA_VERSION, BenchmarkConfig, GameConfig
from mm_game.engine import Settings
from mm_game.protocol import PROTOCOL_VERSION
from mm_game.records import FORMAT_VERSION
from mm_game.storage import DirectoryStorage

FIXTURES = Path(__file__).resolve().parent / "records"
PRESET, SEED, QUIT_MINUTE = 30, 3, 25


def current_name() -> str:
    return f"protocol{PROTOCOL_VERSION}-format{FORMAT_VERSION}-config{CONFIG_SCHEMA_VERSION}"


def main() -> None:
    target = FIXTURES / current_name()
    if target.exists():
        sys.exit(f"{target} exists; fixtures are never overwritten")
    config = GameConfig.for_session(PRESET)
    g = game.Game(
        config, SEED, storage=DirectoryStorage(target), settings=Settings(preset=PRESET, pricing_timer_s=15.0)
    )
    player = agents.ScriptedPlayer(
        config, BenchmarkConfig(), quit_at_step=round(QUIT_MINUTE * 60 / config.dt), let_timer_expire_every=9
    )
    game.run_headless(g, player)
    print(f"saved {g.session_id} under {target}")


if __name__ == "__main__":
    main()
