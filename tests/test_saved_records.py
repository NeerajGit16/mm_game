"""Recorded sessions stay readable.

Real sessions feed the track record and coaching reviews, so a format change
must never strand them. ``tests/fixtures/records/`` keeps one session saved
under every (protocol, record format, config schema) there has been, written
by ``tests/fixtures/make_record_fixture.py`` and never overwritten. Every one
must load with today's code -- through a reader per version or a migration
-- and replay to the score it recorded. Each docstring says what silent
failure it catches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mm_game import agents, records
from mm_game.events import Kind
from mm_game.records import TrackRecord
from mm_game.storage import DirectoryStorage

from fixtures.make_record_fixture import FIXTURES, current_name

SAVED = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir())


def test_a_session_saved_under_the_current_formats_is_kept():
    """A fixture exists for today's protocol, record format and config
    schema. A version bump without one would leave nothing to prove the
    next bump keeps today's sessions readable."""
    assert current_name() in SAVED, (
        f"no saved session for {current_name()}: run `uv run python tests/fixtures/make_record_fixture.py` "
        "and commit the new directory"
    )


@pytest.mark.parametrize("name", SAVED)
def test_every_saved_session_still_loads_and_replays_to_its_score(name):
    """The track record, the replay file and the coaching record saved under
    that version all load; the replay regenerates the world and re-applies
    the inputs to the same P&L and risk-adjusted score the record and the
    track record hold. A loader that refused an old version, or read it
    into a different session, fails here."""
    storage = DirectoryStorage(FIXTURES / name)
    track = TrackRecord.load(storage)
    assert track.entries
    for entry in track.entries:
        base = f"sessions/{entry['session_id']}"
        rf = records.load_replay(json.loads(storage.read_text(f"{base}/replay.json")))
        record = records.load_coaching_record(json.loads(storage.read_text(f"{base}/coaching_record.json")))
        session, _ = records.replay(rf)
        end = session.log.of(Kind.SESSION_END)[0].data
        assert end["pnl_usd"] == record["scores"]["pnl_usd"] == entry["pnl_usd"]
        assert end["risk_adjusted_usd"] == record["scores"]["risk_adjusted_usd"] == entry["risk_adjusted_usd"]
        assert storage.read_text(f"{base}/summary.md").startswith("# Session summary")


def test_the_current_fixture_replays_to_its_coaching_record_bit_for_bit():
    """Under today's formats, the saved replay rebuilds the saved coaching
    record exactly, bots and all: the fixture is a real session, not a file
    that merely parses."""
    storage = DirectoryStorage(FIXTURES / current_name())
    entry = TrackRecord.load(storage).entries[0]
    base = f"sessions/{entry['session_id']}"
    rf = records.load_replay(json.loads(storage.read_text(f"{base}/replay.json")))
    session, w = records.replay(rf)
    bots = {b: agents.run_bot(b, w, settings=rf.settings, scoring=rf.scoring, benchmark=rf.benchmark) for b in agents.BOTS}
    again = records.coaching_record(w, session, scoring=rf.scoring, benchmark=rf.benchmark, bots=bots, flags=rf.flags)
    assert records.dumps(again) == storage.read_text(f"{base}/coaching_record.json")
