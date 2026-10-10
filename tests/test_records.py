"""Tests for records, storage and the game lifecycle.

The ones that matter most: a replay file reproduces its coaching record bit
for bit, in another process too; a loader refuses what it would misread; a
quit session rides to the close and says so; a replayed seed is flagged and
kept out of the track record; the benchmarks play the identical world; and
only the storage layer touches files. Each docstring says what silent
failure it catches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mm_game
from mm_game import agents, game, records, rng, world
from mm_game.config import CONFIG_SCHEMA_VERSION, BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Settings
from mm_game.events import Kind
from mm_game.storage import DirectoryStorage, MemoryStorage

SRC_DIR = Path(mm_game.__file__).resolve().parent
BENCH = BenchmarkConfig()


def headless(config: GameConfig, seed: int, storage=None, *, quit_at: int | None = None, preset: int | None = None):
    storage = storage if storage is not None else MemoryStorage()
    g = game.Game(config, seed, storage=storage, settings=Settings(preset=preset, pricing_timer_s=15.0))
    game.run_headless(g, agents.ScriptedPlayer(config, BENCH, quit_at_step=quit_at, let_timer_expire_every=9))
    return g, storage


def files(storage, g) -> tuple[dict, dict, str]:
    base = f"sessions/{g.session_id}"
    return (
        json.loads(storage.read_text(f"{base}/replay.json")),
        json.loads(storage.read_text(f"{base}/coaching_record.json")),
        storage.read_text(f"{base}/summary.md"),
    )


@pytest.fixture(scope="module")
def drill_game():
    return headless(GameConfig.for_session(30), 12, preset=30)


# --------------------------------------------------------------------------
# Done when: a headless session at each preset writes both files
# --------------------------------------------------------------------------


@pytest.mark.parametrize("minutes", [30, 60, 120])
def test_headless_session_runs_end_to_end_and_writes_both_files(minutes):
    """The scripted stand-in plays a whole preset session through the
    protocol; the replay file, the coaching record and its summary are
    written; every RFQ carries all four layers; the benchmarks and capture
    ratio are in the record."""
    g, storage = headless(GameConfig.for_session(minutes), 21, preset=minutes)
    replay, record, summary = files(storage, g)
    assert replay["format"] == records.REPLAY_FORMAT and replay["actions"]
    assert record["format"] == records.RECORD_FORMAT
    assert set(record["benchmarks"]) == {"naive", "disciplined", "oracle", "passing_oracle"}
    assert {"pnl", "risk_adjusted"} <= set(record["capture_ratio"])
    shown = [r for r in record["rfqs"] if r["shown"]]
    assert shown and all(
        r["observable"] and r["action"] and r["outcome"] and r["hidden"]["market"] for r in shown
    )
    assert summary.startswith("# Session summary") and "Flow oracle" in summary
    assert json.loads(records.dumps(record)) == record  # plain JSON, exactly


# --------------------------------------------------------------------------
# Determinism and replay
# --------------------------------------------------------------------------


def test_replay_reproduces_the_coaching_record_bit_for_bit(drill_game):
    """Re-run a replay file -- regenerate the world from (config, seed),
    re-apply the actions at their steps, re-run the benchmarks -- and the
    coaching record is byte-identical. Anything that depends on wall time,
    iteration order of a set, or a draw outside the world breaks this."""
    g, storage = drill_game
    replay, record, _ = files(storage, g)
    rf = records.load_replay(replay)
    session, w = records.replay(rf)
    bots = {b: agents.run_bot(b, w, settings=rf.settings, scoring=rf.scoring, benchmark=rf.benchmark) for b in agents.BOTS}
    again = records.coaching_record(w, session, scoring=rf.scoring, benchmark=rf.benchmark, bots=bots, flags=rf.flags)
    assert records.dumps(again) == records.dumps(record)


REPLAY_SCRIPT = """
import hashlib, json, sys
from mm_game import agents, records
rf = records.load_replay(json.loads(open(sys.argv[1], encoding="utf-8").read()))
session, w = records.replay(rf)
bots = {b: agents.run_bot(b, w, settings=rf.settings, scoring=rf.scoring, benchmark=rf.benchmark) for b in agents.BOTS}
rec = records.coaching_record(w, session, scoring=rf.scoring, benchmark=rf.benchmark, bots=bots, flags=rf.flags)
print(hashlib.sha256(records.dumps(rec).encode()).hexdigest())
"""


def test_replay_is_stable_across_processes(tmp_path, drill_game):
    """The same replay in fresh interpreters with different hash seeds gives
    the same record. Within one process, set ordering and dict-from-set
    construction are stable; between processes they are not, and a record
    that depended on them would replay correctly in tests and diverge on
    another machine."""
    g, storage = drill_game
    replay, record, _ = files(storage, g)
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    script = tmp_path / "replay_hash.py"
    script.write_text(REPLAY_SCRIPT, encoding="utf-8")
    expected = hashlib.sha256(records.dumps(record).encode()).hexdigest()
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, str(script), str(path)], env=env, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == expected


def test_paired_replay_differs_only_from_the_changed_rfq_on(drill_game):
    """Two action sequences differing at one RFQ: the event logs are
    identical up to that RFQ's quote, and the world -- every screen reading,
    print and RFQ arrival, and every market side -- is identical throughout.
    A market drawn lazily, or anything keyed on the count of actions, would
    shift the world after the change and the comparison would be between
    two different sessions."""
    g, storage = drill_game
    replay, _, _ = files(storage, g)
    rf = records.load_replay(replay)
    quotes = [a for a in rf.actions if a["type"] == "quote"]
    k = quotes[len(quotes) // 2]
    changed = [
        {"seq": a["seq"], "step": a["step"], "type": "pass", "rfq_ordinal": a["rfq_ordinal"]} if a is k else a
        for a in rf.actions
    ]
    s1, w1 = records.replay(rf)
    s2, w2 = records.replay(records.ReplayFile(**{**rf.__dict__, "actions": changed}))
    cut = next(e.seq for e in s1.log if e.kind is Kind.QUOTE_SENT and e.data["rfq_ordinal"] == k["rfq_ordinal"])
    assert [e for e in s1.log if e.seq < cut] == [e for e in s2.log if e.seq < cut]
    assert s1.log[cut] != s2.log[cut]
    world_kinds = (Kind.SCREEN_UPDATE, Kind.PRINT_RELEASED, Kind.RFQ_ARRIVED)
    strip = lambda log: [(e.step, e.kind, e.data) for e in log.of(*world_kinds)]
    assert strip(s1.log) == strip(s2.log)
    assert w1.sides == w2.sides and w1.schedule.rfqs == w2.schedule.rfqs


# --------------------------------------------------------------------------
# What a loader refuses
# --------------------------------------------------------------------------


def test_header_carries_the_config_schema_and_round_trips():
    w = world.generate(GameConfig(), 3)
    assert w.header["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert CONFIG_SCHEMA_VERSION == 2  # bumped when unannounced moves were added
    assert w.header["seed"] == 3 and w.header["canary"] == rng.canary()
    assert GameConfig.from_record(json.loads(json.dumps(w.header["config"]))) == GameConfig()
    for cls in (ScoringConfig, BenchmarkConfig):
        assert cls.from_record(json.loads(json.dumps(cls().to_record()))) == cls()


def _mutate(replay: dict, change) -> dict:
    r = copy.deepcopy(replay)
    change(r)
    return r


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda r: r["header"].__setitem__("config_schema_version", CONFIG_SCHEMA_VERSION + 1), "config schema"),
        (lambda r: r["header"].pop("config_schema_version"), "config schema"),
        (lambda r: r.__setitem__("format_version", 99), "replay file"),
        (lambda r: r.__setitem__("format", "something.else"), "replay file"),
        (lambda r: r["header"].__setitem__("scheme_version", 99), "scheme"),
        (lambda r: r["header"]["canary"]["normal"].__setitem__(0, 0.5), "transforms have changed"),
        (lambda r: r["header"]["config"].pop("hedge_cost_bp"), "does not match"),
        (lambda r: r["header"]["config"].__setitem__("hedge_cost_ticks", 2.0), "does not match"),
        (lambda r: r["scoring_config"].__setitem__("kappa", 1.0), "unreadable settings"),
    ],
)
def test_loader_refuses_what_it_would_misread(drill_game, change, message):
    """An unknown config-schema or format version, another stream scheme, a
    moved numpy transform, a config with a field missing or a field from an
    older layout: each is refused with a reason. Each, read leniently, would
    load and replay a different session than the one recorded -- a
    tick-denominated header read as bp, a missing field filled with today's
    default."""
    g, storage = drill_game
    replay, _, _ = files(storage, g)
    with pytest.raises(records.RecordError, match=message):
        records.load_replay(_mutate(replay, change))


# --------------------------------------------------------------------------
# Quit, replays of played seeds, paired benchmarks
# --------------------------------------------------------------------------


def test_quit_rides_to_the_close_and_is_flagged():
    """Quit at minute 20 of a two-hour session with a position on: no RFQ is
    shown after the quit, the position is carried through every later print
    and liquidated at the scheduled close at the final screen, and both
    files say where it ended."""
    cfg = GameConfig()
    seed = next(s for s in range(50) if (world.generate(cfg, s).path.event_steps > 12000).sum() >= 1)
    g, storage = headless(cfg, seed, quit_at=12000)
    replay, record, summary = files(storage, g)
    s = g.session
    assert s.quit_step == 12000
    assert replay["flags"]["ended_early_at_step"] == 12000 == record["flags"]["ended_early_at_step"]
    assert "Ended early at step 12000" in summary
    assert all(not r["shown"] for r in record["rfqs"] if r["step"] > 12000)
    quit_event = s.log.of(Kind.SESSION_QUIT)[0]
    close = s.log.of(Kind.CLOSE_LIQUIDATION)[0]
    assert close.step == cfg.n_steps and close.data["quantity"] == -quit_event.data["position"]
    assert not any(e.kind is Kind.RFQ_ARRIVED and e.step > 12000 for e in s.log)
    later = [p for p in record["diagnostics"]["position"]["into_prints"] if p["step"] > 12000]
    assert later and all(p["position_into_print"] == quit_event.data["position"] for p in later)


def test_replay_of_a_played_seed_is_flagged_and_excluded():
    """Play a seed, then play it again on the same store: the second is
    flagged in the replay file, the record and the track record, and left
    out of the aggregates -- you know where the prints went. The same seed
    on another preset is a different world and is not a replay."""
    storage = MemoryStorage()
    cfg = GameConfig.for_session(30)
    first, _ = headless(cfg, 5, storage, preset=30)
    second, _ = headless(cfg, 5, storage, preset=30)
    other, _ = headless(GameConfig.for_session(60), 5, storage, preset=60)
    assert not first.replay_of_played_seed and second.replay_of_played_seed and not other.replay_of_played_seed
    replay, record, summary = files(storage, second)
    assert replay["flags"]["replay_of_played_seed"] and record["flags"]["replay_of_played_seed"]
    assert "Replay of a seed already played" in summary
    track = records.TrackRecord.load(storage)
    assert [e["replay_of_played_seed"] for e in track.entries] == [False, True, False]
    agg = track.aggregates(min_gap_usd=1.0)
    assert (agg["sessions"], agg["replays_excluded"]) == (2, 1)
    scored = [track.entries[0], track.entries[2]]
    assert agg["pnl_usd"] == pytest.approx(sum(e["pnl_usd"] for e in scored) / 2)
    gap = sum(e["oracle_pnl_usd"] - e["disciplined_pnl_usd"] for e in scored)
    pooled = sum(e["pnl_usd"] - e["disciplined_pnl_usd"] for e in scored) / gap if gap >= 1.0 else None
    assert agg["pooled_capture_ratio_pnl"] == (pytest.approx(pooled) if pooled is not None else None)


def test_benchmarks_play_the_identical_world(monkeypatch):
    """Every bot is run on the very world the player faced -- same object --
    and sees the same screen, prints and RFQ arrivals. A benchmark on a
    regenerated or different world would make the capture ratio compare
    sessions that never faced the same flow."""
    seen = []
    real = agents.run_bot

    def spy(name, w, **kw):
        seen.append((name, w))
        return real(name, w, **kw)

    monkeypatch.setattr(agents, "run_bot", spy)
    g, _ = headless(GameConfig.for_session(30), 4, preset=30)
    assert [n for n, _ in seen] == list(agents.BOTS)
    assert all(w is g._world for _, w in seen)
    bot = real("naive", g._world, settings=g.session.settings, scoring=ScoringConfig(), benchmark=BENCH)
    arrivals = lambda s: [(e.step, e.data) for e in s.log.of(Kind.RFQ_ARRIVED)]
    assert arrivals(bot)[: len(arrivals(g.session))] == arrivals(g.session)


def test_summary_reveals_moves_and_every_client():
    """The summary reveals each unannounced move in its own line under
    "Position, prints and moves" -- size, ramp, warning, the position, the
    informed RFQs that foretold it -- and lists every client with true type
    and what they knew, net lots labelled as yours. Mark-outs read through
    the next event. A record that kept moves only in the JSON would hide
    the lesson from the review that reads the summary."""
    cfg = GameConfig.for_session(120)
    seed = next(s for s in range(100) if len(world.generate(cfg, s).path.move_list) >= 2)
    g, storage = headless(cfg, seed, preset=120)
    _, record, summary = files(storage, g)
    moves = record["diagnostics"]["position"]["moves"]
    assert len(moves) >= 2
    assert "## Position, prints and moves" in summary
    for m in moves:
        assert f"Unannounced move {m['move_index'] + 1}" in summary
    assert any(m["informed_rfqs"] for m in moves) and "foretold by RFQ" in summary
    table = [l for l in summary.splitlines() if l.startswith("| ") and l.split("|")[1].strip().isdigit()]
    assert len(table) >= cfg.n_clients
    assert "| Client | Category | True type |" in summary
    assert "| Your net lots | 60s | 300s | Next event |" in summary
    assert "| | 10s | 60s | 300s | next event |" in summary


def test_reveal_is_sent_once_and_only_after_the_end(drill_game):
    g, _ = drill_game
    assert g.record is not None
    assert g.handle({"v": 1, "type": "pause"})[0]["type"] == "error"  # nothing after the end


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_only_the_storage_layer_touches_files():
    """All file I/O goes through ``storage`` so it can become a browser
    download or local storage later. Any other module opening a file, or
    reaching for pathlib or os to write one, would be a second path the
    in-browser build has to find."""
    pattern = re.compile(r"\bopen\(|\.write_text\(|\.read_text\(|^\s*import (os|shutil|io)\b|from pathlib", re.M)
    for path in SRC_DIR.glob("*.py"):
        if path.name == "storage.py":
            continue
        code = path.read_text(encoding="utf-8")
        hits = [m.group(0) for m in pattern.finditer(code)]
        allowed = {".write_text(", ".read_text("} if path.name in ("game.py", "records.py") else set()
        assert set(hits) <= allowed, (path.name, hits)
        for line in code.splitlines():
            if ".write_text(" in line or ".read_text(" in line:
                assert "storage" in line, (path.name, line)


@pytest.mark.parametrize("make", [MemoryStorage, lambda: DirectoryStorage("TMP")])
def test_storages_behave_alike_and_refuse_escapes(make, tmp_path):
    store = make() if make is MemoryStorage else DirectoryStorage(tmp_path)
    store.write_text("sessions/a/replay.json", "{}")
    assert store.exists("sessions/a/replay.json") and store.read_text("sessions/a/replay.json") == "{}"
    assert store.names("sessions/") == ["sessions/a/replay.json"]
    for bad in ("../x", "/abs", "a\\b", "a//b", "", "a/./b"):
        with pytest.raises(ValueError):
            store.write_text(bad, "x")
