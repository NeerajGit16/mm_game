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
from mm_game import agents, game, protocol, records, rng, world
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
        (lambda r: r.__setitem__("protocol_version", 99), "protocol 99"),
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
    agg = track.aggregates()
    assert (agg["sessions"], agg["replays_excluded"]) == (2, 1)
    scored = [track.entries[0], track.entries[2]]
    assert agg["pnl_usd"] == pytest.approx(sum(e["pnl_usd"] for e in scored) / 2)
    gap = sum(e["oracle_pnl_usd"] - e["disciplined_pnl_usd"] for e in scored)
    assert agg["pnl"]["oracle_minus_disciplined"]["total_usd"] == pytest.approx(gap)
    pooled = sum(e["pnl_usd"] - e["disciplined_pnl_usd"] for e in scored) / gap if gap >= 10_000 else None
    assert agg["pnl"]["capture_ratio"] == (pytest.approx(pooled) if pooled is not None else None)
    # Each summary carries the track record as it stood when it was written.
    assert "This practice replay is not counted; the track record holds 1 scored session (1 practice replay left out)." in summary
    assert "After this session the track record holds 1 scored session." in files(storage, first)[2]
    assert "After this session the track record holds 2 scored sessions (1 practice replay left out)." in files(storage, other)[2]


def track_entry(sid: str, you: float, disc: float, oracle: float, *, risk: tuple[float, float, float] | None = None,
                replay: bool = False) -> dict:
    """A track-record line with the scores given: P&L (you, disciplined,
    oracle) and risk-adjusted (the same unless given)."""
    ry, rd, ro = risk if risk is not None else (you, disc, oracle)
    return {
        "session_id": sid, "config_digest": "x", "seed": 1, "preset": 60,
        "replay_of_played_seed": replay, "ended_early_at_step": None,
        "pnl_usd": you, "risk_adjusted_usd": ry,
        "capture_ratio_pnl": None, "capture_ratio_risk_adjusted": None,
        "disciplined_pnl_usd": disc, "disciplined_risk_adjusted_usd": rd,
        "oracle_pnl_usd": oracle, "oracle_risk_adjusted_usd": ro,
    }


# Three scored sessions and a practice replay, worked out by hand.
#   P&L, you - disciplined:     1,000   3,000   5,000  -> sum 9,000, mean 3,000
#     deviations -2,000, 0, 2,000; sample variance 8e6 / 2 = 4e6;
#     standard error sqrt(4e6 / 3) = 1,154.70
#   P&L, oracle - disciplined:  4,000  -2,000   7,000  -> sum 9,000, mean 3,000
#     deviations 1,000, -5,000, 4,000; sample variance 42e6 / 2 = 21e6;
#     standard error sqrt(21e6 / 3) = sqrt(7e6) = 2,645.75
#   Risk-adjusted, you - disciplined: 0, 600, -300 -> sum 300, mean 100;
#     deviations -100, 500, -400; variance 42e4 / 2 = 21e4; s.e. sqrt(7e4) = 264.58
#   Risk-adjusted, oracle - disciplined: 3,000, 3,000, 3,000 -> sum 9,000, s.e. 0
# The replay's +$50,000 against the disciplined bot, if it were counted,
# would move every one of these.
HAND = [
    track_entry("a", 11_000, 10_000, 14_000, risk=(9_000, 9_000, 12_000)),
    track_entry("b", 5_000, 2_000, 0, risk=(1_600, 1_000, 4_000)),
    track_entry("c", -1_000, -6_000, 1_000, risk=(-5_300, -5_000, -2_000)),
    track_entry("r", 60_000, 10_000, 70_000, replay=True),
]


def test_the_pooled_gaps_and_standard_errors_match_a_hand_calculation():
    """Per session and pooled, your gap and the oracle's to the disciplined
    bot, with the mean per session and its standard error (sample standard
    deviation over root n), on both scores, are the numbers worked out by
    hand above -- the replay left out, and the session the oracle lost
    (b: -$2,000) kept in. A population standard deviation, a replay
    counted, or a pool that dropped the oracle's losing sessions would each
    move a number."""
    track = records.TrackRecord(copy.deepcopy(HAND))
    agg = track.aggregates()
    assert (agg["sessions"], agg["replays_excluded"]) == (3, 1)
    want = {
        ("pnl", "you_minus_disciplined"): (9_000, 3_000, 1_154.7005),
        ("pnl", "oracle_minus_disciplined"): (9_000, 3_000, 2_645.7513),
        ("risk_adjusted", "you_minus_disciplined"): (300, 100, 264.5751),
        ("risk_adjusted", "oracle_minus_disciplined"): (9_000, 3_000, 0.0),
    }
    for (score, who), (total, mean, se) in want.items():
        got = agg[score][who]
        assert (got["total_usd"], got["mean_usd"]) == (pytest.approx(total), pytest.approx(mean)), (score, who)
        assert got["se_usd"] == pytest.approx(se, abs=1e-3), (score, who)
    entries = {e["session_id"]: e for e in track.message()["entries"]}
    assert set(entries) == {"a", "b", "c", "r"}  # the replay is shown, not counted
    assert [entries[s]["oracle_minus_disciplined_pnl_usd"] for s in "abcr"] == [4_000, -2_000, 7_000, 60_000]
    assert [entries[s]["you_minus_disciplined_risk_adjusted_usd"] for s in "abcr"] == [0, 600, -300, 50_000]
    one = records.TrackRecord(copy.deepcopy(HAND[:1])).aggregates()["pnl"]["you_minus_disciplined"]
    assert (one["total_usd"], one["mean_usd"], one["se_usd"]) == (1_000, 1_000, None)
    lines = records.track_lines(agg, counted=True)
    assert "| You, P&L | +$9,000 | +$3,000 | $1,155 |" in lines
    assert "| Flow oracle, P&L | +$9,000 | +$3,000 | $2,646 |" in lines
    assert "| You, risk-adjusted | +$300 | +$100 | $265 |" in lines


def test_the_pooled_capture_ratio_is_hidden_below_10000_and_shown_at_it():
    """The pooled ratio stays n/a, with how far the oracle's pooled edge has
    got, until that edge is $10,000; at exactly $10,000 it appears, on each
    score by its own pooled gap. The per-session $2,000 floor does not open
    it (three sessions of $2,500 each stay n/a), and neither does a
    practice replay with a huge oracle edge."""
    below = HAND[:3] + [track_entry("d", 0, 0, 999)]  # oracle pooled: 9,999
    agg = records.TrackRecord(copy.deepcopy(below)).aggregates()
    assert agg["pnl"]["oracle_minus_disciplined"]["total_usd"] == 9_999
    assert agg["pnl"]["capture_ratio"] is None
    text = "\n".join(records.track_lines(agg, counted=True))
    assert "Pooled capture ratio: P&L n/a — not enough flow-reading edge yet ($9,999 of $10,000)" in text

    # Oracle pooled: P&L 10,000 (you 9,000); risk-adjusted 9,999.
    at = HAND[:3] + [track_entry("d", 0, 0, 1_000, risk=(0, 0, 999))]
    agg = records.TrackRecord(copy.deepcopy(at)).aggregates()
    assert agg["pnl"]["capture_ratio"] == pytest.approx(9_000 / 10_000)
    assert agg["risk_adjusted"]["capture_ratio"] is None
    assert ("Pooled capture ratio: P&L 0.90; risk-adjusted n/a — not enough flow-reading edge yet ($9,999 of $10,000)."
            in records.track_lines(agg, counted=True))
    # And the other way round: risk-adjusted at 10,000 (you 300), P&L at 9,999.
    other = HAND[:3] + [track_entry("d", 0, 0, 999, risk=(0, 0, 1_000))]
    agg = records.TrackRecord(copy.deepcopy(other)).aggregates()
    assert (agg["pnl"]["capture_ratio"], agg["risk_adjusted"]["capture_ratio"]) == (None, pytest.approx(300 / 10_000))

    two_thousand_each = [track_entry(s, 0, 0, 2_500) for s in "abc"]  # 7,500 pooled
    agg = records.TrackRecord(two_thousand_each).aggregates()
    assert agg["pnl"]["capture_ratio"] is None
    with_replay = copy.deepcopy(below) + [track_entry("r", 0, 0, 100_000, replay=True)]
    assert records.TrackRecord(with_replay).aggregates()["pnl"]["capture_ratio"] is None


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
    assert "| Client | Category | True type | RFQs | Informed RFQs (filled) |" in summary
    assert (
        "| Lots traded (gross) | Your net (+ = you bought) | Mark-out 60s | Mark-out 300s | "
        "Mark-out to next print/move, informed fills | Mark-out to next print/move, other fills |"
    ) in summary
    assert "| | Mark-out 10s | Mark-out 60s | Mark-out 300s | Mark-out to next print/move |" in summary


def test_reveal_is_sent_once_and_only_after_the_end(drill_game):
    g, _ = drill_game
    assert g.record is not None
    assert g.handle({"v": protocol.PROTOCOL_VERSION, "type": "pause"})[0]["type"] == "error"  # nothing after the end


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_only_the_storage_layer_touches_files():
    """All file I/O goes through ``storage`` so it can become a browser
    download or local storage later. Any other module opening a file, or
    reaching for pathlib or os to write one, would be a second path the
    in-browser build has to find."""
    # The builtin open(), not a method or a definition named open (a
    # websocket handler's open, webbrowser.open).
    pattern = re.compile(
        r"(?<![.\w])(?<!def )open\(|\.write_text\(|\.read_text\(|^\s*import (os|shutil|io)\b|from pathlib", re.M
    )
    for path in SRC_DIR.glob("*.py"):
        if path.name == "storage.py":
            continue
        code = path.read_text(encoding="utf-8")
        hits = [m.group(0) for m in pattern.finditer(code)]
        allowed = {".write_text(", ".read_text("} if path.name in ("game.py", "records.py") else set()
        if path.name == "server.py":  # it finds the front end's files; records it reads through storage
            allowed = {"from pathlib", ".read_text("}
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
