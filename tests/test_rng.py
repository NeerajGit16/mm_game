"""Tests for mm_game.rng.

The independence test is the one that earns the construct; the cross-process
test is the one that catches the ``hash()`` trap. Both come first.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game.rng import (
    CANARY_SEED,
    SCHEME_VERSION,
    RngStreams,
    Stream,
    canary,
    stream_key,
)
from mm_game.rng import _RUN_KEY

SRC_DIR = Path(mm_game.__file__).resolve().parent


# --------------------------------------------------------------------------
# The two that matter
# --------------------------------------------------------------------------


def test_streams_are_independent():
    """Variable consumption on one stream must not move another.

    This is the whole point: change how much flow you draw, and the market path
    stays bit-identical.
    """
    a = RngStreams(42)
    b = RngStreams(42)

    a.stream(Stream.FLOW_ARRIVAL).exponential(size=1000)
    from_a = a.stream(Stream.FAIR_VALUE).normal(size=5)
    from_b = b.stream(Stream.FAIR_VALUE).normal(size=5)

    np.testing.assert_array_equal(from_a, from_b)


def test_stable_across_processes():
    """Keys must not depend on PYTHONHASHSEED.

    Builtin ``hash()`` of a str is randomised per process. A scheme built on it
    replays correctly within one process and silently diverges between them --
    this is the only test that catches that.
    """
    script = (
        "from mm_game.rng import RngStreams, Stream\n"
        "r = RngStreams(12345)\n"
        "print([r.stream(s).normal() for s in Stream])\n"
    )
    outputs = []
    for hash_seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_DIR.parent), env.get("PYTHONPATH", "")]
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1], "stream derivation depends on PYTHONHASHSEED"


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_draws():
    a, b = RngStreams(2024), RngStreams(2024)
    for name in Stream:
        np.testing.assert_array_equal(
            a.stream(name).normal(size=10), b.stream(name).normal(size=10)
        )


def test_different_seeds_give_different_draws():
    a, b = RngStreams(1), RngStreams(2)
    for name in Stream:
        assert not np.array_equal(
            a.stream(name).normal(size=10), b.stream(name).normal(size=10)
        )


def test_order_of_first_touch_does_not_matter():
    """Name-keyed, not positional: touching streams in a different order is fine."""
    forward = RngStreams(99)
    reverse = RngStreams(99)

    for name in Stream:
        forward.stream(name).normal(size=3)
    for name in reversed(list(Stream)):
        reverse.stream(name).normal(size=3)

    for name in Stream:
        np.testing.assert_array_equal(
            forward.stream(name).normal(size=5), reverse.stream(name).normal(size=5)
        )


# --------------------------------------------------------------------------
# Memoization
# --------------------------------------------------------------------------


def test_stream_is_memoized():
    rng = RngStreams(7)
    assert rng.stream(Stream.AGENTS) is rng.stream(Stream.AGENTS)


def test_repeated_calls_advance_rather_than_restart():
    """Without memoization every call restarts from the same state.

    The failure is not "slightly wrong": every draw returns the *same* number,
    giving a degenerate constant-innovation path that looks plausible and never
    raises.
    """
    rng = RngStreams(7)
    draws = [rng.stream(Stream.FAIR_VALUE).normal() for _ in range(5)]
    assert len(set(draws)) == 5


# --------------------------------------------------------------------------
# Substreams
# --------------------------------------------------------------------------


def test_substreams_are_isolated():
    rng = RngStreams(5)
    before = rng.substream(Stream.FLOW_SIZE, 3).normal(size=4)
    rng.substream(Stream.FLOW_SIZE, 2).normal(size=500)
    after = rng.substream(Stream.FLOW_SIZE, 3).normal(size=4)
    np.testing.assert_array_equal(before, after)


def test_substream_is_reproducible_and_index_keyed():
    a, b = RngStreams(5), RngStreams(5)
    np.testing.assert_array_equal(
        a.substream(Stream.FLOW_SIZE, 9).normal(size=4),
        b.substream(Stream.FLOW_SIZE, 9).normal(size=4),
    )
    assert not np.array_equal(
        a.substream(Stream.FLOW_SIZE, 9).normal(size=4),
        a.substream(Stream.FLOW_SIZE, 10).normal(size=4),
    )


def test_substream_does_not_alias_stream():
    """spawn_key (k,) and (k, 0) must land on different states."""
    rng = RngStreams(11)
    assert not np.array_equal(
        rng.stream(Stream.FAIR_VALUE).normal(size=5),
        rng.substream(Stream.FAIR_VALUE, 0).normal(size=5),
    )


# --------------------------------------------------------------------------
# Batch runs
# --------------------------------------------------------------------------


def test_spawn_run_is_reproducible_and_independent():
    master = RngStreams(2024)
    assert master.spawn_run(3).seed == RngStreams(2024).spawn_run(3).seed
    assert master.spawn_run(3).seed != master.spawn_run(4).seed

    draws = [
        tuple(master.spawn_run(i).stream(Stream.FAIR_VALUE).normal(size=3))
        for i in range(20)
    ]
    assert len(set(draws)) == 20


def test_spawn_run_child_is_standalone():
    """A child's seed is a plain int, so a batch run is reconstructible alone."""
    child = RngStreams(2024).spawn_run(7)
    np.testing.assert_array_equal(
        child.stream(Stream.AGENTS).normal(size=5),
        RngStreams(child.seed).stream(Stream.AGENTS).normal(size=5),
    )


# --------------------------------------------------------------------------
# Key hygiene
# --------------------------------------------------------------------------


def test_stream_keys_are_distinct_and_disjoint_from_run_key():
    keys = [stream_key(name) for name in Stream]
    assert len(set(keys)) == len(keys)
    assert _RUN_KEY not in keys


def test_typo_is_rejected_not_silently_created():
    """A misspelled name must raise, not mint a valid-but-wrong stream."""
    rng = RngStreams(1)
    with pytest.raises(ValueError):
        rng.stream("fair_valeu")
    with pytest.raises(ValueError):
        rng.substream("flow_sizes", 0)


@pytest.mark.parametrize("bad", [-1, "3", 3.0, True, None])
def test_bad_seed_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        RngStreams(bad)


def test_negative_substream_index_rejected():
    with pytest.raises(ValueError):
        RngStreams(1).substream(Stream.AGENTS, -1)


def test_no_reseed_and_seed_is_read_only():
    rng = RngStreams(1)
    assert not hasattr(rng, "reseed")
    with pytest.raises(AttributeError):
        rng.seed = 2


# --------------------------------------------------------------------------
# Provenance and numpy stream stability (NEP 19 froze RandomState, not Generator)
# --------------------------------------------------------------------------


def test_provenance_records_what_the_log_header_needs():
    header = RngStreams(42).provenance()
    assert header["scheme_version"] == SCHEME_VERSION
    assert header["seed"] == 42
    assert header["numpy_version"] == np.__version__
    assert header["bit_generator"] == "PCG64"
    assert header["canary"] == canary()


def test_canary_is_independent_of_the_session():
    """The canary tests numpy's transforms, not the session, so every header
    carries the same values and replay can compare without knowing the seed."""
    assert RngStreams(1).provenance()["canary"] == RngStreams(2).provenance()["canary"]
    assert RngStreams(2024).spawn_run(7).provenance()["canary"] == canary()
    assert canary() == canary()


def test_canary_covers_each_transform_separately():
    """One entry per transform, each from its own stream, so a mismatch names
    the transform that moved rather than shifting everything after it."""
    c = canary()
    assert set(c) == {"uniform", "normal", "exponential"}
    assert all(len(v) == 8 for v in c.values())
    assert all(isinstance(x, float) for v in c.values() for x in v)
    assert all(0.0 <= x < 1.0 for x in c["uniform"])
    assert all(x >= 0.0 for x in c["exponential"])


def test_provenance_does_not_advance_session_streams():
    """Writing the header must not consume session draws.

    If the canary were drawn from the session's own streams, a logged session
    would silently differ from an unlogged one by exactly the canary's draws --
    the header written to make replay safe would be what broke it. Checked
    both for streams first touched after the header and for one already open.
    """
    logged, unlogged = RngStreams(9), RngStreams(9)
    logged.stream(Stream.FAIR_VALUE).normal(size=3)
    unlogged.stream(Stream.FAIR_VALUE).normal(size=3)

    logged.provenance()

    for name in Stream:
        np.testing.assert_array_equal(
            logged.stream(name).normal(size=5), unlogged.stream(name).normal(size=5)
        )


def test_provenance_round_trips_through_json():
    """The header is written to a log and read back by a different process.

    A child from spawn_run has a 128-bit seed, and the canary is floats: both
    must survive json exactly, or the replay comparison fails for the wrong
    reason.
    """
    header = RngStreams(2024).spawn_run(3).provenance()
    assert json.loads(json.dumps(header)) == header


def test_golden_values_detect_numpy_stream_drift():
    """Hardcoded expectations, so a numpy upgrade that changes Generator output
    fails here loudly instead of silently altering every recorded session.

    If this fails after a numpy bump, the draws changed -- old logs no longer
    replay. Bump SCHEME_VERSION and re-baseline deliberately.
    """
    rng = RngStreams(42)
    np.testing.assert_allclose(
        rng.stream(Stream.FAIR_VALUE).normal(size=3),
        [1.4578419413331305, -0.8058221570277185, 0.8412311601006737],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        rng.stream(Stream.FLOW_ARRIVAL).normal(size=3),
        [-0.8832314315039385, -1.0517194890577055, -0.5949103489592373],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        RngStreams(42).substream(Stream.FLOW_SIZE, 7).normal(size=3),
        [1.3388990529050158, -2.2434228195027788, -0.7708430592872917],
        rtol=0,
        atol=0,
    )
    assert (
        RngStreams(2024).spawn_run(0).seed
        == 176984773780486007358012441277627477426
    )


def test_canary_golden_values():
    """Pins what every recorded header holds under numpy as locked today.

    Two ways this goes red, with different remedies:

    - After a numpy bump: the transforms moved, so every recorded log's canary
      will mismatch on replay. That is the header working. Bump SCHEME_VERSION
      and re-baseline these values deliberately.
    - After editing canary() itself (seed, streams, draw count): you changed
      what old headers are compared against, and every one of them becomes
      unreplayable. Same remedy, and make sure it was on purpose.
    """
    assert CANARY_SEED == 0
    assert canary() == {
        "uniform": [
            0.19832048442242656,
            0.9822061104066486,
            0.19702657623328468,
            0.46168153481564456,
            0.8875674742141777,
            0.33499009500010324,
            0.9503600523614734,
            0.7873596643943469,
        ],
        "normal": [
            -0.2690047809335765,
            -1.0766519465814615,
            0.2052198643135207,
            -0.7599372785895425,
            0.4766047261485515,
            1.3420144866823223,
            -0.19662665338487875,
            2.3260120848287684,
        ],
        "exponential": [
            0.8402710746863868,
            0.2508906614229817,
            0.00483297276775258,
            0.0285235328058251,
            0.540277865849249,
            4.312198958523308,
            0.39248528351981965,
            0.6549924093535895,
        ],
    }


# --------------------------------------------------------------------------
# Codebase hygiene (brief section 7)
# --------------------------------------------------------------------------


def test_no_ungoverned_randomness_in_src():
    """A single bare np.random call makes the replay guarantee a lie.

    Nothing else in the test suite would catch it, so grep for it here.
    """
    banned = re.compile(r"\bnp\.random\.|\brandom\.|^\s*import\s+random\b", re.MULTILINE)
    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        if path.name == "rng.py":
            continue  # the one module allowed to touch numpy's RNG API
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{path.relative_to(SRC_DIR.parent)}:{lineno}: {line.strip()}")

    assert not offenders, "ungoverned randomness found:\n" + "\n".join(offenders)


def test_rng_module_has_no_internal_imports():
    """rng sits at the bottom of the dependency graph -- keep it there."""
    source = (SRC_DIR / "rng.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+mm_game\b", source, re.MULTILINE)
