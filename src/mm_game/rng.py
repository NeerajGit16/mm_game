"""Independent, name-keyed random streams.

A session must be reconstructible from ``(config, seed, event_log)``. Threading
one generator through every module gives you that, but only until you change
something: add a draw inside the fair value process and every subsequent draw in
the whole simulation shifts by one, so the market path, the flow and the fills
all move. You can no longer ask "did my new quoting rule do better?" because you
did not hold the market fixed.

The property that buys controlled counterfactuals is *stream independence*:
consuming a variable number of draws from one stream has no effect on any other.
Streams here are keyed by name, not by spawn position, so adding a stream later
leaves every existing stream's values untouched.

This module imports nothing from ``mm_game`` and must stay that way -- it sits at
the bottom of the dependency graph.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

import numpy as np

# Bump whenever the name -> state derivation changes (the key function, the
# spawn_key shape, or the bit generator). Written into the event log header so
# an old log that can no longer replay says so loudly instead of quietly
# producing different numbers.
SCHEME_VERSION: int = 1

_BIT_GENERATOR: str = "PCG64"

# Width of the blake2b digest used to key a stream, in bytes.
_KEY_BYTES: int = 8


class Stream(StrEnum):
    """The streams a session draws from.

    Split along the axis you want to hold fixed in a counterfactual, not along
    module boundaries. ``FLOW_INFORMED`` is separate from ``FLOW_DIRECTION`` so
    the informed fraction can be swept from 0 to 1 on an otherwise identical
    session; ``AGENTS`` is separate from everything so "strategy A vs strategy B
    on the same market" is a thing you can actually run.

    ``scoring``, ``book`` and ``events`` consume no randomness and have no
    stream.
    """

    FAIR_VALUE = "fair_value"
    FLOW_ARRIVAL = "flow_arrival"
    FLOW_SIZE = "flow_size"
    FLOW_DIRECTION = "flow_direction"
    FLOW_INFORMED = "flow_informed"
    COMPETITION = "competition"
    AGENTS = "agents"


def _digest_key(name: str) -> int:
    """Hash a name to a stable integer.

    Deliberately not ``hash()``: ``PYTHONHASHSEED`` is randomised by default, so
    the builtin hash of a str differs between processes. Using it would give a
    simulation that replays correctly within one process and silently diverges
    between them.
    """
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=_KEY_BYTES).digest(),
        "big",
    )


def stream_key(name: Stream) -> int:
    """Stable integer key for a stream, depending only on its name."""
    return _digest_key(Stream(name).value)


# Reserved key for run derivation in spawn_run. Namespaced away from the stream
# names so a run can never alias a stream; test_rng asserts the separation.
_RUN_KEY: int = _digest_key("__run__")

# uint32 words drawn to build a child seed in spawn_run -> a 128-bit seed.
_RUN_SEED_WORDS: int = 4

# Canary: fixed draws at a fixed seed, recorded in every log header by
# provenance(). numpy froze RandomState, not Generator (NEP 19): the PCG64 bit
# stream is stable, but the transform from bits to a distribution -- the
# ziggurat behind normal() and exponential() -- may change in a minor release.
# The version string only says numpy differs; the canary says whether the
# numbers did. Replay regenerates canary() and compares it with the header: a
# match means the log replays whatever the version strings say, a mismatch
# names the transform that moved.
#
# The keys below are deliberately NOT Stream members. The stream inventory is
# expected to grow and be renamed as the model develops, and a stream's key is
# derived from its name -- so routing the canary through Stream would make an
# ordinary rename change the canary's values, and every previously recorded
# header would then mismatch and report numpy drift when nothing drifted. These
# three names are frozen for the life of the scheme; the Stream enum is free to
# change around them. Each entry gets its own key so one transform changing
# does not shift the others.
#
# The canary is part of the scheme: changing its seed, these keys or its draw
# count invalidates every recorded header, so it bumps SCHEME_VERSION.
CANARY_SEED: int = 0
_CANARY_DRAWS: int = 8
_CANARY_KEYS: dict[str, int] = {
    "uniform": _digest_key("__canary_uniform__"),
    "normal": _digest_key("__canary_normal__"),
    "exponential": _digest_key("__canary_exponential__"),
}


class RngStreams:
    """Independent random streams derived from one session seed.

    Constructed once, by the engine, at session start. There is no ``reseed()``
    and no mutation after construction: a session's randomness is fixed the
    moment the object exists.
    """

    __slots__ = ("_seed", "_cache")

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._seed = seed
        self._cache: dict[Stream, np.random.Generator] = {}

    def __repr__(self) -> str:
        return f"RngStreams(seed={self._seed})"

    @property
    def seed(self) -> int:
        """The session seed, for the event log header."""
        return self._seed

    def _seed_sequence(self, spawn_key: tuple[int, ...]) -> np.random.SeedSequence:
        return np.random.SeedSequence(entropy=self._seed, spawn_key=spawn_key)

    def stream(self, name: Stream) -> np.random.Generator:
        """Return the generator for ``name``, memoized.

        Repeated calls return the *same* ``Generator`` object at its current
        state. Returning a fresh generator each call would silently restart from
        the same state every time, so every draw would return the same number.
        """
        # Coerces a bare str and rejects a typo. Without this a misspelled name
        # would derive a perfectly valid, perfectly independent stream that is
        # not the one you meant.
        key = Stream(name)
        generator = self._cache.get(key)
        if generator is None:
            generator = np.random.default_rng(self._seed_sequence((stream_key(key),)))
            self._cache[key] = generator
        return generator

    def substream(self, name: Stream, index: int) -> np.random.Generator:
        """Return a fresh generator keyed on ``(seed, name, index)``.

        Isolates per-event draws, so changing how many draws one event consumes
        does not shift any other event. Not memoized -- ``index`` is unbounded.
        Costs a ``Generator`` construction per call, which is fine per RFQ and
        unacceptable per simulation tick.
        """
        key = Stream(name)
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an int, got {type(index).__name__}")
        index = int(index)
        if index < 0:
            raise ValueError(f"index must be non-negative, got {index}")
        return np.random.default_rng(self._seed_sequence((stream_key(key), index)))

    def spawn_run(self, run_index: int) -> RngStreams:
        """Derive an independent session from this one, for batch runs.

        The child holds a plain integer seed, so it is loggable on its own and
        reconstructible standalone as ``RngStreams(child.seed)``.
        """
        if isinstance(run_index, bool) or not isinstance(run_index, (int, np.integer)):
            raise TypeError(
                f"run_index must be an int, got {type(run_index).__name__}"
            )
        run_index = int(run_index)
        if run_index < 0:
            raise ValueError(f"run_index must be non-negative, got {run_index}")
        words = self._seed_sequence((_RUN_KEY, run_index)).generate_state(
            _RUN_SEED_WORDS, dtype=np.uint32
        )
        return RngStreams(int.from_bytes(words.tobytes(), "little"))

    def provenance(self) -> dict[str, object]:
        """Everything the event log header needs to detect an unreplayable log.

        numpy froze ``RandomState``, not ``Generator`` (NEP 19), so a numpy
        upgrade can change the draws under an identical state. The version
        makes that noticeable; the canary makes it *decidable* -- see
        ``canary()``. Everything here is plain Python scalars and lists, so
        the header round-trips through JSON exactly.
        """
        return {
            "scheme_version": SCHEME_VERSION,
            "seed": self._seed,
            "numpy_version": np.__version__,
            "bit_generator": _BIT_GENERATOR,
            "canary": canary(),
        }


def _canary_generator(entry: str) -> np.random.Generator:
    """Generator for one canary entry, derived independently of ``Stream``."""
    return np.random.default_rng(
        np.random.SeedSequence(entropy=CANARY_SEED, spawn_key=(_CANARY_KEYS[entry],))
    )


def canary() -> dict[str, list[float]]:
    """The canary draws recorded in every log header; see ``CANARY_SEED``.

    Independent of any session and of the ``Stream`` inventory: derived from
    ``_CANARY_KEYS``, which are frozen, so renaming or adding a stream leaves
    these values untouched and old headers keep verifying. Calling it never
    advances a caller's streams.

    Three entries, one per transform the simulation depends on, each from its
    own key so a mismatch localises: ``uniform`` moving means the bit stream
    itself changed and everything else will have too; ``normal`` or
    ``exponential`` alone means that transform's algorithm did.
    """
    return {
        "uniform": _canary_generator("uniform").random(size=_CANARY_DRAWS).tolist(),
        "normal": _canary_generator("normal").normal(size=_CANARY_DRAWS).tolist(),
        "exponential": _canary_generator("exponential")
        .exponential(size=_CANARY_DRAWS)
        .tolist(),
    }
