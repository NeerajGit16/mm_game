"""Benchmark players, and the loop that drives any strategy through a session.

Three bots run after every session on the identical pre-generated world --
same RFQs, same competitors, same client limits -- so their results are a
paired comparison, not a statistical one:

- **Naive**: a fixed width around the screen; never hedges. The floor.
- **Disciplined**: public information only, including how many dealers are in
  competition. Widens with size, into prints and when competition is thin;
  skews against inventory; hedges toward a modest carry before each print;
  flattens and stops before the close. Good habits, no flow reading.
- **Flow oracle**: the disciplined bot plus knowledge of which RFQs are
  informed -- and nothing else: not the truth, not the print outcomes. An
  oracle that knew the print would trade it through the venue, a useless
  bound for market making. Knowing who is informed isolates exactly the skill
  the game teaches.

Bots play through the same protocol as the player and see only what its
messages say; the oracle's one extra input is handed to it explicitly. Their
parameters are fixed in ``BenchmarkConfig`` and never tuned to a seed.

Everything here is deterministic: a bot needs no randomness, and drawing any
would need a stream of its own.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from mm_game import protocol
from mm_game.config import BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings
from mm_game.world import World

V = protocol.PROTOCOL_VERSION


def advance(to_step: int) -> dict[str, Any]:
    return {"v": V, "type": "advance", "to_step": to_step}


def quote(ordinal: int, bid: float | None, ask: float | None) -> dict[str, Any]:
    return {"v": V, "type": "quote", "rfq_ordinal": ordinal, "bid": bid, "ask": ask}


def pass_(ordinal: int) -> dict[str, Any]:
    return {"v": V, "type": "pass", "rfq_ordinal": ordinal}


def hedge(quantity: int) -> dict[str, Any]:
    return {"v": V, "type": "hedge", "quantity": quantity}


class Strategy:
    """A player that speaks the protocol. Override what you need.

    A strategy names, at the start, the steps it wants the clock stopped at
    (``wakes``) -- all from public information, the print calendar and the
    session length. The driver stops there unless an RFQ stops it first, and
    calls ``on_wake`` once per wake, even when an RFQ landed on the same step.
    """

    def on_start(self, start: dict[str, Any], state: dict[str, Any]) -> None:
        self.n_steps = round(start["config"]["session_length"] / start["config"]["dt"])
        self.print_steps = list(start["print_steps"])
        self._wakes = sorted({w for w in self.wakes() if 0 < w < self.n_steps})

    def wakes(self) -> list[int]:
        return []

    def next_wake(self) -> int | None:
        return self._wakes[0] if self._wakes else None

    def pop_wake(self) -> int:
        return self._wakes.pop(0)

    def observe(self, message: dict[str, Any]) -> None:
        """Every outbound message, in order."""

    def answer(self, ticket: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def after_answer(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return []

    def on_wake(self, state: dict[str, Any], wake: int) -> list[dict[str, Any]]:
        return []


def play(session: Session, strategy: Strategy) -> None:
    """Drive ``session`` to its end with ``strategy``. A refused action is a
    bug in the strategy and raises."""
    state: dict[str, Any] = {}

    def send(messages: Iterable[dict[str, Any]]) -> None:
        nonlocal state
        for m in messages:
            strategy.observe(m)
            if m["type"] == "error":
                raise RuntimeError(f"strategy action refused: {m['detail']}")
            if m["type"] == "state":
                state = m

    def act(actions: Iterable[dict[str, Any]]) -> None:
        for action in actions:
            if session.ended:
                return
            send(session.handle(action))

    first = session.start()
    send(first)
    start = next(m for m in first if m["type"] == "event" and m["kind"] == "session_start")
    strategy.on_start(start["data"], state)
    while not session.ended:
        if state["ticket"] is not None:
            send(session.handle(strategy.answer(state["ticket"], state)))
            act(strategy.after_answer(state))
            continue
        wake = strategy.next_wake()
        if wake is not None and wake <= state["step"]:
            strategy.pop_wake()
            act(strategy.on_wake(state, wake))
            continue
        send(session.handle(advance(strategy.n_steps if wake is None else wake)))


# --------------------------------------------------------------------------
# The bots
# --------------------------------------------------------------------------


def _on_grid(price: float, config: GameConfig) -> float:
    """``price`` rounded to the nearest tick. Nearest rather than away from
    the mid: rounding outward would add up to a tick to every width, and a
    small size adjustment would silently cost a whole tick of hit rate."""
    k = round((price - config.start_level) / config.tick_size)
    return config.start_level + k * config.tick_size


def _two_way(config: GameConfig, ordinal: int, mid: float, half_width: float) -> dict[str, Any]:
    bid = _on_grid(mid - half_width, config)
    ask = _on_grid(mid + half_width, config)
    if not bid < ask:
        ask = bid + config.tick_size
    return quote(ordinal, bid, ask)


class Naive(Strategy):
    """Screen plus and minus a fixed width. Never hedges."""

    def __init__(self, config: GameConfig, benchmark: BenchmarkConfig) -> None:
        self.config = config
        self.half_width = benchmark.naive_half_width_bp

    def answer(self, ticket, state):
        return _two_way(self.config, ticket["ordinal"], state["screen"]["mid"], self.half_width)


class Disciplined(Strategy):
    """Good habits on public information only. See the module docstring."""

    def __init__(self, config: GameConfig, benchmark: BenchmarkConfig) -> None:
        self.config = config
        self.b = benchmark
        self.widths = benchmark.disciplined_half_width_bp
        if len(self.widths) < config.n_competitors + 1:
            raise ValueError("disciplined_half_width_bp needs one width per possible count in competition")

    def on_start(self, start, state):
        self.n_steps = round(start["config"]["session_length"] / start["config"]["dt"])
        self.print_steps = list(start["print_steps"])
        dt = self.config.dt
        self.close_wake = self.n_steps - round(self.b.disciplined_close_lead_s / dt)
        lead = round(self.b.disciplined_hedge_lead_s / dt)
        self.print_wakes = {e - lead for e in self.print_steps if e - lead > 0}
        super().on_start(start, state)

    def wakes(self):
        return sorted(self.print_wakes | {self.close_wake})

    def half_width(self, ticket, state) -> float:
        hw = self.widths[ticket["in_comp"]] + self.b.disciplined_size_widen_bp * ticket["size"] / 100.0
        step = state["step"]
        dt = self.config.dt
        ahead = [(e - step) * dt for e in self.print_steps if e > step]
        if ahead and ahead[0] <= self.b.disciplined_pre_print_window_s:
            hw += self.pre_print_widen()
        if any(abs(e - step) * dt <= self.b.disciplined_print_window_s for e in self.print_steps):
            hw += self.b.disciplined_print_widen_bp
        return hw

    def pre_print_widen(self) -> float:
        return self.b.disciplined_pre_print_widen_bp

    def skip(self, ticket, state) -> bool:
        return state["step"] >= self.close_wake

    def answer(self, ticket, state):
        if self.skip(ticket, state):
            return pass_(ticket["ordinal"])
        position = state["book"]["position"]
        mid = state["screen"]["mid"] - self.b.disciplined_skew_bp_per_100_lots * position / 100.0
        return _two_way(self.config, ticket["ordinal"], mid, self.half_width(ticket, state))

    def after_answer(self, state):
        position = state["book"]["position"]
        cap, target = self.b.disciplined_inventory_cap_lots, self.b.disciplined_inventory_target_lots
        if abs(position) > cap:
            return [hedge(int(math.copysign(target, position)) - position)]
        return []

    def on_wake(self, state, wake):
        position = state["book"]["position"]
        if wake == self.close_wake:
            return [hedge(-position)] if position else []
        carry = self.b.disciplined_print_carry_lots
        if wake in self.print_wakes and abs(position) > carry:
            return [hedge(int(math.copysign(carry, position)) - position)]
        return []


class FlowOracle(Disciplined):
    """The disciplined bot, told which RFQs are informed. It passes them. It
    is not told which way they trade, the truth, or any print outcome.

    It keeps every disciplined habit except the one that only stands in for
    what it already knows: the widening in the run-up to a print exists to
    price clients the bot cannot tell apart, so an oracle that can tell them
    apart quotes everyone else normally. The widening right around a print --
    print risk, not information -- stays."""

    def __init__(self, config: GameConfig, benchmark: BenchmarkConfig, informed: frozenset[int]) -> None:
        super().__init__(config, benchmark)
        self.informed = informed

    def skip(self, ticket, state):
        return ticket["ordinal"] in self.informed or super().skip(ticket, state)

    def pre_print_widen(self) -> float:
        return 0.0


class ScriptedPlayer(Disciplined):
    """A stand-in for a human in headless runs: disciplined quoting plus the
    things a person does -- runs fast through quiet stretches and slow into
    prints, pauses once, hedges by hand, can let the pricing timer run out,
    and can quit early. Exercises every input the protocol has."""

    def __init__(
        self,
        config: GameConfig,
        benchmark: BenchmarkConfig,
        *,
        quit_at_step: int | None = None,
        let_timer_expire_every: int | None = None,
    ) -> None:
        super().__init__(config, benchmark)
        self.quit_at_step = quit_at_step
        self.timer_every = let_timer_expire_every
        self.answered = 0
        self.manual_hedge_done = False

    def on_start(self, start, state):
        n = round(start["config"]["session_length"] / start["config"]["dt"])
        dt = self.config.dt
        self.slow_wakes = {max(1, e - round(600 / dt)) for e in start["print_steps"]}
        self.fast_wakes = {e + round(60 / dt) for e in start["print_steps"] if e + round(60 / dt) < n}
        self.pause_wake = max(1, n // 3)
        self.pending_start: list[dict[str, Any]] = [{"v": V, "type": "set_speed", "speed": 4}]
        super().on_start(start, state)

    def wakes(self):
        extra = self.slow_wakes | self.fast_wakes | {self.pause_wake}
        if self.quit_at_step is not None:
            extra.add(self.quit_at_step)
        return sorted(set(super().wakes()) | extra)

    def answer(self, ticket, state):
        self.answered += 1
        if self.timer_every and self.answered % self.timer_every == 0:
            return {"v": V, "type": "timer_expired", "rfq_ordinal": ticket["ordinal"]}
        return super().answer(ticket, state)

    def after_answer(self, state):
        out = list(self.pending_start)
        self.pending_start = []
        if not self.manual_hedge_done and state["book"]["position"] != 0 and self.answered > 20:
            self.manual_hedge_done = True
            out.append(hedge(-state["book"]["position"] // 2 or -state["book"]["position"]))
            return out
        return out + super().after_answer(state)

    def on_wake(self, state, wake):
        if wake == self.quit_at_step:
            return [{"v": V, "type": "quit"}]
        out: list[dict[str, Any]] = list(self.pending_start)
        self.pending_start = []
        if wake in self.slow_wakes and state["clock"]["speed"] != 1:
            out.append({"v": V, "type": "set_speed", "speed": 1})
        if wake in self.fast_wakes:
            out.append({"v": V, "type": "set_speed", "speed": 4})
        if wake == self.pause_wake:
            out += [{"v": V, "type": "pause"}, {"v": V, "type": "resume"}]
        return out + super().on_wake(state, wake)


# --------------------------------------------------------------------------
# Running the benchmarks
# --------------------------------------------------------------------------

BOTS = ("naive", "disciplined", "oracle")


def make_bot(name: str, world: World, benchmark: BenchmarkConfig) -> Strategy:
    if name == "naive":
        return Naive(world.config, benchmark)
    if name == "disciplined":
        return Disciplined(world.config, benchmark)
    if name == "oracle":
        return FlowOracle(world.config, benchmark, world.informed_ordinals())
    raise ValueError(f"unknown bot {name!r}")


def run_bot(name: str, world: World, *, settings: Settings, scoring: ScoringConfig, benchmark: BenchmarkConfig) -> Session:
    """Play one bot on ``world`` -- the identical pre-generated world the
    player faced -- and return the finished session."""
    session = Session(world.public, world.resolve, settings=settings, scoring=scoring)
    play(session, make_bot(name, world, benchmark))
    return session
