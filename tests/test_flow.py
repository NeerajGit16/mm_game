"""Tests for mm_game.flow.

The ones that matter most: alignment under a changed informed share (the
ordinal trap), size overlap measured rather than eyeballed, informed status
only inside windows from informed types with ~80% direction accuracy, one
view per informed client per print, and the invariance list. Each docstring
says what silent failure it catches.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game import fair_value, flow
from mm_game.config import GameConfig
from mm_game.flow import Category, PublicRFQ, Side
from mm_game.rng import RngStreams, Stream

SRC_DIR = Path(mm_game.__file__).resolve().parent

FLOW_STREAMS = {
    Stream.FLOW_ARRIVAL,
    Stream.FLOW_CLIENT,
    Stream.FLOW_CLIENT_TYPE,
    Stream.FLOW_SIZE,
    Stream.FLOW_DIRECTION,
    Stream.FLOW_INFORMED,
    Stream.FLOW_PARTICIPATION,
}
FAIR_VALUE_STREAMS = (Stream.FAIR_VALUE, Stream.EVENTS, Stream.JUMPS, Stream.DRIFT, Stream.SCREEN, Stream.SCREEN_PRINT)

# A dense short session for the statistical tests: ten minutes at six times the
# default rate, ~100 RFQs and ~10 informed per session, so 300 sessions give
# ~27,000 uninformed and ~3,000 informed RFQs in about two seconds. With a
# 600 s session every drift is shortened to start at 0, so windows cover the
# whole run-up to each print and two-print sessions always overlap.
STAT = GameConfig(
    session_length=600,
    event_min_time_s=60.0,
    event_end_margin_s=60.0,
    event_min_gap_s=120.0,
    rfq_rate_per_hour=600.0,
)


def session(cfg: GameConfig, seed: int) -> tuple[fair_value.FairValuePath, flow.FlowSchedule]:
    rng = RngStreams(seed)
    path = fair_value.generate(cfg, rng)
    return path, flow.generate(cfg, rng, path)


@pytest.fixture(scope="module")
def stat_sessions() -> list[tuple[fair_value.FairValuePath, flow.FlowSchedule]]:
    return [session(STAT, s) for s in range(300)]


def in_window(path: fair_value.FairValuePath, step: int) -> bool:
    return any(w.start_step <= step < w.end_step for w in path.drift_windows)


def size_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Overlapping coefficient of two size distributions: the sum over bins of
    the smaller of the two empirical probabilities. 1 means identical, 0
    disjoint. Sixteen log-spaced bins over 1..100 lots, so the tail is
    resolved and sampling noise in sparse bins stays small."""
    edges = np.unique(np.round(np.geomspace(1, 101, 16)))
    ha, _ = np.histogram(a, edges)
    hb, _ = np.histogram(b, edges)
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


# --------------------------------------------------------------------------
# The ones that matter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "low,high",
    [
        (
            dict(informed_prob_hedge_fund=0.50, informed_prob_real_money=0.15, informed_prob_corporate=0.05),
            dict(informed_prob_hedge_fund=0.90, informed_prob_real_money=0.60, informed_prob_corporate=0.40),
        ),
        (
            dict(informed_prob_hedge_fund=0.0, informed_prob_real_money=0.0, informed_prob_corporate=0.0),
            dict(informed_prob_hedge_fund=1.0, informed_prob_real_money=1.0, informed_prob_corporate=1.0),
        ),
    ],
)
def test_alignment_under_a_changed_informed_share(low, high):
    """The ordinal trap.

    Raise the informed share. Every RFQ keeps its ordinal, step, client and
    base size, and every RFQ that is uninformed in both worlds keeps its size,
    side and one-way flag bit for bit. Informed clients under the low share
    stay informed under the high one: raising a probability only adds.

    If informed flow were a separate arrival process merged by time, raising
    its rate would insert RFQs, renumber everything after, and re-roll the
    size, side and client of every uninformed inquiry -- "same uninformed
    flow, more informed flow" would silently compare two different markets.
    """
    checked_uninformed = moved_to_informed = 0
    for seed in range(6):
        p_low, f_low = session(GameConfig(**low), seed)
        p_high, f_high = session(GameConfig(**high), seed)
        assert len(f_low.rfqs) == len(f_high.rfqs)
        low_types = {c.client_id for c in f_low.clients if c.informed_type}
        high_types = {c.client_id for c in f_high.clients if c.informed_type}
        assert low_types <= high_types
        for a, b in zip(f_low.rfqs, f_high.rfqs):
            assert (a.ordinal, a.step, a.client_id, a.base_size) == (b.ordinal, b.step, b.client_id, b.base_size)
            if not a.informed and not b.informed:
                assert (a.size, a.side, a.one_way) == (b.size, b.side, b.one_way)
                checked_uninformed += 1
            if a.informed and b.informed:
                # A client informed in both worlds holds the same view.
                assert (a.signal_bp, a.side, a.size) == (b.signal_bp, b.side, b.size)
            assert not (a.informed and not b.informed)
            moved_to_informed += (not a.informed) and b.informed
    assert checked_uninformed > 500 and moved_to_informed > 10


def test_informed_sizes_are_larger_and_overlap(stat_sessions):
    """Informed tickets are bigger on average and the two
    distributions overlap ~70%, measured.

    Size is the one tell the player sees before quoting. If informed sizes
    sat above the uninformed range, size alone would settle it and the game
    would be trivial; if they matched, size would carry nothing. Expected
    overlap for two lognormals of log-sd 0.9 shifted by ln 2 is 70%; the
    factor's spread and the cap move it a little, and ~3,000 informed sizes
    put the estimate within about +/-0.02. The factor itself averages ~2x and
    stays inside [1, 3].
    """
    uninformed = np.array([q.size for _, f in stat_sessions for q in f.rfqs if not q.informed])
    informed = np.array([q.size for _, f in stat_sessions for q in f.rfqs if q.informed])
    factors = np.array(
        [flow._size_factor(STAT, q.signal_bp) for _, f in stat_sessions for q in f.rfqs if q.informed]
    )
    assert informed.size > 2000
    assert informed.mean() > 1.6 * uninformed.mean()
    assert 1.6 < np.median(informed) / np.median(uninformed) < 2.3
    overlap = size_overlap(uninformed, informed)
    assert 0.63 < overlap < 0.80, overlap
    assert 1.85 < factors.mean() < 2.15
    assert factors.min() >= 1.0 and factors.max() <= 3.0
    # A big uninformed ticket is a genuine possibility, so size never settles it.
    assert np.mean(uninformed >= 40) > 0.04


def test_informed_only_inside_windows_and_only_from_informed_types(stat_sessions):
    """Informed status is a property of (client
    type, timing): an informed-type client inside a drift window is informed,
    anyone else is not -- in both directions, so an informed RFQ outside a
    window or from an uninformed-type client fails, and so does an
    informed-type client in a window who was silently left uninformed. The
    side is the signal's sign and the print is the nearer upcoming one."""
    seen = 0
    for path, sched in stat_sessions:
        types = {c.client_id: c.informed_type for c in sched.clients}
        for q in sched.rfqs:
            expected = types[q.client_id] and in_window(path, q.step)
            assert q.informed == expected
            if q.informed:
                seen += 1
                assert q.side == (Side.BUY if q.signal_bp > 0 else Side.SELL)
                upcoming = np.flatnonzero(path.event_steps > q.step)
                assert q.event_index == int(upcoming[0])
            else:
                assert q.signal_bp is None and q.event_index is None
    assert seen > 2000


def informed_views(stat_sessions):
    """One (signal, true jump) per informed (client, print) pair, pooled."""
    views = []
    for path, sched in stat_sessions:
        seen = {}
        for q in sched.rfqs:
            if q.informed:
                seen.setdefault((q.client_id, q.event_index), (q.signal_bp, path.jump_sizes[q.event_index]))
        views.extend(seen.values())
    return views


def test_signal_direction_accuracy_is_q_and_magnitude_is_roughly_right(stat_sessions):
    """A client's view of a print has the right
    sign with probability 0.80. Counted per (client, print) pair, since that
    is the unit drawn; ~900 pairs give SE 0.013, so +/-0.055 is ~4 SE, and it
    separates 0.80 from the crowd's 0.60. Magnitude is the print's times a
    lognormal error: median ratio ~1, log-sd ~0.25, so they know roughly how
    much -- the property that makes size informative."""
    views = informed_views(stat_sessions)
    right = [np.sign(sig) == np.sign(jump) for sig, jump in views]
    log_ratio = [math.log(abs(sig) / abs(jump)) for sig, jump in views]
    assert len(views) > 700
    assert abs(np.mean(right) - 0.80) < 0.055
    assert abs(np.median(log_ratio)) < 0.04
    assert abs(np.std(log_ratio) - STAT.informed_signal_noise) < 0.03


def test_an_informed_client_holds_one_view_per_print(stat_sessions):
    """Every informed inquiry a client sends about one print carries the same
    signal, so the same side. Drawn per RFQ instead, a client re-drew its view
    on every inquiry: half of informed clients with several inquiries bought
    and sold the same print, which no informed client does and which erases
    the "this client keeps lifting me before the print" read. Different
    clients about the same print still form independent views."""
    pairs_with_repeats, same_print_disagreements = 0, 0
    for _, sched in stat_sessions:
        by_pair: dict[tuple[int, int], set[float]] = {}
        for q in sched.rfqs:
            if q.informed:
                by_pair.setdefault((q.client_id, q.event_index), set()).add(q.signal_bp)
        counts = {}
        for q in sched.rfqs:
            if q.informed:
                counts[(q.client_id, q.event_index)] = counts.get((q.client_id, q.event_index), 0) + 1
        for pair, signals in by_pair.items():
            assert len(signals) == 1
            pairs_with_repeats += counts[pair] >= 2
        by_print: dict[int, set[float]] = {}
        for (client, event), signals in by_pair.items():
            by_print.setdefault(event, set()).update(signals)
        same_print_disagreements += sum(len(v) >= 2 for v in by_print.values())
    assert pairs_with_repeats > 300 and same_print_disagreements > 100


def test_flow_never_draws_from_a_fair_value_stream():
    """Behaviourally: every stream flow requests is recorded,
    and only flow streams appear. A flow module drawing informed noise from
    FAIR_VALUE would make the price path depend on how many informed clients
    arrived -- the coupling independent streams exist to prevent. Statically
    too: every Stream member named in flow.py's source is a flow stream."""

    class Recording(RngStreams):
        __slots__ = ("used",)

        def __init__(self, seed: int) -> None:
            super().__init__(seed)
            self.used: set[Stream] = set()

        def stream(self, name):
            self.used.add(Stream(name))
            return super().stream(name)

        def substream(self, name, index):
            self.used.add(Stream(name))
            return super().substream(name, index)

    path = fair_value.generate(GameConfig(), RngStreams(3))
    assert path.event_steps.size > 0
    rec = Recording(3)
    sched = flow.generate(GameConfig(), rec, path)
    assert any(q.informed for q in sched.rfqs)
    assert rec.used == FLOW_STREAMS

    source = (SRC_DIR / "flow.py").read_text(encoding="utf-8")
    named = {Stream[m] for m in re.findall(r"Stream\.([A-Z_]+)", source)}
    assert named <= FLOW_STREAMS


# --------------------------------------------------------------------------
# Invariance requirements
# --------------------------------------------------------------------------


def test_size_model_never_moves_anything_but_size():
    """Change the base distribution, the cap and the informed scaling: step,
    client, side, one-way flag, informed status and signal are bit-identical."""
    other = GameConfig(
        size_median_lots=25.0,
        size_log_sd=0.4,
        size_cap_lots=50,
        informed_size_ref_bp=0.5,
        informed_size_factor_max=6.0,
    )
    for seed in range(5):
        _, a = session(GameConfig(), seed)
        _, b = session(other, seed)
        assert len(a.rfqs) == len(b.rfqs)
        for x, y in zip(a.rfqs, b.rfqs):
            assert (x.step, x.client_id, x.side, x.one_way, x.informed, x.signal_bp) == (
                y.step,
                y.client_id,
                y.side,
                y.one_way,
                y.informed,
                y.signal_bp,
            )
        assert any(x.size != y.size for x, y in zip(a.rfqs, b.rfqs))


def test_participation_never_moves_anything_else_and_only_adds_dealers():
    """Raise the participation probability: step, client, side, size,
    one-way flag, informed status and signal are bit-identical, and every
    dealer in competition at the lower probability is still in. One uniform
    per dealer compared against the probability is what guarantees it; a
    draw that picked "how many" first and then "which" would reshuffle the
    street whenever the probability moved."""
    low, high = GameConfig(competitor_participation_prob=0.4), GameConfig(competitor_participation_prob=0.7)
    added = 0
    for seed in range(5):
        _, a = session(low, seed)
        _, b = session(high, seed)
        for x, y in zip(a.rfqs, b.rfqs):
            assert (x.step, x.client_id, x.side, x.size, x.one_way, x.informed, x.signal_bp) == (
                y.step,
                y.client_id,
                y.side,
                y.size,
                y.one_way,
                y.informed,
                y.signal_bp,
            )
            assert all(hi or not lo for lo, hi in zip(x.dealers_in_comp, y.dealers_in_comp))
            added += y.in_comp - x.in_comp
    assert added > 100


def test_competition_count_is_binomial():
    """Each of three dealers in independently at 0.6: none 6.4%, one 28.8%,
    two 43.2%, all three 21.6%."""
    counts = np.bincount([q.in_comp for s in range(40) for q in session(GameConfig(), s)[1].rfqs], minlength=4)
    shares = counts / counts.sum()
    for k, expected in enumerate((0.064, 0.288, 0.432, 0.216)):
        assert abs(shares[k] - expected) < 0.015


def test_informed_participation_lever_is_built_and_inert_by_default(stat_sessions):
    """At the default multiplier of 1.0, every RFQ's dealers are exactly what
    a build without the lever draws: one uniform per dealer from the RFQ's own
    substream against the plain probability, informed or not. An
    implementation that drew differently for informed RFQs would move the
    street for exactly the inquiries that matter, with the lever "off". Turned
    down, it thins competition on informed RFQs only."""
    cfg = STAT
    rfqs = 0
    for seed in range(20):
        _, sched = session(cfg, seed)
        rng = RngStreams(seed)
        for q in sched.rfqs:
            u = rng.substream(Stream.FLOW_PARTICIPATION, q.ordinal).random(cfg.n_competitors)
            assert q.dealers_in_comp == tuple(bool(x < cfg.competitor_participation_prob) for x in u)
            rfqs += 1
    assert rfqs > 1000

    shy = dataclasses.replace(cfg, informed_participation_multiplier=0.3)
    fewer = 0
    for seed in range(20):
        _, a = session(cfg, seed)
        _, b = session(shy, seed)
        for x, y in zip(a.rfqs, b.rfqs):
            if x.informed:
                assert y.in_comp <= x.in_comp
                fewer += y.in_comp < x.in_comp
            else:
                assert x.dealers_in_comp == y.dealers_in_comp
    assert fewer > 20


def test_one_way_rates_never_move_a_side():
    """Side is drawn before the one-way flag from the same per-RFQ substream,
    so tuning one-way rates changes which RFQs show their side, never which
    side they are."""
    other = GameConfig(one_way_prob_uninformed=0.5, one_way_prob_informed=0.9)
    for seed in range(5):
        _, a = session(GameConfig(), seed)
        _, b = session(other, seed)
        assert [q.side for q in a.rfqs] == [q.side for q in b.rfqs]
        assert sum(q.one_way for q in b.rfqs) > sum(q.one_way for q in a.rfqs)


def test_signal_noise_never_flips_a_direction():
    """Direction is drawn before the magnitude error, so making the client's
    magnitude estimate noisier never changes which way they trade."""
    other = GameConfig(informed_signal_noise=1.0)
    for seed in range(8):
        _, a = session(GameConfig(), seed)
        _, b = session(other, seed)
        assert [q.side for q in a.rfqs] == [q.side for q in b.rfqs]
        assert [np.sign(q.signal_bp) if q.informed else 0 for q in a.rfqs] == [
            np.sign(q.signal_bp) if q.informed else 0 for q in b.rfqs
        ]


@pytest.mark.parametrize(
    "change",
    [
        dict(size_median_lots=30.0),
        dict(informed_prob_hedge_fund=0.9),
        dict(one_way_prob_uninformed=0.4),
        dict(crowd_drift_tilt=0.9),
        dict(informed_accuracy=0.55),
        dict(informed_activity_multiplier=4.0),
        dict(n_hedge_funds=2, n_real_money=10, n_corporates=8),
        dict(event_count_probs=(1.0,)),
        dict(drift_magnitude_min_bp=3.0, drift_magnitude_max_bp=5.0),
        dict(screen_noise_bp=2.0),
    ],
)
def test_only_the_rate_moves_arrivals(change):
    """Arrival steps depend on the rate and the grid and nothing else: not the
    roster, the informed share, the size model, the print calendar or the
    screen. The rate itself is allowed to change everything."""
    for seed in range(3):
        _, a = session(GameConfig(), seed)
        _, b = session(GameConfig(**change), seed)
        assert [q.step for q in a.rfqs] == [q.step for q in b.rfqs]
    _, faster = session(GameConfig(rfq_rate_per_hour=150.0), 0)
    _, base = session(GameConfig(), 0)
    assert [q.step for q in faster.rfqs] != [q.step for q in base.rfqs]


def test_independent_of_fair_value_draws_beyond_the_path():
    """Consuming draws from every fair-value stream before
    generating the schedule changes nothing: the path is an input, not a
    shared stream."""
    cfg = GameConfig()
    rng = RngStreams(11)
    path = fair_value.generate(cfg, rng)
    quiet = flow.generate(cfg, rng, path)
    for name in FAIR_VALUE_STREAMS:
        rng.stream(name).random(10_000)
        for i in range(50):
            rng.substream(name, i).random(10)
    assert flow.generate(cfg, rng, path) == quiet


# --------------------------------------------------------------------------
# Arrivals, determinism, shape
# --------------------------------------------------------------------------


def test_deterministic_and_pure():
    """Same (config, seed, path), same schedule; and generating twice on one
    RngStreams gives the same schedule, because every draw is a substream. A
    memoised stream() would advance between calls, and a benchmark rerun on
    the same world would silently face different RFQs."""
    cfg = GameConfig()
    _, a = session(cfg, 2024)
    _, b = session(cfg, 2024)
    assert a == b
    rng = RngStreams(2024)
    path = fair_value.generate(cfg, rng)
    assert flow.generate(cfg, rng, path) == flow.generate(cfg, rng, path) == a
    _, c = session(cfg, 2025)
    assert a != c


def test_arrival_count_matches_the_rate():
    """~200 RFQs per two-hour session. Over 40 sessions the
    mean's SE is ~2.2, so +/-10 is ~4.5 SE; the variance-to-mean ratio of a
    Poisson count is 1, and clustering would push it well above. RFQs are
    sorted, on distinct steps, ordinal equals position, and none is on the
    closing step."""
    cfg = GameConfig()
    counts = []
    for seed in range(40):
        _, sched = session(cfg, seed)
        steps = np.array([q.step for q in sched.rfqs])
        assert np.all(np.diff(steps) > 0)
        assert steps.min() >= 0 and steps.max() < cfg.n_steps
        assert [q.ordinal for q in sched.rfqs] == list(range(len(sched.rfqs)))
        counts.append(len(sched.rfqs))
    expected = cfg.rfq_rate_per_hour * cfg.session_length / 3600
    assert abs(np.mean(counts) - expected) < 10
    assert 0.5 < np.var(counts, ddof=1) / np.mean(counts) < 1.6


def test_rejects_a_path_from_a_different_grid():
    """A path generated under another config would index the wrong steps
    without raising. Refuse instead."""
    path = fair_value.generate(STAT, RngStreams(1))
    with pytest.raises(ValueError, match="does not match"):
        flow.generate(GameConfig(), RngStreams(1), path)


def test_flow_takes_no_player_input():
    """Flow never responds to the player, so paired replay stays clean.
    Structural: the generator's only inputs are config, rng and the path."""
    assert list(inspect.signature(flow.generate).parameters) == ["config", "rng", "path"]


# --------------------------------------------------------------------------
# Roster and types
# --------------------------------------------------------------------------


@pytest.mark.parametrize("minutes,clients", [(30, 6), (60, 10), (120, 16)])
def test_presets_set_the_roster_explicitly(minutes, clients):
    cfg = GameConfig.for_session(minutes)
    _, sched = session(cfg, 1)
    assert len(sched.clients) == clients
    cats = [c.category for c in sched.clients]
    assert cats.count(Category.HEDGE_FUND) == cfg.n_hedge_funds
    assert cats.count(Category.REAL_MONEY) == cfg.n_real_money
    assert cats.count(Category.CORPORATE) == cfg.n_corporates
    assert {q.client_id for q in sched.rfqs} <= set(range(clients))


def test_category_type_probabilities_and_redrawn_types():
    """Types are redrawn every session, at the category's probability. Over
    500 rosters: P(informed | category) within ~4 SE of 0.50 / 0.15 / 0.05,
    and every client is informed in some sessions and not in others -- a
    client informed every session would be memorised in a few games."""
    cfg = GameConfig()
    by_cat = {c: [] for c in Category}
    per_client = np.zeros(cfg.n_clients)
    for seed in range(500):
        for c in flow._draw_roster(cfg, RngStreams(seed)):
            by_cat[c.category].append(c.informed_type)
            per_client[c.client_id] += c.informed_type
    assert abs(np.mean(by_cat[Category.HEDGE_FUND]) - 0.50) < 0.04
    assert abs(np.mean(by_cat[Category.REAL_MONEY]) - 0.15) < 0.03
    assert abs(np.mean(by_cat[Category.CORPORATE]) - 0.05) < 0.02
    assert np.all(per_client > 0) and np.all(per_client < 500)


def test_informed_count_is_random_and_sometimes_zero():
    """The number of informed clients is not fixed: a known count would let
    the player stop suspecting everyone else once they had caught that many.
    In the 6-client preset P(no informed client) = 0.5^2 * 0.85^3 * 0.95 =
    0.146; over 2,000 rosters SE 0.008."""
    cfg = GameConfig.for_session(30)
    counts = np.array(
        [sum(c.informed_type for c in flow._draw_roster(cfg, RngStreams(s))) for s in range(2000)]
    )
    assert abs(np.mean(counts == 0) - 0.146) < 0.035
    assert len(set(counts)) >= 4


def test_raising_a_probability_only_adds_informed_clients():
    """One uniform per client, keyed on client id: raising a category's
    probability can turn an uninformed client informed, never the reverse,
    and never touches another category."""
    for seed in range(50):
        low = flow._draw_roster(GameConfig(informed_prob_real_money=0.15), RngStreams(seed))
        high = flow._draw_roster(GameConfig(informed_prob_real_money=0.60), RngStreams(seed))
        for a, b in zip(low, high):
            assert b.informed_type or not a.informed_type
            if a.category != Category.REAL_MONEY:
                assert a.informed_type == b.informed_type


def test_activity_multiplier_is_built_and_inert_by_default(stat_sessions):
    """The mechanism exists and works; at 1.0 it does nothing. At 3.0,
    informed-type clients take a visibly larger share of in-window RFQs, and
    every out-of-window RFQ keeps its client. Building it inert now is what
    makes enabling it later a parameter change rather than a structural one."""
    louder = GameConfig(**{**dataclasses.asdict(STAT), "informed_activity_multiplier": 3.0})
    share_default, share_louder, out_checked = [], [], 0
    for seed, (path, base) in enumerate(stat_sessions[:150]):
        _, loud = session(louder, seed)
        types = {c.client_id: c.informed_type for c in base.clients}
        if not any(types.values()):
            continue
        for a, b in zip(base.rfqs, loud.rfqs):
            if in_window(path, a.step):
                share_default.append(types[a.client_id])
                share_louder.append(types[b.client_id])
            else:
                assert a.client_id == b.client_id
                out_checked += 1
    assert out_checked > 500
    assert np.mean(share_louder) > np.mean(share_default) + 0.15


# --------------------------------------------------------------------------
# Size and direction
# --------------------------------------------------------------------------


def test_uninformed_size_distribution(stat_sessions):
    """Lognormal base with median 10 and log-sd 0.9; sizes are whole lots in
    [1, 100]. The cap binds occasionally, which is its job."""
    rfqs = [q for _, f in stat_sessions for q in f.rfqs]
    base = np.log([q.base_size for q in rfqs])
    sizes = np.array([q.size for q in rfqs])
    assert abs(np.exp(np.median(base)) - 10.0) < 0.5
    assert abs(base.std() - 0.9) < 0.02
    assert sizes.dtype.kind == "i" and sizes.min() >= 1 and sizes.max() <= 100
    assert np.any(sizes == 100)
    uninformed = [q for q in rfqs if not q.informed]
    assert all(q.size == max(1, min(100, round(q.base_size))) for q in uninformed)


def test_uninformed_side_is_a_coin_outside_windows_and_tilted_inside(stat_sessions):
    """Fair coin outside drift windows; 60/40 toward the net drift inside. A
    50/50 crowd under a drifting screen would make screen and flow incoherent;
    a tilt that followed the hidden print instead of the visible drift would
    leak the print. Thousands of RFQs on each side: SE <= 0.006, bands 0.03."""
    outside_buy, toward = [], []
    for path, sched in stat_sessions:
        rate = np.diff(path.drift)
        for q in sched.rfqs:
            if q.informed:
                continue
            if rate[q.step] == 0:
                outside_buy.append(q.side == Side.BUY)
            else:
                toward.append(int(q.side) == np.sign(rate[q.step]))
    assert len(outside_buy) > 3000 and len(toward) > 3000
    assert abs(np.mean(outside_buy) - 0.5) < 0.03
    assert abs(np.mean(toward) - 0.60) < 0.03


def test_one_way_rates_by_population(stat_sessions):
    """5% of uninformed and 10% of informed RFQs are one-way: a one-way
    inquiry is evidence, never proof."""
    uninf = [q.one_way for _, f in stat_sessions for q in f.rfqs if not q.informed]
    inf = [q.one_way for _, f in stat_sessions for q in f.rfqs if q.informed]
    assert abs(np.mean(uninf) - 0.05) < 0.01
    assert abs(np.mean(inf) - 0.10) < 0.025


# --------------------------------------------------------------------------
# Public view and records
# --------------------------------------------------------------------------


def test_public_view_cannot_reach_hidden_fields():
    """The public RFQ carries ordinal, step, client, size, the
    one-way flag, the side only when one-way, and how many dealers are in
    competition. Informed status, signal, the print it concerns, base size,
    a two-way side and which dealers were asked are structurally absent, not
    merely ignored."""
    _, sched = session(GameConfig(), 5)
    assert {f.name for f in dataclasses.fields(PublicRFQ)} == {
        "ordinal",
        "step",
        "client_id",
        "size",
        "one_way",
        "side",
        "in_comp",
    }
    assert {f.name for f in dataclasses.fields(flow.PublicClient)} == {"client_id", "category"}
    pub = sched.public
    for q, p in zip(sched.rfqs, pub.rfqs):
        assert p.side == (q.side if q.one_way else None)
        assert p.in_comp == sum(q.dealers_in_comp)
        for hidden in ("informed", "signal_bp", "event_index", "base_size", "dealers_in_comp"):
            assert not hasattr(p, hidden)
    assert any(p.side is None for p in pub.rfqs) and any(p.side is not None for p in pub.rfqs)
    record = json.dumps(pub.to_record())
    for hidden in ("informed", "signal", "event_index", "base_size", "dealers_in_comp"):
        assert hidden not in record


def test_records_round_trip_json():
    _, sched = session(GameConfig(), 7)
    for rec in (sched.to_record(), sched.public.to_record()):
        assert json.loads(json.dumps(rec)) == rec


def test_every_preset_generates():
    for minutes in (30, 60, 120):
        cfg = GameConfig.for_session(minutes)
        path, sched = session(cfg, 3)
        assert path.n_steps == cfg.n_steps
        assert all(q.step < cfg.n_steps for q in sched.rfqs)


def test_module_imports_only_rng_config_fair_value_and_numpy():
    source = (SRC_DIR / "flow.py").read_text(encoding="utf-8")
    internal = set(re.findall(r"^\s*from\s+mm_game\.(\w+)\s+import", source, re.MULTILINE))
    assert internal == {"rng", "config", "fair_value"}
    assert not re.search(r"^\s*import\s+mm_game", source, re.MULTILINE)
