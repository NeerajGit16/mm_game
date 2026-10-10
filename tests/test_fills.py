"""Tests for mm_game.fills.

The ones that matter most: pre-generation independence (nothing about the
market moves with what you quote), paired replay (change one quote and only
that RFQ, and limit blocks it causes, can differ), cover that cannot be
averaged down to the truth (decay measured against an independent-error
control), and a winner's-curse fingerprint that exists and is faint. Each
docstring says what silent failure it catches.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game import fair_value, fills, flow
from mm_game.config import GameConfig
from mm_game.fills import MarketSide, Outcome, Quote, from_ticks, street_ticks, to_ticks
from mm_game.flow import RFQ, FlowSchedule, Side
from mm_game.rng import RngStreams, Stream

SRC_DIR = Path(mm_game.__file__).resolve().parent

# A dense short session for the statistical tests: ten minutes at six times the
# default rate, prints in a 60-540 s window, ~100 RFQs and ~11% informed.
STAT = GameConfig(
    session_length=600,
    event_min_time_s=60.0,
    event_end_margin_s=60.0,
    event_min_gap_s=120.0,
    rfq_rate_per_hour=600.0,
)
# Every dealer in on every RFQ: the street as it was before competition
# varied, for the tests that calibrate against a full street.
FULL_STREET = GameConfig(competitor_participation_prob=1.0)


def world(cfg: GameConfig, seed: int):
    rng = RngStreams(seed)
    path = fair_value.generate(cfg, rng)
    schedule = flow.generate(cfg, rng, path)
    return path, schedule, fills.pregenerate(cfg, rng, path, schedule)


def around_screen(cfg: GameConfig, width_ticks: int):
    def quote(rfq: RFQ, public: fair_value.PublicView, position: int) -> Quote:
        s = public.screen_at(rfq.step)
        return Quote(rfq.ordinal, bid=s - width_ticks * cfg.tick_size, ask=s + width_ticks * cfg.tick_size)

    return quote


def ticks_for(cfg: GameConfig, width_bp: float) -> int:
    """A half-width given in bp, as whole ticks. Statistical tests state their
    widths in bp so they test the same economics whatever the tick; a width
    that is not on the grid is a test bug, not a rounding to absorb."""
    k = round(width_bp / cfg.tick_size)
    if not math.isclose(k * cfg.tick_size, width_bp):
        raise ValueError(f"{width_bp}bp is not a whole number of {cfg.tick_size}bp ticks")
    return k


def play(cfg: GameConfig, seed: int, quote_fn):
    """The smallest session loop: quote, apply the limit, resolve, book."""
    path, schedule, sides = world(cfg, seed)
    position, reports, blocks = 0, [], []
    for rfq, market in zip(schedule.rfqs, sides):
        quote = quote_fn(rfq, path.public, position)
        blocked: tuple[str, ...] = ()
        if quote is not None:
            quote, blocked = fills.apply_position_limit(quote, rfq.size, position, cfg)
        report = fills.resolve(rfq, quote, market, cfg)
        position += report.quantity
        reports.append(report)
        blocks.append(blocked)
    return path, schedule, sides, reports, blocks


@pytest.fixture(scope="module")
def stat_worlds():
    return [world(STAT, s) for s in range(300)]


def make_rfq(
    side: Side = Side.BUY, size: int = 10, informed: bool = False, dealers: tuple[bool, ...] = (True, True, True)
) -> RFQ:
    return RFQ(
        ordinal=0,
        step=10,
        client_id=0,
        size=size,
        one_way=False,
        side=side,
        informed=informed,
        signal_bp=None,
        informed_about=None,
        base_size=float(size),
        dealers_in_comp=dealers,
    )


def make_market(limit: float, asks=(2, 3, 4), bids=(-2, -3, -4), tie_u: float = 0.5) -> MarketSide:
    return MarketSide(
        rfq_ordinal=0,
        competitor_bid_ticks=tuple(bids),
        competitor_ask_ticks=tuple(asks),
        competitor_mids=(0.0,) * len(asks),
        client_limit=limit,
        tie_break_u=tie_u,
    )


# --------------------------------------------------------------------------
# The ones that matter
# --------------------------------------------------------------------------


def test_pregeneration_never_sees_the_player():
    """``pregenerate`` takes no quote and no position, and the
    market it builds is bit-identical however the session is then played and
    whatever the player's own randomness consumed first. A market drawn lazily
    at quote time, or from a stream the player's strategy also uses, would
    make "same session, quote differently" compare two different markets."""
    params = list(inspect.signature(fills.pregenerate).parameters)
    assert params == ["config", "rng", "path", "schedule"]

    cfg = GameConfig()
    _, _, sides_tight, _, _ = play(cfg, 21, around_screen(cfg, 1))
    _, _, sides_wide, _, _ = play(cfg, 21, around_screen(cfg, 3))
    assert sides_tight == sides_wide

    rng = RngStreams(21)
    rng.stream(Stream.AGENTS).random(50_000)
    path = fair_value.generate(cfg, rng)
    schedule = flow.generate(cfg, rng, path)
    assert fills.pregenerate(cfg, rng, path, schedule) == sides_tight


@pytest.mark.parametrize("limit", [10_000, 200])
def test_paired_replay(limit):
    """Two quote sequences differing only at RFQ k: every
    MarketSide is identical, every RFQ before k resolves identically, and
    every RFQ after k resolves identically unless the different position has
    changed which side the limit blocks there. With an effectively unbounded
    limit nothing but k may differ; with the real one, a one-sided strategy
    that rides the limit shows the exception and nothing else."""
    cfg = GameConfig(position_limit_lots=limit)
    tick = cfg.tick_size

    def strategy(skip: int | None):
        def quote(rfq, public, position):
            if rfq.ordinal == skip:
                return None
            s = public.screen_at(rfq.step)
            return Quote(rfq.ordinal, bid=s, ask=s + 6 * tick)  # always buying at screen

        return quote

    _, _, sides_a, base, blocks_a = play(cfg, 4, strategy(None))
    k = next(i for i, r in enumerate(base) if r.outcome is Outcome.WON and i > 20)
    _, _, sides_b, changed, blocks_b = play(cfg, 4, strategy(k))
    assert sides_a == sides_b
    assert base[:k] == changed[:k]
    assert base[k] != changed[k]
    block_differences = 0
    for j in range(k + 1, len(base)):
        if blocks_a[j] == blocks_b[j]:
            assert base[j] == changed[j]
        else:
            block_differences += 1
    if limit == 10_000:
        assert block_differences == 0 and base[k + 1 :] == changed[k + 1 :]
    else:
        assert block_differences > 0  # the exception is genuinely exercised


def test_competitors_use_only_public_information():
    """Strip every hidden field from every RFQ -- informed flag,
    signal, the print it concerns -- and flip every latent side: every
    competitor quote is bit-identical. A dealer that widened for informed
    clients would be reading the answer key; the player's only defence must
    be reading flow. Client limits do change, as they should."""
    cfg = GameConfig()
    checked = 0
    for seed in range(20):
        rng = RngStreams(seed)
        path = fair_value.generate(cfg, rng)
        schedule = flow.generate(cfg, rng, path)
        if not any(q.informed for q in schedule.rfqs):
            continue  # nothing hidden to strip; proves nothing
        checked += 1
        if checked > 3:
            break
        blind = FlowSchedule(
            clients=schedule.clients,
            rfqs=tuple(
                dataclasses.replace(
                    q,
                    informed=False,
                    signal_bp=None,
                    informed_about=None,
                    side=Side.SELL if q.side is Side.BUY else Side.BUY,
                )
                for q in schedule.rfqs
            ),
        )
        a = fills.pregenerate(cfg, rng, path, schedule)
        b = fills.pregenerate(cfg, rng, path, blind)
        for x, y in zip(a, b):
            assert (x.competitor_bid_ticks, x.competitor_ask_ticks, x.competitor_mids) == (
                y.competitor_bid_ticks,
                y.competitor_ask_ticks,
                y.competitor_mids,
            )
        assert any(x.client_limit != y.client_limit for x, y in zip(a, b))
    assert checked > 3


def cover_errors(cfg: GameConfig, seeds) -> list[np.ndarray]:
    """Per session, the error against the truth of the mid implied by each
    RFQ's cover: the best competing price on the client's side, moved back by
    the dealers' (public) half-spread."""
    out = []
    for seed in seeds:
        path, schedule, sides = world(cfg, seed)
        errs = []
        for rfq, market in zip(schedule.rfqs, sides):
            hs = cfg.competitor_half_spread_bp_at(rfq.size, path.public.seconds_to_nearest_print(rfq.step))
            if rfq.side is Side.BUY:
                implied = from_ticks(market.best_ask_ticks, cfg) - hs
            else:
                implied = from_ticks(market.best_bid_ticks, cfg) + hs
            errs.append(implied - path.values[rfq.step])
        out.append(np.array(errs))
    return out


def averaging_slope(sessions: list[np.ndarray], windows=(1, 4, 16, 64)) -> tuple[float, dict[int, float]]:
    sds = {}
    for n in windows:
        blocks = [e[: (len(e) // n) * n].reshape(-1, n).mean(axis=1) for e in sessions if len(e) >= n]
        sds[n] = float(np.concatenate(blocks).std())
    slope = float(np.polyfit(np.log(list(sds)), np.log(list(sds.values())), 1)[0])
    return slope, sds


def test_cover_cannot_be_averaged_down_to_the_truth():
    """Average the cover-implied mid's error over n consecutive
    RFQs: with independent errors it would fall like 1/sqrt(n), a log-log
    slope of -0.5. Dealers share the screen's persistent error and carry
    their own, so it falls far more slowly -- measured slope about -0.32,
    and at 16 RFQs (about five minutes) the error is still half its
    single-RFQ size rather than a quarter.

    Over long windows it does fall: both components decorrelate over
    minutes, and by 64 RFQs (twenty minutes) the error is a quarter -- by
    which time the truth itself has moved more than that. The negative
    control, dealers with independent per-RFQ errors and no screen share,
    shows the measure detects 1/sqrt(n) when it is there.
    """
    slope, sds = averaging_slope(cover_errors(GameConfig(), range(40)))
    control_slope, control = averaging_slope(
        cover_errors(GameConfig(competitor_screen_weight=0.0, competitor_error_corr_s=0.0), range(40))
    )
    assert -0.40 < slope < -0.20, (slope, sds)
    assert sds[16] / sds[1] > 0.40
    assert control_slope < -0.42, (control_slope, control)
    assert control[16] / control[1] < 0.32


def test_winners_curse_fingerprint_exists_and_is_faint(stat_worlds):
    """Among fills won at a wide price, informed clients are
    over-represented -- they accept prices uninformed clients walk from --
    but nowhere near certain. At +/-1bp the informed share of wins is
    several times the base rate and still a minority: "if they trade at
    my wide price, I should worry", never "they must be informed"."""
    cfg = STAT
    w = ticks_for(cfg, 1.0) * cfg.tick_size
    won = informed_won = rfqs = informed_rfqs = 0
    for path, schedule, sides in stat_worlds:
        for rfq, market in zip(schedule.rfqs, sides):
            s = path.public.screen_at(rfq.step)
            quote = Quote(rfq.ordinal, bid=s - w, ask=s + w)
            report = fills.resolve(rfq, quote, market, cfg)
            rfqs += 1
            informed_rfqs += rfq.informed
            if report.outcome is Outcome.WON:
                won += 1
                informed_won += rfq.informed
    base = informed_rfqs / rfqs
    share = informed_won / won
    assert won > 500
    assert share > 2.0 * base
    assert share < 0.55


# --------------------------------------------------------------------------
# Resolution: feedback, ties, limits, purity
# --------------------------------------------------------------------------


def test_feedback_per_outcome():
    """What the player is told, outcome by outcome. A done-away that hid the
    side, or a pass that leaked the cover, would change how fast the
    crowd-versus-informed read builds -- silently."""
    cfg = GameConfig()
    s = cfg.start_level
    t = cfg.tick_size
    rfq = make_rfq(Side.BUY, size=10)
    market = make_market(limit=s + 4 * t)  # best competitor ask is 2 ticks up

    won = fills.resolve(rfq, Quote(0, ask=s + 1 * t), market, cfg)
    assert (won.outcome, won.side, won.price, won.cover, won.quantity) == (Outcome.WON, Side.BUY, s + t, s + 2 * t, -10)

    away = fills.resolve(rfq, Quote(0, ask=s + 3 * t), market, cfg)
    assert (away.outcome, away.side, away.price, away.cover, away.quantity) == (
        Outcome.DONE_AWAY,
        Side.BUY,
        None,
        s + 2 * t,
        0,
    )

    walked = fills.resolve(rfq, Quote(0, ask=s + 3 * t), make_market(limit=s + 1 * t), cfg)
    assert (walked.outcome, walked.side, walked.price, walked.cover, walked.quantity) == (
        Outcome.NOT_DONE,
        None,
        None,
        None,
        0,
    )

    for passed in (None, Quote(0)):
        p = fills.resolve(rfq, passed, market, cfg)
        assert (p.outcome, p.side, p.price, p.cover, p.quantity) == (Outcome.PASSED, None, None, None, 0)

    seller = fills.resolve(make_rfq(Side.SELL, size=7), Quote(0, bid=s - t), make_market(limit=s - 4 * t), cfg)
    assert (seller.outcome, seller.quantity, seller.cover) == (Outcome.WON, 7, s - 2 * t)


def test_only_dealers_in_competition_compete():
    """A dealer who was not asked cannot win the trade, set the cover or
    split a tie, however good its price. Dealer 0 shows the best ask of all;
    with it out, you win at a price it would have beaten, the cover is the
    best of the dealers who were in, and a tie is split with those only."""
    cfg = GameConfig()
    s, t = cfg.start_level, cfg.tick_size
    market = make_market(limit=s + 9 * t, asks=(1, 3, 4), tie_u=0.4)
    rfq = make_rfq(Side.BUY, dealers=(False, True, True))
    won = fills.resolve(rfq, Quote(0, ask=s + 2 * t), market, cfg)
    assert (won.outcome, won.cover) == (Outcome.WON, s + 3 * t)
    away = fills.resolve(make_rfq(Side.BUY, dealers=(True, True, True)), Quote(0, ask=s + 2 * t), market, cfg)
    assert (away.outcome, away.cover) == (Outcome.DONE_AWAY, s + t)
    # A tie with one dealer in: yours with probability 1/2 (u = 0.4 wins),
    # where counting the absent dealer at the same price would make it 1/3.
    tie_market = make_market(limit=s + 9 * t, asks=(3, 3, 5), tie_u=0.4)
    tie = fills.resolve(make_rfq(Side.BUY, dealers=(True, False, True)), Quote(0, ask=s + 3 * t), tie_market, cfg)
    assert tie.outcome is Outcome.WON


def test_with_nobody_else_in_you_face_only_the_limit():
    """In comp 0: inside the limit you win with no cover; outside it the
    client walks; a done-away is impossible; quoting only the other side is
    a walk, not a pass."""
    cfg = GameConfig()
    s, t = cfg.start_level, cfg.tick_size
    alone = (False, False, False)
    market = make_market(limit=s + 4 * t, asks=(1, 1, 1))
    won = fills.resolve(make_rfq(Side.BUY, dealers=alone), Quote(0, ask=s + 4 * t), market, cfg)
    assert (won.outcome, won.price, won.cover, won.quantity) == (Outcome.WON, s + 4 * t, None, -10)
    walked = fills.resolve(make_rfq(Side.BUY, dealers=alone), Quote(0, ask=s + 5 * t), market, cfg)
    assert walked.outcome is Outcome.NOT_DONE
    wrong_side = fills.resolve(make_rfq(Side.BUY, dealers=alone), Quote(0, bid=s - t), market, cfg)
    assert wrong_side.outcome is Outcome.NOT_DONE
    assert fills.resolve(make_rfq(Side.BUY, dealers=alone), None, market, cfg).outcome is Outcome.PASSED
    seen = set()
    for path, schedule, sides in (world(STAT, s) for s in range(30)):
        for rfq, m in zip(schedule.rfqs, sides):
            if rfq.in_comp == 0:
                p = path.public.screen_at(rfq.step)
                for w in (1, 3, 6):
                    r = fills.resolve(rfq, Quote(rfq.ordinal, bid=p - w * t, ask=p + w * t), m, cfg)
                    seen.add(r.outcome)
                    assert r.outcome is not Outcome.DONE_AWAY
                    assert r.cover is None
    assert seen == {Outcome.WON, Outcome.NOT_DONE}


def test_participation_never_moves_a_quote_a_limit_or_the_street():
    """Dealers' prices and errors are drawn for every dealer whether or not
    it is asked, and limits and tie-breaks per RFQ, so moving the
    participation probability or the informed multiplier leaves every
    market side bit-identical. Drawn only for dealers in competition, the
    street's errors would re-roll whenever competition changed, and "same
    session, different competition" would compare two different markets."""
    base = world(GameConfig(), 6)[2]
    for change in (dict(competitor_participation_prob=0.2), dict(competitor_participation_prob=1.0),
                   dict(informed_participation_multiplier=0.3)):
        assert world(GameConfig(**change), 6)[2] == base


def test_client_walks_unless_the_best_price_is_inside_the_limit():
    """The client trades with the best price only if it is inside their
    limit -- even when you are best, a price beyond the limit is NOT_DONE,
    and a competitor beyond the limit never trades."""
    cfg = GameConfig()
    s, t = cfg.start_level, cfg.tick_size
    rfq = make_rfq(Side.BUY)
    tight_limit = make_market(limit=s + 0.5 * t)
    assert fills.resolve(rfq, Quote(0, ask=s + t), tight_limit, cfg).outcome is Outcome.NOT_DONE
    on_limit = make_market(limit=s + t)
    assert fills.resolve(rfq, Quote(0, ask=s + t), on_limit, cfg).outcome is Outcome.WON
    # Your only price on the wrong side: the competitor's ask is still judged.
    assert fills.resolve(rfq, Quote(0, bid=s - t), make_market(limit=s + 4 * t), cfg).outcome is Outcome.DONE_AWAY
    assert fills.resolve(rfq, Quote(0, bid=s - t), make_market(limit=s + t), cfg).outcome is Outcome.NOT_DONE


@pytest.mark.parametrize("at_best,threshold", [(1, 1 / 2), (2, 1 / 3), (3, 1 / 4)])
def test_ties_are_broken_by_the_pre_drawn_uniform(at_best, threshold):
    """A tie with k dealers at the best price goes to you with probability
    1 / (k + 1), decided by the RFQ's own pre-drawn uniform -- never a fixed
    rule that always favours one side, which would make the natural width
    either a free win or a sure loss."""
    cfg = GameConfig()
    s, t = cfg.start_level, cfg.tick_size
    asks = (2,) * at_best + (5,) * (3 - at_best)
    rfq = make_rfq(Side.BUY)
    below = fills.resolve(rfq, Quote(0, ask=s + 2 * t), make_market(s + 9 * t, asks=asks, tie_u=threshold - 1e-9), cfg)
    above = fills.resolve(rfq, Quote(0, ask=s + 2 * t), make_market(s + 9 * t, asks=asks, tie_u=threshold + 1e-9), cfg)
    assert below.outcome is Outcome.WON and above.outcome is Outcome.DONE_AWAY


def test_tie_win_rate_matches_the_rule(stat_worlds):
    """Statistically: over every accepted tie at +/-0.5bp, the win rate
    matches the mean of 1 / (k + 1). Ties are still a large share of RFQs at
    the natural width, so a biased tie-break would bias the hit rate there."""
    cfg = STAT
    w = ticks_for(cfg, 0.5) * cfg.tick_size
    won, expected = [], []
    for path, schedule, sides in stat_worlds[:100]:
        for rfq, market in zip(schedule.rfqs, sides):
            s = path.public.screen_at(rfq.step)
            quote = Quote(rfq.ordinal, bid=s - w, ask=s + w)
            street = street_ticks(rfq, market)
            if not street:
                continue
            best = min(street, key=lambda k: int(rfq.side) * k)
            yours = to_ticks(quote.ask if rfq.side is Side.BUY else quote.bid, cfg)
            report = fills.resolve(rfq, quote, market, cfg)
            if yours == best and report.outcome is not Outcome.NOT_DONE:
                won.append(report.outcome is Outcome.WON)
                expected.append(1 / (street.count(best) + 1))
    assert len(won) > 2000
    assert abs(np.mean(won) - np.mean(expected)) < 0.03


def test_resolve_is_pure(stat_worlds):
    """Same inputs, same output, in any order and any number of
    times. Hidden state in resolve -- a counter, a cache keyed on something
    stale -- would make replays drift."""
    cfg = STAT
    path, schedule, sides = stat_worlds[0]
    calls = []
    for rfq, market in zip(schedule.rfqs, sides):
        s = path.public.screen_at(rfq.step)
        calls.append((rfq, Quote(rfq.ordinal, bid=s - cfg.tick_size, ask=s + cfg.tick_size), market))
    first = [fills.resolve(r, q, m, cfg) for r, q, m in calls]
    shuffled = list(range(len(calls)))
    random.Random(5).shuffle(shuffled)  # test-local ordering, not src/
    again = {i: fills.resolve(*calls[i][:3], cfg) for i in shuffled}
    assert [again[i] for i in range(len(calls))] == first
    assert [fills.resolve(r, q, m, cfg) for r, q, m in calls] == first


def test_resolve_refuses_mismatched_ordinals():
    cfg = GameConfig()
    with pytest.raises(ValueError):
        fills.resolve(make_rfq(), Quote(1, ask=cfg.start_level + 1), make_market(cfg.start_level + 9), cfg)
    with pytest.raises(ValueError):
        fills.resolve(make_rfq(), Quote(0), dataclasses.replace(make_market(1.0), rfq_ordinal=3), cfg)


def test_quotes_must_be_on_the_grid_and_not_crossed():
    """Off-grid prices are refused rather than rounded: a price a hair off a
    tick would silently break a tie."""
    cfg = GameConfig()
    with pytest.raises(ValueError, match="tick grid"):
        fills.resolve(make_rfq(), Quote(0, ask=cfg.start_level + 0.3), make_market(cfg.start_level + 9), cfg)
    with pytest.raises(ValueError):
        Quote(0, bid=9551.0, ask=9550.5)
    with pytest.raises(ValueError):
        Quote(0, bid=float("nan"))
    assert to_ticks(from_ticks(-7, cfg), cfg) == -7


# --------------------------------------------------------------------------
# Position limit
# --------------------------------------------------------------------------


def test_position_limit_blocks_the_risk_increasing_side():
    """A side whose fill would take the position through the
    limit is blocked and reported; the risk-reducing side still trades; a
    fill landing exactly on the limit is allowed."""
    cfg = GameConfig()
    s, t = cfg.start_level, cfg.tick_size
    assert fills.blocked_sides(20, 190, cfg) == ("bid",)
    assert fills.blocked_sides(20, -190, cfg) == ("ask",)
    assert fills.blocked_sides(20, 180, cfg) == ()
    assert fills.blocked_sides(100, 0, cfg) == ()

    quote = Quote(0, bid=s - t, ask=s + t)
    limited, blocked = fills.apply_position_limit(quote, 20, 190, cfg)
    assert blocked == ("bid",) and limited.bid is None and limited.ask == s + t

    # Client sells to you: your bid was blocked, so you cannot buy.
    report = fills.resolve(make_rfq(Side.SELL, size=20), limited, make_market(limit=s - 4 * t), cfg)
    assert report.outcome is Outcome.DONE_AWAY and report.quantity == 0
    # Client buys from you: the risk-reducing ask still trades.
    report = fills.resolve(make_rfq(Side.BUY, size=20), limited, make_market(limit=s + 4 * t), cfg)
    assert report.outcome is Outcome.WON and report.quantity == -20


def test_limit_binds_in_play_and_every_block_is_recorded():
    """Riding the limit with a one-sided strategy: the position never leaves
    [-200, 200], blocks happen, and each is reported at its RFQ."""
    cfg = GameConfig()

    def buyer(rfq, public, position):
        s = public.screen_at(rfq.step)
        return Quote(rfq.ordinal, bid=s, ask=s + 6 * cfg.tick_size)

    _, _, _, reports, blocks = play(cfg, 4, buyer)
    positions = np.cumsum([r.quantity for r in reports])
    assert positions.max() <= cfg.position_limit_lots and positions.min() >= -cfg.position_limit_lots
    assert sum(1 for b in blocks if b == ("bid",)) > 10
    assert positions.max() > cfg.position_limit_lots - cfg.size_cap_lots


# --------------------------------------------------------------------------
# Competitors and client limits
# --------------------------------------------------------------------------


def test_competitor_quotes_sit_on_the_grid_and_widen_for_size_and_prints(stat_worlds):
    """Dealers quote whole ticks, never locked or crossed, and their spread
    grows with ticket size and around prints -- public information. A dealer
    that did not widen into a print would make winning there safe; one that
    widened on hidden information is caught by the public-information test."""
    by_size, near, far = {"small": [], "large": []}, [], []
    for path, schedule, sides in stat_worlds[:150]:
        for rfq, market in zip(schedule.rfqs, sides):
            spreads = np.array(market.competitor_ask_ticks) - np.array(market.competitor_bid_ticks)
            assert np.all(spreads >= 1)
            spread = float(spreads.mean()) * STAT.tick_size  # bp
            if rfq.size <= 8:
                by_size["small"].append(spread)
            elif rfq.size >= 40:
                by_size["large"].append(spread)
            seconds = path.public.seconds_to_nearest_print(rfq.step)
            (near if seconds <= 30 else far if seconds >= 200 else []).append(spread)
    assert np.mean(by_size["large"]) > np.mean(by_size["small"]) + 0.25
    assert np.mean(near) > np.mean(far) + 0.5


def test_uninformed_tolerance_is_mostly_three_quarters_of_a_bp():
    """Uninformed limits: median 0.75bp from the screen, ~16% under 0.5bp,
    ~4% beyond 1.5bp -- "most accept a little over the spread, the
    occasional urgent client three times that". Checked with size growth off
    so the shape is the configured lognormal's."""
    cfg = dataclasses.replace(STAT, client_tolerance_size_growth=0.0)
    distances = []
    for seed in range(150):
        path, schedule, sides = world(cfg, seed)
        for rfq, market in zip(schedule.rfqs, sides):
            if not rfq.informed:
                screen = path.public.screen_at(rfq.step)
                distances.append(int(rfq.side) * (market.client_limit - screen))
    d = np.array(distances)
    assert d.size > 10_000
    assert abs(np.median(d) - 0.75) < 0.025
    assert 0.12 < np.mean(d < 0.5) < 0.20
    assert 0.02 < np.mean(d > 1.5) < 0.06


def test_informed_clients_pay_wider_and_rarely_walk(stat_worlds):
    """Informed limits sit wider than uninformed -- they give up about half
    their expected move -- but the ranges overlap, so a wide fill is evidence
    and not proof. At the natural width (+/-0.5bp) informed clients almost
    never walk, so a NOT_DONE carries the opposite signal."""
    cfg = STAT
    w = ticks_for(cfg, 0.5) * cfg.tick_size
    dist = {True: [], False: []}
    walked = {True: [], False: []}
    for path, schedule, sides in stat_worlds:
        for rfq, market in zip(schedule.rfqs, sides):
            s = path.public.screen_at(rfq.step)
            dist[rfq.informed].append(int(rfq.side) * (market.client_limit - s))
            report = fills.resolve(rfq, Quote(rfq.ordinal, bid=s - w, ask=s + w), market, cfg)
            walked[rfq.informed].append(report.outcome is Outcome.NOT_DONE)
    inf, unin = np.array(dist[True]), np.array(dist[False])
    assert np.median(inf) > np.median(unin) + 0.2
    assert np.mean(unin > np.median(inf)) > 0.05  # overlap: plenty of uninformed beyond the informed median
    assert np.mean(walked[True]) < 0.04 < np.mean(walked[False])


def hit_and_tie_curve(
    cfg: GameConfig, widths: tuple[int, ...], seeds, in_comp: int | None = None
) -> dict[int, tuple[float, float]]:
    """Per half-width in ticks: (hit rate, share of wins that were ties),
    over every RFQ or only those with ``in_comp`` dealers in competition."""
    out = {}
    worlds = [world(cfg, s) for s in seeds]
    for k in widths:
        won = won_on_tie = total = 0
        for path, schedule, sides in worlds:
            for rfq, market in zip(schedule.rfqs, sides):
                if in_comp is not None and rfq.in_comp != in_comp:
                    continue
                s = path.public.screen_at(rfq.step)
                quote = Quote(rfq.ordinal, bid=s - k * cfg.tick_size, ask=s + k * cfg.tick_size)
                report = fills.resolve(rfq, quote, market, cfg)
                street = street_ticks(rfq, market)
                best = min(street, key=lambda x: int(rfq.side) * x) if street else None
                mine = to_ticks(quote.ask if rfq.side is Side.BUY else quote.bid, cfg)
                total += 1
                if report.outcome is Outcome.WON:
                    won += 1
                    won_on_tie += mine == best
        out[k] = (won / total, won_on_tie / max(won, 1))
    return out


def test_hit_rate_at_half_a_bp_against_a_full_street_is_near_a_third():
    """Calibration target: quoting 0.5bp either side of the screen against
    all three dealers wins roughly a third of RFQs. Measured ~34%; the band
    allows tuning in play without silently drifting to a different game."""
    cfg = FULL_STREET
    won = total = 0
    for seed in range(20):
        _, _, _, reports, _ = play(cfg, seed, around_screen(cfg, ticks_for(cfg, 0.5)))
        won += sum(r.outcome is Outcome.WON for r in reports)
        total += len(reports)
    assert 0.27 < won / total < 0.42


def test_width_is_graded_when_competition_is_thin_and_binary_when_full():
    """The point of variable competition. With nobody else in, the client's
    limit is the only filter and the hit rate falls gradually with width
    (measured 100 / 89 / 58 / 32% at one to four ticks); with one dealer it
    still grades (90 / 56 / 17 / 3%); against all three it is near-binary
    (78 / 34 / 6 / 0.6%). At every width, more competition never wins more.
    Catches the count being drawn but ignored in resolution -- a public
    number that changes nothing -- or a resolution that compares against
    dealers who were never asked."""
    cfg = GameConfig()
    curves = {k: hit_and_tie_curve(cfg, (1, 2, 3, 4), range(40), in_comp=k) for k in (0, 1, 3)}
    for w in (1, 2, 3, 4):
        assert curves[0][w][0] >= curves[1][w][0] >= curves[3][w][0]
    assert curves[0][4][0] > 0.2 and curves[0][3][0] > 0.45
    assert curves[1][3][0] > 0.10
    assert curves[3][3][0] < 0.10 and curves[3][4][0] < 0.02
    assert curves[0][2][1] == 0.0  # nobody to tie with


def test_the_grid_resolves_the_competitive_range():
    """At the quarter-bp tick, one, two and three ticks either side of the
    screen are three distinct choices -- hit rates near 78%, 34% and 6% --
    and at the first two, most wins are outright rather than tie-breaks.

    The failure it catches is a grid as coarse as the dealers' dispersion
    around the screen. The negative control is the old half-bp tick at the
    same bp economics: there one tick wide was decided mostly by ties and one
    tick wider won almost nothing, so the width decision was binary."""
    fine = hit_and_tie_curve(FULL_STREET, (1, 2, 3), range(15))
    (h1, t1), (h2, t2), (h3, _) = fine[1], fine[2], fine[3]
    assert h1 > h2 > h3
    assert 0.65 < h1 < 0.88 and 0.27 < h2 < 0.42 and 0.03 < h3 < 0.10
    assert t1 < 0.30 and t2 < 0.55

    coarse_cfg = dataclasses.replace(FULL_STREET, tick_size=0.5, tick_value=25.0)
    coarse = hit_and_tie_curve(coarse_cfg, (1, 2), range(15))
    assert coarse[1][1] > 0.55  # wins at the natural width mostly ties
    assert coarse[2][0] < 0.03  # and one tick wider wins almost nothing


def test_a_tick_change_moves_prices_not_economics():
    """Same seed at a quarter-bp and a half-bp tick (both $50/bp/lot):
    dealer mids and every client's limit distance from the screen are
    identical in bp, and dealer quotes differ only by rounding to the grid.
    A tolerance or spread reintroduced in ticks would rescale silently with
    the tick, and this is where it shows."""
    fine, coarse = GameConfig(), GameConfig(tick_size=0.5, tick_value=25.0)
    pa, sa, ma = world(fine, 8)
    pb, sb, mb = world(coarse, 8)
    assert sa.rfqs == sb.rfqs
    for rfq, x, y in zip(sa.rfqs, ma, mb):
        assert x.competitor_mids == y.competitor_mids
        assert x.client_limit - pa.public.screen_at(rfq.step) == pytest.approx(
            y.client_limit - pb.public.screen_at(rfq.step), abs=1e-9
        )
        hs = fine.competitor_half_spread_bp_at(rfq.size, pa.public.seconds_to_nearest_print(rfq.step))
        for cfg, m in ((fine, x), (coarse, y)):
            for mid, ask in zip(m.competitor_mids, m.competitor_ask_ticks):
                assert abs(from_ticks(ask, cfg) - (mid + hs)) <= cfg.tick_size / 2 + 1e-9


# --------------------------------------------------------------------------
# Stream discipline
# --------------------------------------------------------------------------


def test_fills_draws_only_from_competition_and_acceptance():
    """Every stream pregenerate requests is recorded: only the dealers'
    errors and the per-RFQ acceptance draws. Acceptance drawn from the
    player's own stream, or folded into the competitors', would re-randomise
    the comparison whenever quoting changed."""

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

    cfg = GameConfig()
    rng = RngStreams(2)
    path = fair_value.generate(cfg, rng)
    schedule = flow.generate(cfg, rng, path)
    rec = Recording(2)
    fills.pregenerate(cfg, rec, path, schedule)
    assert rec.used == {Stream.COMPETITION, Stream.ACCEPTANCE}


def test_acceptance_and_competition_never_move_each_other():
    """Retune the client limits: every dealer quote is bit-identical. Retune
    the dealers: every limit and tie-break is bit-identical."""
    base = fills.pregenerate(GameConfig(), *world_inputs(GameConfig(), 6))
    tolerant = fills.pregenerate(
        GameConfig(client_tolerance_bp=1.5, client_tolerance_log_sd=0.1),
        *world_inputs(GameConfig(client_tolerance_bp=1.5, client_tolerance_log_sd=0.1), 6),
    )
    dealers = GameConfig(competitor_error_bp=0.4, competitor_half_spread_bp=0.75, hedge_cost_bp=1.3)
    retuned = fills.pregenerate(dealers, *world_inputs(dealers, 6))
    for a, b, c in zip(base, tolerant, retuned):
        assert (a.competitor_bid_ticks, a.competitor_ask_ticks) == (b.competitor_bid_ticks, b.competitor_ask_ticks)
        assert (a.client_limit, a.tie_break_u) == (c.client_limit, c.tie_break_u)


def world_inputs(cfg: GameConfig, seed: int):
    rng = RngStreams(seed)
    path = fair_value.generate(cfg, rng)
    return rng, path, flow.generate(cfg, rng, path)


def test_the_street_is_a_time_process_not_an_rfq_process():
    """Dealer errors are keyed on dealer and sampled in time, so changing the
    arrival rate never re-rolls the street: an RFQ at the same step sees the
    same dealer mids whichever schedule it belongs to. Keyed on RFQ ordinal,
    every dealer error would move with the rate, and the errors' persistence
    would be measured in inquiries rather than seconds."""
    slow, fast = GameConfig(), GameConfig(rfq_rate_per_hour=150.0)
    _, sched_slow, sides_slow = world(slow, 9)
    _, sched_fast, sides_fast = world(fast, 9)
    mids_slow = {q.step: m.competitor_mids for q, m in zip(sched_slow.rfqs, sides_slow)}
    common = 0
    for q, m in zip(sched_fast.rfqs, sides_fast):
        if q.step in mids_slow:
            assert m.competitor_mids == mids_slow[q.step]
            common += 1
    assert common > 10


# --------------------------------------------------------------------------
# Records and hygiene
# --------------------------------------------------------------------------


def test_reports_round_trip_json():
    cfg = GameConfig()
    _, _, _, reports, _ = play(cfg, 3, around_screen(cfg, 1))
    for r in reports[:50]:
        record = r.to_record()
        assert json.loads(json.dumps(record)) == record


def test_module_imports():
    source = (SRC_DIR / "fills.py").read_text(encoding="utf-8")
    internal = set(re.findall(r"^\s*from\s+mm_game\.(\w+)\s+import", source, re.MULTILINE))
    assert internal == {"rng", "config", "fair_value", "flow"}
