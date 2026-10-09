"""Tests for mm_game.market: the hedge venue and the closing liquidation.

The ones that matter most: the venue is structurally blind to the truth, the
hedge cost exceeds the competitors' quotes everywhere (checked exactly, not
sampled), and quote-fill-hedge is not a money machine in play. Each docstring
says what silent failure it catches.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

import mm_game
from mm_game import fair_value, fills, flow, market
from mm_game.config import GameConfig
from mm_game.fair_value import PublicView
from mm_game.fills import Outcome, Quote
from mm_game.rng import RngStreams

SRC_DIR = Path(mm_game.__file__).resolve().parent


def path_for(cfg: GameConfig, seed: int) -> fair_value.FairValuePath:
    return fair_value.generate(cfg, RngStreams(seed))


# --------------------------------------------------------------------------
# Truth-blind
# --------------------------------------------------------------------------


def test_hedge_and_liquidate_are_structurally_truth_blind():
    """Both take the public view and nothing else that carries
    the truth: their signatures say so, the module never imports the truth
    type or names a truth field, and swapping the truth under an unchanged
    screen changes nothing. A venue that filled off the truth would leak fair
    value through every hedge price and defeat the screen's noise."""
    for fn in (market.hedge, market.liquidate):
        annotations = inspect.signature(fn).parameters
        assert annotations["public"].annotation in (PublicView, "PublicView")
        assert all("FairValuePath" not in str(p.annotation) for p in annotations.values())
    source = (SRC_DIR / "market.py").read_text(encoding="utf-8")
    code = re.sub(r'"""(.|\n)*?"""', "", source)  # docstrings may discuss the truth
    assert "FairValuePath" not in code
    for field in (".values", ".relative", "screen_error", "jump_sizes", "drift_windows", ".diffusion"):
        assert field not in code

    cfg = GameConfig()
    path = path_for(cfg, 3)
    other_truth = dataclasses.replace(path, values=path.values + 25.0, relative=path.relative + 25.0)
    for step in (0, 1234, path.n_steps):
        assert market.hedge(step, 30, 0, path.public, cfg) == market.hedge(step, 30, 0, other_truth.public, cfg)
    assert market.liquidate(-80, path.n_steps, path.public, cfg) == market.liquidate(
        -80, path.n_steps, other_truth.public, cfg
    )


def test_market_draws_nothing():
    """Hedging is deterministic: same request, same step, same fill. A random
    impact draw would need a stream, and if that stream were shared with
    anything else, the player's hedging would re-roll the market."""
    source = (SRC_DIR / "market.py").read_text(encoding="utf-8")
    assert "RngStreams" not in source and "Stream" not in source
    cfg = GameConfig()
    path = path_for(cfg, 4)
    assert market.hedge(500, -40, 10, path.public, cfg) == market.hedge(500, -40, 10, path.public, cfg)


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------


def test_hedge_prices_off_the_screen_plus_cost():
    """Buy at the screen showing at the step plus the per-lot cost for that
    size and distance from a print; sell at the screen minus it. Cost grows
    with size and widens around prints."""
    cfg = GameConfig()
    path = next(p for p in (path_for(cfg, s) for s in range(50)) if p.event_steps.size)
    pub = path.public
    step = 1000
    seconds = pub.seconds_to_nearest_print(step)
    buy = market.hedge(step, 40, 0, pub, cfg)
    sell = market.hedge(step, -40, 0, pub, cfg)
    cost = cfg.hedge_cost_bp_at(40, seconds)
    assert buy.price == pytest.approx(pub.screen_at(step) + cost)
    assert sell.price == pytest.approx(pub.screen_at(step) - cost)
    assert buy.cost_bp_per_lot == cost and buy.cost_bp == pytest.approx(40 * cost)

    small = market.hedge(step, 5, 0, pub, cfg).cost_bp_per_lot
    large = market.hedge(step, 90, 0, pub, cfg).cost_bp_per_lot
    assert large > small
    e = int(pub.event_steps[0])
    assert market.hedge(e, 10, 0, pub, cfg).cost_bp_per_lot > market.hedge(e - 3000, 10, 0, pub, cfg).cost_bp_per_lot


def test_hedge_cost_in_bp_does_not_move_with_the_tick():
    """The same hedge at the same step costs the same bp per lot at a
    quarter-bp and a half-bp tick. A cost reintroduced in ticks would halve
    every hedge with the tick change, and nothing else would notice."""
    fine, coarse = GameConfig(), GameConfig(tick_size=0.5, tick_value=25.0)
    a, b = path_for(fine, 5).public, path_for(coarse, 5).public
    for step in (0, 4321, 50_000):
        for qty in (3, -40, 100):
            x, y = market.hedge(step, qty, 0, a, fine), market.hedge(step, qty, 0, b, coarse)
            assert x.cost_bp_per_lot == y.cost_bp_per_lot
            assert x.price - a.screen_at(step) == pytest.approx(y.price - b.screen_at(step))


def test_no_stale_hedge_after_a_print():
    """At a print step the venue prices off the screen observed at that step,
    which already shows the print. Without the forced observation it would
    price off the reading from up to 5 s earlier, and hedging the instant a
    print lands would fill at the pre-print level: free money."""
    cfg = GameConfig()
    r = cfg.screen_refresh_steps
    found = 0
    for seed in range(60):
        path = path_for(cfg, seed)
        pub = path.public
        for e in (int(x) for x in path.event_steps if x % r):
            stale = float(pub.screen[np.searchsorted(pub.screen_steps, e - 1, side="right") - 1])
            fill = market.hedge(e, 1, 0, pub, cfg)
            assert fill.price == pytest.approx(pub.screen_at(e) + fill.cost_bp_per_lot)
            if pub.screen_at(e) != stale:
                found += 1
                assert fill.price != pytest.approx(stale + fill.cost_bp_per_lot)
    assert found > 20


# --------------------------------------------------------------------------
# Limit and liquidation
# --------------------------------------------------------------------------


def test_hedges_count_toward_the_limit_and_are_clipped():
    """A hedge can never take the position through the limit: it is clipped
    to the room left and says so. Risk-reducing hedges are never clipped."""
    cfg = GameConfig()
    pub = path_for(cfg, 1).public
    clipped = market.hedge(100, 50, 180, pub, cfg)
    assert (clipped.quantity, clipped.requested, clipped.clipped) == (20, 50, True)
    at_limit = market.hedge(100, -10, -200, pub, cfg)
    assert (at_limit.quantity, at_limit.price, at_limit.clipped) == (0, None, True)
    reducing = market.hedge(100, -150, 180, pub, cfg)
    assert (reducing.quantity, reducing.clipped) == (-150, False)
    none = market.hedge(100, 0, 0, pub, cfg)
    assert (none.quantity, none.clipped, none.cost_bp) == (0, False, 0.0)


def test_liquidation_cost_is_zero_flat_and_grows_with_size():
    """Flat costs nothing; the cost per lot and in total grows
    with the residual position, symmetrically long and short; and it is the
    hedge cost at the final screen times the end-of-day multiplier."""
    cfg = GameConfig()
    pub = path_for(cfg, 2).public
    n = pub.n_steps
    assert market.liquidate(0, n, pub, cfg).cost_bp == 0.0
    totals = [market.liquidate(p, n, pub, cfg).cost_bp for p in (10, 50, 100, 200)]
    assert all(b > a for a, b in zip(totals, totals[1:]))
    per_lot = [market.liquidate(p, n, pub, cfg).cost_bp_per_lot for p in (10, 50, 100, 200)]
    assert all(b > a for a, b in zip(per_lot, per_lot[1:]))
    assert market.liquidate(120, n, pub, cfg).cost_bp == market.liquidate(-120, n, pub, cfg).cost_bp
    long = market.liquidate(120, n, pub, cfg)
    expected = cfg.hedge_cost_bp_at(120, pub.seconds_to_nearest_print(n)) * cfg.close_liquidation_multiplier
    assert long.quantity == -120 and long.cost_bp_per_lot == pytest.approx(expected)
    assert long.price == pytest.approx(pub.screen_at(n) - expected)
    # Dearer than hedging the same position just before the close.
    assert long.cost_bp > market.hedge(n - 1, -120, 120, pub, cfg).cost_bp


# --------------------------------------------------------------------------
# The hedge-margin rule
# --------------------------------------------------------------------------


def brute_force_min_excess(cfg: GameConfig) -> float:
    sizes = np.arange(1, cfg.size_cap_lots + 1)
    seconds = np.concatenate([np.arange(0, 400, 0.5), [1e9]])
    return min(
        cfg.hedge_cost_bp_at(s, d) - cfg.competitor_max_quote_distance_bp(s, d)
        for s in sizes
        for d in seconds
    )


def test_hedge_margin_holds_everywhere_at_the_defaults_and_the_check_is_exact():
    """The rule: per-lot hedge cost exceeds the furthest a competitor quote
    can sit from its own mid (half-spread plus half a tick of rounding) by
    hedge_margin_bp at every size and every distance from a print. The
    config checks a handful of corners; a brute-force sweep over every size
    and distances in half-second steps agrees with it, so the corners are
    not missing an interior minimum. At the quarter-bp tick the rounding
    term is 0.125bp, so the excess is 0.375bp against a required 0.25bp."""
    cfg = GameConfig()
    assert cfg.hedge_margin_shortfall() is None
    assert brute_force_min_excess(cfg) >= cfg.hedge_margin_bp - 1e-9
    assert brute_force_min_excess(cfg) == pytest.approx(0.375, abs=0.01)  # binds far from a print


def unvalidated(**changes) -> GameConfig:
    """A config that skips __post_init__, so the brute-force sweep can be run
    on exactly the settings validation refuses."""
    cfg = object.__new__(GameConfig)
    for f in dataclasses.fields(GameConfig):
        object.__setattr__(cfg, f.name, changes.get(f.name, getattr(GameConfig(), f.name)))
    return cfg


@pytest.mark.parametrize(
    "change",
    [
        dict(hedge_cost_bp=0.85),  # too cheap everywhere
        dict(hedge_impact_bp=0.25),  # dealers widen for size faster than the venue
        dict(hedge_print_widen=0.2),  # dealers widen into the print, the venue barely does
        dict(hedge_print_window_s=30.0),  # the venue stops widening while dealers still are
    ],
)
def test_hedge_margin_violations_are_refused(change):
    """Each way the rule can fail -- too cheap, too little impact, too little
    widening at a print, too short a widening window -- is refused at
    construction. Unrefused, any of them opens a pocket where quote, fill and
    hedge instantly is riskless, and adverse selection mostly disappears
    because the player is flat before every print.

    The point named in the error genuinely attains the worst excess, and that
    excess equals a brute-force sweep's. Where several corners tie for worst
    -- equal size slopes make far-from-print excess the same at every size --
    which one is named is arbitrary, so only the value is pinned.
    """
    with pytest.raises(ValueError, match="hedge cost must exceed"):
        GameConfig(**change)
    bad = unvalidated(**change)
    size, seconds, excess = bad.hedge_margin_shortfall()
    assert excess < GameConfig().hedge_margin_bp
    assert excess == pytest.approx(
        bad.hedge_cost_bp_at(size, seconds) - bad.competitor_max_quote_distance_bp(size, seconds)
    )
    assert brute_force_min_excess(bad) == pytest.approx(excess, abs=1e-9)


def test_quote_fill_hedge_is_not_riskless_in_play():
    """The rule's purpose, checked in play. Quote each RFQ at the hedge cost
    rounded up to the tick -- the tightest price at which an instant hedge
    cannot lose -- and hedge every fill at once. If this ever filled, it would
    be riskless profit. Over 40 two-hour sessions it fills nothing: dealers
    and client limits sit inside the hedge cost, as the margin intends."""
    cfg = GameConfig()
    fills_count = 0
    for seed in range(40):
        rng = RngStreams(seed)
        path = fair_value.generate(cfg, rng)
        schedule = flow.generate(cfg, rng, path)
        sides = fills.pregenerate(cfg, rng, path, schedule)
        pub = path.public
        position = 0
        for rfq, side_ in zip(schedule.rfqs, sides):
            cost = cfg.hedge_cost_bp_at(rfq.size, pub.seconds_to_nearest_print(rfq.step))
            x = math.ceil(cost / cfg.tick_size - 1e-9)
            s = pub.screen_at(rfq.step)
            quote = Quote(rfq.ordinal, bid=s - x * cfg.tick_size, ask=s + x * cfg.tick_size)
            report = fills.resolve(rfq, quote, side_, cfg)
            if report.outcome is Outcome.WON:
                fills_count += 1
                position += report.quantity
                hedge = market.hedge(rfq.step, -report.quantity, position, pub, cfg)
                position += hedge.quantity
                edge = abs(report.price - s) - hedge.cost_bp_per_lot
                assert edge >= 0  # if it ever fills, it is riskless: that is the failure
    assert fills_count / 40 < 0.1
