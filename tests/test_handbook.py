"""Tests for the desk handbook.

The handbook tells the player their desk exactly and the market roughly.
Its exact numbers are generated from the config, so the failures that
matter are a number that drifts from the model it describes, a number that
was typed rather than generated, Part 3 reaching the page before it has
been earned, and anything in it coming from the hidden world. Each
docstring says which.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from pathlib import Path

import pytest

from mm_game import game, handbook, host as host_module, market, world
from mm_game.config import SESSION_PRESETS, GameConfig, ScoringConfig
from mm_game.engine import Settings
from mm_game.records import TrackRecord
from mm_game.storage import MemoryStorage

from wire import ManualClock

SCORING = ScoringConfig()
# A desk with different numbers everywhere Part 1 speaks: if any were typed
# rather than generated, the text would not follow.
PERTURBED = (
    GameConfig.for_session(
        60,
        position_limit_lots=150,
        hedge_cost_bp=1.5,
        hedge_impact_bp=0.75,
        hedge_print_window_s=180.0,
        hedge_print_widen=0.5,
        close_liquidation_multiplier=2.0,
        rfq_rate_per_hour=80.0,
        n_competitors=2,
        screen_refresh_s=10.0,
        tick_value=25.0,
        event_count_probs=(0.5, 0.5),
    ),
    ScoringConfig(risk_kappa=4.2e-5),
)
DESKS = [(GameConfig.for_session(m), SCORING) for m in (30, 60, 120)] + [PERTURBED]
WORDS = ("none", "one", "two", "three", "four", "five")
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")  # 1,225 and 0.25, not "0," in a list


def value(shown: str) -> float:
    """A number as the handbook wrote it: $1,225 -> 1225, 60% -> 0.6, double -> 2."""
    if shown == "double":
        return 2.0
    m = re.search(r"\d[\d,]*\.?\d*|\.\d+", shown)
    x = float(m.group(0).replace(",", ""))
    return x / 100 if shown.endswith("%") else x


def far_cost(config: GameConfig, size: float) -> float:
    return config.hedge_cost_bp_at(size, math.inf)


def expected_desk(config: GameConfig, scoring: ScoringConfig) -> dict[str, float | str]:
    """Part 1's numbers, worked out independently -- from the formulas and
    from what the venue and the risk penalty actually do."""
    a, b = config.jump_size_min_bp, config.jump_size_max_bp
    jump_var = (a * a + a * b + b * b) / 3  # E[J^2], magnitude uniform on [a, b]
    usd = config.tick_value / config.tick_size
    sigma2_hour = config.vol_annual_bp**2 / (config.trading_days_per_year * config.trading_hours_per_day)
    window = config.hedge_print_window_s
    assert config.hedge_cost_bp_at(10, window - 1) > far_cost(config, 10) == config.hedge_cost_bp_at(10, window)
    w = world.generate(config, 1)
    step = 100
    liquidation = market.liquidate(10, step, w.path.public, config).cost_bp_per_lot
    return {
        "tick_price": config.tick_size / config.bp_per_point,
        "tick_bp": config.tick_size,
        "tick_usd": config.tick_value,
        "bp_usd": usd,
        "minutes": config.session_length / 60,
        "presets": ", ".join(str(p) for p in sorted(SESSION_PRESETS)[:-1]) + f" or {sorted(SESSION_PRESETS)[-1]}",
        "close_mult": liquidation / market.cost_bp_per_lot(step, 10, w.path.public, config),
        "rate": config.rfq_rate_per_hour,
        "dealers": config.n_competitors,
        "limit": market.hedge(step, 10_000, 0, w.path.public, config).quantity,
        "base": far_cost(config, 0),
        "impact": far_cost(config, 100) - far_cost(config, 0),
        "per": 100,
        "window": window / 60,
        "widen": config.hedge_print_widen + 1.0,
        "counts": _counts(config.event_count_probs),
        "refresh": config.screen_refresh_s,
        "per_lot2": scoring.risk_kappa * usd**2 * jump_var,
        "lots_a": 50,
        "pen_a": scoring.risk_kappa * (usd * 50) ** 2 * jump_var,
        "lots_b": 100,
        "pen_b": scoring.risk_kappa * (usd * 100) ** 2 * jump_var,
        "carry_lots": 100,
        "carry_min": 10,
        "carry_usd": scoring.risk_kappa * (usd * 100) ** 2 * sigma2_hour * (10 / 60),
    }


def _counts(probs) -> str:
    ks = [WORDS[k] for k, p in enumerate(probs) if p > 0]
    return f"exactly {ks[0]}" if len(ks) == 1 else ", ".join(ks[:-1]) + " or " + ks[-1]


def expected_market(config: GameConfig) -> dict[str, float | str]:
    # The dealers' size widening is quoted per 100 lots: their half-spread at
    # 100 lots, less at none, is exactly the configured widening.
    widening = config.competitor_half_spread_bp_at(100, math.inf) - config.competitor_half_spread_bp_at(0, math.inf)
    assert widening == pytest.approx(config.competitor_size_widen_bp)
    rates = {p: SESSION_PRESETS[p].get("move_rate_per_free_hour", GameConfig.move_rate_per_free_hour) for p in SESSION_PRESETS}
    return {
        "hf": config.n_hedge_funds, "rm": config.n_real_money, "corp": config.n_corporates,
        "p_hf": config.informed_prob_hedge_fund, "p_rm": config.informed_prob_real_money,
        "p_corp": config.informed_prob_corporate,
        "accuracy": config.informed_accuracy, "noise": config.informed_signal_noise,
        "jmin": config.jump_size_min_bp, "jmax": config.jump_size_max_bp,
        **{f"p{k}": p for k, p in enumerate(config.event_count_probs)},
        "dmin": config.drift_magnitude_min_bp, "dmax": config.drift_magnitude_max_bp,
        "lmin": config.drift_duration_min_s / 60, "lmax": config.drift_duration_max_s / 60,
        "agree": config.drift_prob_agree,
        "rate": config.move_rate_per_free_hour,
        **{f"rate_{p}": r for p, r in rates.items()},
        **{f"preset_{p}": p for p in rates},
        "mmin": config.move_magnitude_min_bp, "mmax": config.move_magnitude_max_bp,
        "move_dmin": config.move_duration_min_s, "move_dmax": config.move_duration_max_s,
        "move_lmin": config.move_lead_min_s / 60, "move_lmax": config.move_lead_max_s / 60,
        "tilt": config.crowd_drift_tilt,
        "n": config.n_competitors, "p": config.competitor_participation_prob,
        "hs": config.competitor_half_spread_bp, "sw": config.competitor_size_widen_bp,
        "dealer_per": 100, "tolerance_per": 100,
        "widen": 1.0 + config.competitor_print_widen, "window": config.competitor_print_window_s / 60,
        "tol": config.client_tolerance_bp, "sd": config.client_tolerance_log_sd,
        "growth": config.client_tolerance_size_growth,
        "vol": config.vol_annual_bp,
        "hour": config.vol_annual_bp / math.sqrt(config.trading_days_per_year * config.trading_hours_per_day),
        "screen_noise": config.screen_noise_bp, "corr": config.screen_noise_corr_s,
    }


# Facts that share a name inside one item: the expected table qualifies them.
QUALIFIED = {("Unannounced moves", "dmin"): "move_dmin", ("Unannounced moves", "dmax"): "move_dmax",
             ("Unannounced moves", "lmin"): "move_lmin", ("Unannounced moves", "lmax"): "move_lmax",
             ("Volatility and the screen", "noise"): "screen_noise",
             ("Dealers", "per"): "dealer_per", ("Uninformed clients' tolerance", "per"): "tolerance_per"}


def check_items(items: list[handbook.Item], expected: dict[str, float | str]) -> int:
    """Every fact is in its item's text and equals the expected value; no
    number in the text is anything but a fact. Returns the facts checked."""
    checked = 0
    for item in items:
        text = item.text
        for name, shown in item.facts.items():
            assert shown in item.text, (item.head, name, shown)
            want = expected[QUALIFIED.get((item.head, name), name)]
            if isinstance(want, str):
                assert shown == want, (item.head, name, shown, want)
            else:
                got = value(shown)
                assert got == pytest.approx(want, rel=1e-9, abs=0.51 if shown.startswith("$") and "." not in shown else 0.006), (
                    item.head, name, shown, want)
            checked += 1
        generated = {n for shown in item.facts.values() for n in NUMBER.findall(shown)}
        typed = [n for n in NUMBER.findall(text) if n not in generated]
        assert not typed, (item.head, "numbers in the text that are not generated facts", typed)
    return checked


@pytest.mark.parametrize("config,scoring", DESKS)
def test_every_part_1_number_is_the_config_value_it_describes(config, scoring):
    """Part 1 -- your desk, exactly -- for every preset and for a desk with
    different numbers everywhere: each number in the text equals the config
    value it describes, worked out independently (the venue's actual cost
    and clip, the liquidation's multiple, the penalty from kappa and the
    print sizes), and no number appears that is not one of those. A number
    typed into the text, or one computed from the wrong field, fails on the
    perturbed desk if not before."""
    assert check_items(handbook.your_desk(config, scoring), expected_desk(config, scoring)) >= 20


def test_the_text_follows_the_config():
    """Changing the config changes the words: the default and the perturbed
    desk read differently wherever their numbers differ."""
    a = {i.head: i.text for i in handbook.your_desk(GameConfig.for_session(60), SCORING)}
    b = {i.head: i.text for i in handbook.your_desk(*PERTURBED)}
    for head in ("The product", "The session", "Clients", "Competition", "Position limit", "The hedge venue",
                 "The calendar", "The screen", "Your score"):
        assert a[head] != b[head], head
    assert "±150 lots" in b["Position limit"] and "up to 1.5× at the print" in b["The hedge venue"]


@pytest.mark.parametrize("minutes", [30, 60, 120])
def test_every_part_3_number_is_the_config_value_it_describes(minutes):
    """Part 3 -- the exact numbers -- the same way: every number generated
    from this session's config, none typed."""
    config = GameConfig.for_session(minutes)
    assert check_items(handbook.exact_numbers(config), expected_market(config)) >= 35


def test_part_3_stays_out_of_the_handbook_until_10_scored_sessions():
    """Below the threshold Part 3 is not in the message at all -- not hidden
    on the page, absent -- and the note says how many sessions so far. At
    ten it is there in full. Sending it locked and hiding it in the page
    would put the numbers one dev-tools click away."""
    config = GameConfig.for_session(60)
    exact = [i.text for i in handbook.exact_numbers(config)]
    for n in (0, 9):
        m = handbook.message(config, SCORING, scored_sessions=n)
        part3 = m["parts"][2]
        assert part3["items"] == [] and part3["note"] == f"Unlocks after 10 scored sessions ({n} so far)."
        assert not any(t in json.dumps(m) for t in exact)
    m = handbook.message(config, SCORING, scored_sessions=10)
    assert m["parts"][2]["note"] is None and [i["text"] for i in m["parts"][2]["items"]] == exact


def hosted_handbook(config: GameConfig, seed: int, entries: list[dict]) -> dict:
    storage = MemoryStorage()
    TrackRecord(entries).save(storage)
    g = game.Game(config, seed, storage=storage, settings=Settings(preset=60))
    frames: list[str] = []
    host_module.Host(g, ManualClock(), records_dir="records").connect(frames.append)
    return next(m for m in json.loads(frames[0]) if m["type"] == "handbook")


def entries(scored: int, replays: int) -> list[dict]:
    return [{"config_digest": "elsewhere", "seed": i, "replay_of_played_seed": i >= scored} for i in range(scored + replays)]


def test_the_page_gets_part_3_at_10_scored_sessions_and_replays_do_not_count():
    """Hosted: a track record of nine scored sessions and three replays
    sends the handbook locked, "9 so far"; ten scored sessions unlock it.
    A count that included replays would unlock the numbers on practice."""
    config = GameConfig.for_session(60)
    locked = hosted_handbook(config, 4, entries(9, 3))
    assert locked["scored_sessions"] == 9 and locked["parts"][2]["items"] == []
    assert "(9 so far)" in locked["parts"][2]["note"]
    unlocked = hosted_handbook(config, 4, entries(10, 3))
    assert unlocked["parts"][2]["items"] and unlocked["parts"][2]["note"] is None


def test_no_handbook_number_comes_from_the_hidden_world():
    """The handbook is a function of the config, the scoring config and a
    count -- its signature says so, and its module imports nothing that can
    build a world -- and two sessions on different seeds of the same config
    get the identical handbook. Anything drawn from a session's world (its
    prints, its clients' types, its moves) would differ between seeds."""
    assert list(inspect.signature(handbook.message).parameters) == ["config", "scoring", "scored_sessions"]
    tree = ast.parse(Path(handbook.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    reach = {"mm_game.world", "mm_game.rng", "mm_game.flow", "mm_game.fills", "mm_game.engine", "mm_game.game",
             "mm_game.records", "mm_game.host", "mm_game.market"}
    assert not imported & reach, imported & reach
    config = GameConfig.for_session(60)
    unlocked = entries(10, 0)
    a, b = hosted_handbook(config, 4, unlocked), hosted_handbook(config, 5, unlocked)
    worlds = [world.generate(config, s) for s in (4, 5)]
    assert [len(w.path.event_steps) for w in worlds] != [] and a == b
