"""The desk handbook: what the player is told before playing.

The desk exactly, the market roughly. Part 1 -- your own desk's rules,
costs and scoring -- and Part 3 -- the market's exact parameters -- are
generated here from the session's config and scoring config, never typed
by hand, so they cannot drift from the model. Part 2 is fixed desk lore,
the market as experience would describe it. Part 3 is withheld until the
track record holds enough scored sessions: until then it is not in the
handbook at all, only a line saying how many sessions to go.

Pure: built from a ``GameConfig``, a ``ScoringConfig`` and a count. It is
never handed a world or a seed, so nothing in it can come from one.

Every number in a generated item is recorded in its ``facts`` -- what it
is and how it was written -- so a test can check each against the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mm_game.config import SESSION_PRESETS, BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.fair_value import sigma_per_step
from mm_game.protocol import PROTOCOL_VERSION
from mm_game.scoring import diffusion_penalty_usd, print_penalty_usd

UNLOCK_AFTER_SCORED_SESSIONS = 10
PENALTY_EXAMPLE_LOTS = (50, 100)  # the positions the score item prices
CARRY_EXAMPLE_LOTS = 100  # and the between-prints charge, for this many lots
CARRY_EXAMPLE_S = 10 * 60.0  # held this long
IMPACT_PER_LOTS = 100  # size charges -- the venue's, the dealers', the tolerance's -- are per this many lots


@dataclass(frozen=True)
class Item:
    head: str | None
    text: str
    facts: dict[str, str] = field(default_factory=dict)  # what each number is -> as written

    def to_record(self) -> dict[str, Any]:
        return {"head": self.head, "text": self.text}


def _n(x: float) -> str:
    """A config number as written: 1.0 -> 1, 0.25 -> 0.25."""
    return f"{x:g}"


def _pct(p: float) -> str:
    return f"{p * 100:g}%"


def _usd(x: float) -> str:
    """Dollars: whole above $100 or when whole, else with cents ($12.50)."""
    return f"${x:,.0f}" if abs(x) >= 100 or x == round(x) else f"${x:,.2f}"


_COUNT_WORDS = ("none", "one", "two", "three", "four", "five")


def possible_counts(probs: tuple[float, ...]) -> str:
    """How many prints a session can have, in words: "none, one or two"."""
    ks = [_COUNT_WORDS[k] for k, p in enumerate(probs) if p > 0]
    if len(ks) == 1:
        return f"exactly {ks[0]}"
    return ", ".join(ks[:-1]) + " or " + ks[-1]


def _item(head: str | None, template: str, **facts: str) -> Item:
    return Item(head, template.format(**facts), dict(facts))


# --------------------------------------------------------------------------
# Part 1: your desk, exactly
# --------------------------------------------------------------------------


def your_desk(config: GameConfig, scoring: ScoringConfig) -> list[Item]:
    presets = sorted(SESSION_PRESETS)
    widen = 1.0 + config.hedge_print_widen
    carry_steps = round(CARRY_EXAMPLE_S / config.dt)
    return [
        _item(
            "The product",
            "One futures contract, quoted in price. One tick is {tick_price} in price, which is {tick_bp}bp. "
            "A tick is {tick_usd} a lot; a basis point, {bp_usd} a lot.",
            tick_price=_n(config.tick_size / config.bp_per_point),
            tick_bp=_n(config.tick_size),
            tick_usd=_usd(config.tick_value),
            bp_usd=_usd(config.dollars_per_bp_per_lot),
        ),
        _item(
            "The session",
            "This one runs {minutes} minutes (sessions come in {presets} minutes). You start flat. Anything you "
            "still hold at the end is liquidated at {close_mult}× the venue's cost.",
            minutes=_n(config.session_length / 60),
            presets=", ".join(str(p) for p in presets[:-1]) + f" or {presets[-1]}",
            close_mult=_n(config.close_liquidation_multiplier),
        ),
        _item(
            "Clients",
            "send RFQs, about {rate} an hour. Two-way unless the ticket says one-way, in which case it shows the side.",
            rate=_n(config.rfq_rate_per_hour),
        ),
        _item(
            "Competition",
            "Up to {dealers} other dealers may be asked; the ticket shows how many are in. The best price trades; "
            "ties are broken at random.",
            dealers=str(config.n_competitors),
        ),
        Item(
            "What each outcome tells you",
            "Won: you see the cover, the next-best price. Done away: you see the level and the client's side. "
            "Not done or passed: nothing.",
        ),
        _item(
            "Position limit",
            "±{limit} lots, hard. A quote side that would take you through it is blocked; a hedge that would is clipped.",
            limit=str(config.position_limit_lots),
        ),
        _item(
            "The hedge venue",
            "Any time, at the screen plus {base}bp a lot, plus {impact}bp a lot per {per} lots of size. Within "
            "{window} minutes of a print the cost rises, up to {widen} at the print itself. Hedges don't move "
            "the market.",
            base=_n(config.hedge_cost_bp),
            impact=_n(config.hedge_impact_bp),
            per=str(IMPACT_PER_LOTS),
            window=_n(config.hedge_print_window_s / 60),
            widen="double" if widen == 2.0 else f"{_n(widen)}×",
        ),
        _item(
            "The calendar",
            "shows the times of this session's scheduled prints ({counts}). Never their outcome.",
            counts=possible_counts(config.event_count_probs),
        ),
        _item(
            "The screen",
            "is the market's best estimate, refreshing every {refresh} seconds. It is noisy and can be off for a "
            "minute or two.",
            refresh=_n(config.screen_refresh_s),
        ),
        _item(
            "Your score",
            "is P&L in dollars. Risk-adjusted P&L sits beside it: P&L minus a charge for the risk you carried — "
            "about {per_lot2} × lots² for each print you hold through ({lots_a} lots: {pen_a}; {lots_b} lots: "
            "{pen_b}), plus a small charge for holding between prints ({carry_lots} lots for {carry_min} "
            "minutes: {carry_usd}).",
            per_lot2=f"${print_penalty_usd(config, scoring, 1):.2f}",
            lots_a=str(PENALTY_EXAMPLE_LOTS[0]),
            pen_a=_usd(print_penalty_usd(config, scoring, PENALTY_EXAMPLE_LOTS[0])),
            lots_b=str(PENALTY_EXAMPLE_LOTS[1]),
            pen_b=_usd(print_penalty_usd(config, scoring, PENALTY_EXAMPLE_LOTS[1])),
            carry_lots=str(CARRY_EXAMPLE_LOTS),
            carry_min=_n(CARRY_EXAMPLE_S / 60),
            carry_usd=_usd(diffusion_penalty_usd(config, scoring, CARRY_EXAMPLE_LOTS, carry_steps)),
        ),
        Item(
            "Benchmarks",
            "on the same world: a naive bot; a disciplined bot (public information, fixed rules — its rules are "
            "shown on every reveal); and a flow oracle (told which RFQs are informed, not what about or which way).",
        ),
        Item(
            "Fresh worlds",
            "Every scored session is a new, hidden seed. Replaying a world you have seen is allowed for practice "
            "and never scored.",
        ),
    ]


# --------------------------------------------------------------------------
# Part 2: the market, roughly -- fixed desk lore
# --------------------------------------------------------------------------

MARKET_LORE: tuple[str, ...] = (
    "Some clients know something. Hedge funds often do; real money sometimes; corporates rarely. Which clients, "
    "you find out by trading with them — and it changes every session.",
    "Clients who know are usually right, not always. When they know, they trade one way, keep coming back the "
    "same way, and often trade bigger.",
    "Prints move the market a few basis points, either way.",
    "Before a print the market sometimes drifts. The drift points the print's way more often than not, but far "
    "from reliably.",
    "Sometimes the market jumps a few basis points in under a minute with nothing on the calendar. A few clients "
    "tend to hear about it first.",
    "Most clients are just trading. In a drifting market the crowd leans with the drift.",
    "The other dealers are competent and don't learn during the day. With fewer of them in competition, clients "
    "pay you more.",
)


# --------------------------------------------------------------------------
# Part 3: the exact numbers
# --------------------------------------------------------------------------


def exact_numbers(config: GameConfig) -> list[Item]:
    probs = config.event_count_probs
    rates = {p: float(SESSION_PRESETS[p].get("move_rate_per_free_hour", GameConfig.move_rate_per_free_hour))
             for p in sorted(SESSION_PRESETS)}
    per_hour = sigma_per_step(config) * math.sqrt(3600.0 / config.dt)
    return [
        _item(
            "Clients",
            "This session's roster: {hf} hedge funds, {rm} real money, {corp} corporates. Each is informed with "
            "probability {p_hf} (hedge fund), {p_rm} (real money) or {p_corp} (corporate), fixed for the "
            "session. An informed client's view of the direction is right {accuracy} of the time, and its sense "
            "of the size has a lognormal error of log-sd {noise}.",
            hf=str(config.n_hedge_funds),
            rm=str(config.n_real_money),
            corp=str(config.n_corporates),
            p_hf=_pct(config.informed_prob_hedge_fund),
            p_rm=_pct(config.informed_prob_real_money),
            p_corp=_pct(config.informed_prob_corporate),
            accuracy=_pct(config.informed_accuracy),
            noise=_n(config.informed_signal_noise),
        ),
        _item(
            "One-way RFQs",
            "An uninformed client asks one-way {ow_u} of the time; an informed one, {ow_i}.",
            ow_u=_pct(config.one_way_prob_uninformed),
            ow_i=_pct(config.one_way_prob_informed),
        ),
        _item(
            "Informed size",
            "An informed client trades the usual size times its expected move over {ref}bp, kept between "
            "{fmin}× and {fmax}×: the bigger the move it expects, the bigger the ticket.",
            ref=_n(config.informed_size_ref_bp),
            fmin=_n(config.informed_size_factor_min),
            fmax=_n(config.informed_size_factor_max),
        ),
        _item(
            "Prints",
            "{jmin}–{jmax}bp, uniform, either sign equally. Prints in a session like this one: "
            + ", ".join(f"{_COUNT_WORDS[k]} {{p{k}}}" for k in range(len(probs)))
            + ".",
            jmin=_n(config.jump_size_min_bp),
            jmax=_n(config.jump_size_max_bp),
            **{f"p{k}": _pct(p) for k, p in enumerate(probs)},
        ),
        _item(
            "Run-ups",
            "Before every print the market drifts {dmin}–{dmax}bp over the last {lmin}–{lmax} minutes; the drift "
            "points the print's way with probability {agree}.",
            dmin=_n(config.drift_magnitude_min_bp),
            dmax=_n(config.drift_magnitude_max_bp),
            lmin=_n(config.drift_duration_min_s / 60),
            lmax=_n(config.drift_duration_max_s / 60),
            agree=_pct(config.drift_prob_agree),
        ),
        _item(
            "Unannounced moves",
            "{rate} an hour of time clear of prints in this session ("
            + ", ".join(f"{{preset_{p}}} minutes: {{rate_{p}}}" for p in rates)
            + "); {mmin}–{mmax}bp over {dmin}–{dmax} seconds, either sign; informed clients hear {lmin}–{lmax} "
            "minutes ahead.",
            rate=_n(config.move_rate_per_free_hour),
            mmin=_n(config.move_magnitude_min_bp),
            mmax=_n(config.move_magnitude_max_bp),
            dmin=_n(config.move_duration_min_s),
            dmax=_n(config.move_duration_max_s),
            lmin=_n(config.move_lead_min_s / 60),
            lmax=_n(config.move_lead_max_s / 60),
            **{f"rate_{p}": _n(r) for p, r in rates.items()},
            **{f"preset_{p}": str(p) for p in rates},
        ),
        _item(
            "The crowd",
            "Inside a run-up or a move's ramp, uninformed clients trade with the drift {tilt} of the time; "
            "otherwise it is a coin toss.",
            tilt=_pct(config.crowd_drift_tilt),
        ),
        _item(
            "Dealers",
            "{n} of them, each asked independently with probability {p}. Half-spread {hs}bp plus {sw}bp per {per} "
            "lots, up to {widen}× within {window} minutes of a print, rounded to the tick.",
            n=str(config.n_competitors),
            p=_pct(config.competitor_participation_prob),
            hs=_n(config.competitor_half_spread_bp),
            sw=_n(config.competitor_size_widen_bp),
            per=str(IMPACT_PER_LOTS),
            widen=_n(1.0 + config.competitor_print_widen),
            window=_n(config.competitor_print_window_s / 60),
        ),
        _item(
            "Uninformed clients' tolerance",
            "An uninformed client accepts up to the screen plus a tolerance: median {tol}bp, lognormal with "
            "log-sd {sd}, growing by {growth} per {per} lots of size.",
            tol=_n(config.client_tolerance_bp),
            sd=_n(config.client_tolerance_log_sd),
            growth=_pct(config.client_tolerance_size_growth),
            per=str(IMPACT_PER_LOTS),
        ),
        _item(
            "Volatility and the screen",
            "Volatility {vol}bp a year, about {hour}bp an hour. The screen's error has a standard deviation of "
            "{noise}bp and a correlation time of {corr} seconds.",
            vol=_n(config.vol_annual_bp),
            hour=f"{per_hour:.2f}",
            noise=_n(config.screen_noise_bp),
            corr=_n(config.screen_noise_corr_s),
        ),
    ]


# --------------------------------------------------------------------------
# The message
# --------------------------------------------------------------------------


def message(config: GameConfig, scoring: ScoringConfig, *, scored_sessions: int) -> dict[str, Any]:
    """The handbook as one protocol message. Part 3 is only in it once the
    track record holds ``UNLOCK_AFTER_SCORED_SESSIONS`` scored sessions."""
    unlocked = scored_sessions >= UNLOCK_AFTER_SCORED_SESSIONS
    return {
        "v": PROTOCOL_VERSION,
        "type": "handbook",
        "scored_sessions": scored_sessions,
        "unlock_after": UNLOCK_AFTER_SCORED_SESSIONS,
        "parts": [
            {"title": "Your desk", "subtitle": "exact", "note": None,
             "items": [i.to_record() for i in your_desk(config, scoring)]},
            {"title": "The market", "subtitle": "desk lore, rough", "note": None,
             "items": [{"head": None, "text": t} for t in MARKET_LORE]},
            {
                "title": "The exact numbers",
                "subtitle": "from the model",
                "note": None if unlocked else (
                    f"Unlocks after {UNLOCK_AFTER_SCORED_SESSIONS} scored sessions ({scored_sessions} so far)."
                ),
                "items": [i.to_record() for i in exact_numbers(config)] if unlocked else [],
            },
        ],
    }


# --------------------------------------------------------------------------
# The benchmark bots' rules, shown on every reveal
# --------------------------------------------------------------------------


def disciplined_rules(benchmark: BenchmarkConfig, config: GameConfig) -> list[Item]:
    """The disciplined bot, rule by rule, from its config: what "good
    habits on public information" means in numbers."""
    widths = benchmark.disciplined_half_width_bp[: config.n_competitors + 1]
    by_count = ", ".join(
        ("{w0}bp with no other dealer in competition" if k == 0 else f"{{w{k}}}bp with {_COUNT_WORDS[k]}")
        for k in range(len(widths))
    )
    into = "It quotes {pre}bp wider in the {pre_window} minutes before a print"
    facts: dict[str, str] = {
        "pre": _n(benchmark.disciplined_pre_print_widen_bp),
        "pre_window": _n(benchmark.disciplined_pre_print_window_s / 60),
        "lead": _n(benchmark.disciplined_hedge_lead_s / 60),
        "carry": str(benchmark.disciplined_print_carry_lots),
    }
    if benchmark.disciplined_print_widen_bp > 0:
        into += ", and {at_print}bp wider within {at_window} minutes of one"
        facts.update(
            at_print=_n(benchmark.disciplined_print_widen_bp),
            at_window=_n(benchmark.disciplined_print_window_s / 60),
        )
    return [
        _item(
            "Quotes",
            "Two prices around the screen: half-width " + by_count + ", plus {size}bp per {per} lots of size.",
            **{f"w{k}": _n(w) for k, w in enumerate(widths)},
            size=_n(benchmark.disciplined_size_widen_bp),
            per=str(IMPACT_PER_LOTS),
        ),
        _item("Into prints", into + "; {lead} minutes before each print it hedges down to ±{carry} lots.", **facts),
        _item(
            "Skew",
            "It moves its mid {skew}bp per {per} lots against its position.",
            skew=_n(benchmark.disciplined_skew_bp_per_100_lots),
            per=str(IMPACT_PER_LOTS),
        ),
        _item(
            "Inventory",
            "Whenever a fill takes it past ±{cap} lots, it hedges straight back to ±{target}.",
            cap=str(benchmark.disciplined_inventory_cap_lots),
            target=str(benchmark.disciplined_inventory_target_lots),
        ),
        _item(
            "The close",
            "It stops quoting and goes flat {close} minutes before the end.",
            close=_n(benchmark.disciplined_close_lead_s / 60),
        ),
        Item("What it knows", "Only what you see: the screen, the calendar, the ticket. It never knows which "
                              "clients are informed."),
    ]


def oracle_rules(benchmark: BenchmarkConfig) -> list[Item]:
    return [
        _item(
            "The flow oracle",
            "is the disciplined bot told which RFQs are informed — not about what, not which way. It quotes them "
            "{widen}bp wider, and once it learns an informed client's side it leans its mid {lean}bp that way for "
            "{lean_min} minutes; it drops the run-up widening, and its hedges are the disciplined bot's.",
            widen=_n(benchmark.oracle_informed_widen_bp),
            lean=_n(benchmark.oracle_lean_bp),
            lean_min=_n(benchmark.oracle_lean_s / 60),
        )
    ]
