"""Scoring: the live risk meter, and everything computed after the session.

Two halves with different rights.

**Live** (``RiskMeter``, ``print_risk``): public inputs only -- the config,
the print calendar and the positions held. Shown during play.

**Post-session** (everything else): reads the truth. The headline score
needs none -- after the closing liquidation the book is flat and P&L is
realised cash -- but the decomposition and the diagnostics are defined
against true fair value, and are only ever computed once the session is
over.

The risk penalty is ex ante. It charges the expected variance of the
positions held, from the model's known vol and print-size distribution, not
what the prints happened to do: two players holding identical positions get
identical penalties whichever way the prints landed. It penalises risk taken,
not luck. It is a pure function of the positions, so changing kappa re-scores
every past session from its replay file.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from mm_game.book import Source, Trade
from mm_game.config import GameConfig, ScoringConfig
from mm_game.fair_value import FairValuePath, sigma_per_step

# --------------------------------------------------------------------------
# Live: public inputs only
# --------------------------------------------------------------------------


def print_variance_bp2(config: GameConfig) -> float:
    """E[J^2] for a print: sign symmetric, magnitude uniform on [a, b], so
    (a^2 + ab + b^2) / 3 -- 9.33bp^2 at 2-4bp."""
    a, b = config.jump_size_min_bp, config.jump_size_max_bp
    return (a * a + a * b + b * b) / 3.0


def print_penalty_usd(config: GameConfig, scoring: ScoringConfig, position: int) -> float:
    """The penalty for carrying ``position`` through one print."""
    dollars = config.dollars_per_bp_per_lot * position
    return scoring.risk_kappa * dollars * dollars * print_variance_bp2(config)


def diffusion_penalty_usd(config: GameConfig, scoring: ScoringConfig, position: int, steps: int) -> float:
    """The penalty for carrying ``position`` over ``steps`` grid steps."""
    dollars = config.dollars_per_bp_per_lot * position
    return scoring.risk_kappa * dollars * dollars * sigma_per_step(config) ** 2 * steps


class RiskMeter:
    """The risk penalty accrued so far, advanced with the clock.

    The position held over the increment into step k+1 is the position after
    every action at step k, and a print at step e lands on the increment into
    e -- so moving the clock from s to t charges the current position for
    t - s steps of diffusion and for every print in (s, t]. Actions at a step
    are applied before the clock leaves it, which is exactly that order.
    """

    def __init__(self, config: GameConfig, scoring: ScoringConfig, print_steps: Sequence[int]) -> None:
        self._config = config
        self._scoring = scoring
        self._prints = [int(e) for e in print_steps]
        self._step = 0
        self._next_print = 0
        self._diffusion = 0.0
        self._prints_usd = 0.0

    @property
    def penalty_usd(self) -> float:
        return self._diffusion + self._prints_usd

    def advance(self, to_step: int, position: int) -> None:
        if to_step < self._step:
            raise ValueError(f"risk meter cannot go back from {self._step} to {to_step}")
        self._diffusion += diffusion_penalty_usd(self._config, self._scoring, position, to_step - self._step)
        while self._next_print < len(self._prints) and self._prints[self._next_print] <= to_step:
            self._prints_usd += print_penalty_usd(self._config, self._scoring, position)
            self._next_print += 1
        self._step = to_step


def print_risk(
    config: GameConfig, scoring: ScoringConfig, print_steps: Sequence[int], step: int, position: int
) -> dict[str, Any] | None:
    """The print-risk gauge: the next print, how far off, and what carrying
    the current position through it would cost. None after the last print."""
    later = [(i, int(e)) for i, e in enumerate(print_steps) if e > step]
    if not later:
        return None
    i, e = later[0]
    return {
        "print_index": i,
        "step": e,
        "in_s": (e - step) * config.dt,
        "penalty_if_held_usd": print_penalty_usd(config, scoring, position),
    }


# --------------------------------------------------------------------------
# Post-session: positions, penalty, decomposition, mark-outs
# --------------------------------------------------------------------------


def position_path(trades: Sequence[Trade], n_steps: int) -> np.ndarray:
    """Position held over each increment k -> k+1, for k in 0..n-1: the
    position after every trade at step k or earlier."""
    held = np.zeros(n_steps, dtype=np.int64)
    position = 0
    last = 0
    for t in sorted(trades, key=lambda t: t.seq):
        if t.step >= n_steps:
            break  # the closing liquidation: no increment follows it
        held[last : t.step] = position
        position = t.position_after
        last = t.step
    held[last:] = position
    return held


def ex_ante_penalty_usd(
    config: GameConfig, scoring: ScoringConfig, print_steps: Sequence[int], trades: Sequence[Trade]
) -> dict[str, float]:
    """The risk penalty from the blotter alone: rescorable at any kappa."""
    held = position_path(trades, config.n_steps)
    dollars2 = (config.dollars_per_bp_per_lot * held.astype(float)) ** 2
    diffusion = scoring.risk_kappa * float(dollars2.sum()) * sigma_per_step(config) ** 2
    prints = sum(
        print_penalty_usd(config, scoring, int(held[int(e) - 1])) for e in print_steps
    )
    return {"diffusion_usd": diffusion, "prints_usd": prints, "total_usd": diffusion + prints}


def _fill_markouts(config: GameConfig, path: FairValuePath, step: int, quantity: int) -> dict[str, Any]:
    """Signed mark-out P&L of one fill, bp-lots, at each horizon and to just
    after the next print. Truncated horizons are flagged; the event mark-out
    is None when no print follows -- undefined, not zero."""
    out: dict[str, Any] = {}
    for h_s, m in zip(config.markout_horizons_s, path.markout(step, config.markout_horizons_steps)):
        out[f"{h_s:g}s"] = {"pnl_bp": quantity * m.change, "truncated": m.truncated}
    ev = path.event_markout(step)
    out["event"] = {
        "pnl_bp": None if ev.change is None else quantity * ev.change,
        "print_index": ev.event_index,
    }
    return out


def fill_markouts(config: GameConfig, path: FairValuePath, trades: Sequence[Trade]) -> dict[int, dict[str, Any]]:
    """Mark-outs for every client fill, by blotter sequence number."""
    return {
        t.seq: _fill_markouts(config, path, t.step, t.quantity)
        for t in trades
        if t.source is Source.CLIENT
    }


def horizon_keys(config: GameConfig) -> list[str]:
    return [f"{h:g}s" for h in config.markout_horizons_s] + ["event"]


def decomposition(
    config: GameConfig, path: FairValuePath, screen_at, trades: Sequence[Trade]
) -> dict[str, Any]:
    """The truth-based P&L decomposition, in dollars, at every horizon.

    Three independently computed terms sum exactly to the session's P&L:

        total = spread captured + inventory P&L - hedge and liquidation costs

    - spread captured: every client fill's edge against the truth at the
      fill, q * (V_t - p);
    - inventory P&L: what the truth did to every position held,
      sum over trades of q * (V_close - V_t);
    - hedge and liquidation costs: what the venue and the close cost against
      the truth, split into the venue's charge against the screen and the
      screen's own error at the moment of the hedge.

    Adverse selection is then carved out of inventory, per horizon: the
    mark-out of each fill over that horizon, sign-flipped so a positive
    number is money lost to the flow. What remains is inventory P&L beyond
    the fills' own mark-outs -- what the position did after the information
    in each fill had played out. So each horizon's line reads

        total = spread - adverse selection + inventory beyond it - costs

    and sums to the cent. Fills whose mark-out is truncated by the close, or
    has no print after it (event-relative), are excluded from adverse
    selection and counted; their moves stay in inventory, so nothing is lost.
    """
    usd = config.dollars_per_bp_per_lot
    level = config.start_level
    v = path.relative
    close = float(v[-1])
    spread = inventory = venue = screen_error = 0.0
    cash = 0.0
    for t in trades:
        truth = float(v[t.step])
        p = t.price - level
        cash -= t.quantity * p
        inventory += t.quantity * (close - truth)
        if t.source is Source.CLIENT:
            spread += t.quantity * (truth - p)
        else:
            venue += abs(t.quantity) * t.cost_bp_per_lot
            screen_error += t.quantity * ((screen_at(t.step) - level) - truth)
    costs = venue + screen_error
    marks = fill_markouts(config, path, trades)
    by_horizon = {}
    for key in horizon_keys(config):
        adverse = 0.0
        excluded = 0
        for m in marks.values():
            cell = m[key]
            if cell["pnl_bp"] is None or cell.get("truncated", False):
                excluded += 1
                continue
            adverse -= cell["pnl_bp"]
        by_horizon[key] = {
            "adverse_selection_usd": adverse * usd,
            "inventory_beyond_usd": (inventory + adverse) * usd,
            "fills_excluded": excluded,
        }
    return {
        "total_usd": cash * usd,
        "spread_captured_usd": spread * usd,
        "inventory_usd": inventory * usd,
        "hedge_and_close_costs_usd": costs * usd,
        "venue_charge_usd": venue * usd,
        "screen_error_on_hedges_usd": screen_error * usd,
        "by_horizon": by_horizon,
    }


def capture_ratio(player: float, disciplined: float, oracle: float, min_gap: float) -> float | None:
    """(player - disciplined) / (oracle - disciplined): the share of the
    flow-reading edge above good habits that the player took. None -- shown
    as n/a -- when the oracle's edge over the disciplined bot is under
    ``min_gap``: with little informed flow there is nothing to capture, and
    the ratio of two noise terms explodes."""
    gap = oracle - disciplined
    if not gap >= min_gap:
        return None
    return (player - disciplined) / gap


# --------------------------------------------------------------------------
# Post-session diagnostics
# --------------------------------------------------------------------------

# A quote mid at least this far from the truth, on the same side for
# consecutive quotes, counts toward a sustained excursion: two ticks, beyond
# what the screen's own error would explain on its own most of the time.
EXCURSION_BP = 0.5


def _mean(xs: Sequence[float]) -> float | None:
    return float(np.mean(xs)) if len(xs) else None


def _sd(xs: Sequence[float]) -> float | None:
    return float(np.std(xs)) if len(xs) > 1 else None


def rfq_table(events) -> dict[int, dict[str, Any]]:
    """Per RFQ shown, what the event log says: arrival, the quote, the outcome."""
    table: dict[int, dict[str, Any]] = {}
    for e in events:
        if e.kind.value == "rfq_arrived":
            table[e.data["rfq"]["ordinal"]] = {"step": e.step, "rfq": e.data["rfq"]}
        elif e.kind.value == "quote_sent":
            table[e.data["rfq_ordinal"]]["quote"] = e.data
        elif e.kind.value == "limit_blocked":
            table[e.data["rfq_ordinal"]]["blocked"] = e.data["sides"]
        elif e.kind.value == "rfq_resolved":
            table[e.data["rfq_ordinal"]]["resolved"] = e.data
    return table


def _excursion(points: list[tuple[int, float]], dt: float) -> dict[str, Any] | None:
    """The longest run of consecutive quotes off the truth by EXCURSION_BP or
    more on the same side: when it started, how long it lasted, by how much."""
    best = None
    run: list[tuple[int, float]] = []
    for step, err in points + [(None, 0.0)]:
        same = run and step is not None and abs(err) >= EXCURSION_BP and (err > 0) == (run[-1][1] > 0)
        if same:
            run.append((step, err))
            continue
        if len(run) >= 2:  # one quote off the truth is a miss, not a sustained excursion
            duration = (run[-1][0] - run[0][0]) * dt
            if best is None or duration > best["duration_s"] or (
                duration == best["duration_s"] and len(run) > best["quotes"]
            ):
                best = {
                    "start_step": run[0][0],
                    "end_step": run[-1][0],
                    "duration_s": duration,
                    "quotes": len(run),
                    "mean_error_bp": float(np.mean([x for _, x in run])),
                }
        run = [(step, err)] if step is not None and abs(err) >= EXCURSION_BP else []
    return best


def diagnostics(world, events, trades: Sequence[Trade]) -> dict[str, Any]:
    """The derived diagnostics that turn a log into a review conversation.
    Reads the truth: post-session only."""
    config = world.config
    path = world.path
    view = path.public
    usd = config.dollars_per_bp_per_lot
    dt = config.dt
    rfqs = world.schedule.rfqs
    table = rfq_table(events)

    # Outcomes and hit rates, overall and by count in competition.
    outcomes: dict[str, int] = {}
    by_comp: dict[int, dict[str, int]] = {}
    left_on_table, how_far_off = [], []
    for ordinal, row in table.items():
        res = row.get("resolved")
        if res is None:
            continue
        outcomes[res["outcome"]] = outcomes.get(res["outcome"], 0) + 1
        comp = by_comp.setdefault(row["rfq"]["in_comp"], {"quoted": 0, "won": 0})
        if res["outcome"] != "passed":
            comp["quoted"] += 1
            comp["won"] += res["outcome"] == "won"
        sign = res["side"]
        q = row.get("quote", {})
        mine = q.get("ask") if sign == 1 else q.get("bid")
        if res["outcome"] == "won" and res["cover"] is not None:
            left_on_table.append(sign * (res["cover"] - res["price"]))
        if res["outcome"] == "done_away" and mine is not None:
            how_far_off.append(sign * (mine - res["cover"]))
    quoted = sum(v for k, v in outcomes.items() if k != "passed")

    # Mark-outs and volume, informed against uninformed.
    marks = fill_markouts(config, path, trades)
    buckets: dict[str, dict[str, Any]] = {
        b: {"fills": 0, "lots": 0, "spread_captured_usd": 0.0, "markouts": {}} for b in ("informed", "uninformed")
    }
    for t in trades:
        if t.source is not Source.CLIENT:
            continue
        rfq = rfqs[t.rfq_ordinal]
        b = buckets["informed" if rfq.informed else "uninformed"]
        b["fills"] += 1
        b["lots"] += abs(t.quantity)
        b["spread_captured_usd"] += t.quantity * (path.values[t.step] - t.price) * usd
        for key, cell in marks[t.seq].items():
            m = b["markouts"].setdefault(key, {"pnl_usd": 0.0, "lots": 0, "fills": 0, "excluded": 0})
            if cell["pnl_bp"] is None or cell.get("truncated", False):
                m["excluded"] += 1
                continue
            m["pnl_usd"] += cell["pnl_bp"] * usd
            m["lots"] += abs(t.quantity)
            m["fills"] += 1
    for b in buckets.values():
        for m in b["markouts"].values():
            m["per_lot_bp"] = m["pnl_usd"] / usd / m["lots"] if m["lots"] else None
    total_lots = buckets["informed"]["lots"] + buckets["uninformed"]["lots"]

    # Quote mid against the truth and against the screen.
    vs_truth, vs_screen = [], []
    for ordinal, row in sorted(table.items()):
        q = row.get("quote")
        if q is None or q["bid"] is None or q["ask"] is None:
            continue
        mid = 0.5 * (q["bid"] + q["ask"])
        vs_truth.append((row["step"], mid - float(path.values[row["step"]])))
        vs_screen.append((row["step"], mid - view.screen_at(row["step"])))

    def tracking(points):
        errs = [e for _, e in points]
        return {
            "quotes": len(errs),
            "mean_bp": _mean(errs),
            "sd_bp": _sd(errs),
            "mean_abs_bp": _mean([abs(e) for e in errs]),
            "worst_sustained_excursion": _excursion(points, dt),
        }

    # Position through time, and into each print.
    held = position_path(trades, config.n_steps)
    prints = []
    mids = {}
    for ordinal, row in table.items():
        q = row.get("quote")
        if q and q["bid"] is not None and q["ask"] is not None:
            mids[row["step"]] = 0.5 * (q["bid"] + q["ask"])
    mid_steps = sorted(mids)
    for i, e in enumerate(int(x) for x in path.event_steps):
        window = next((w for w in path.drift_windows if w.event_index == i), None)
        jump = float(path.jump_sizes[i])
        before = [s for s in mid_steps if s < e]
        after = [s for s in mid_steps if s >= e]
        reaction = None
        if before:
            m0 = mids[before[-1]]
            for s in (s for s in mid_steps if s >= e):
                if (mids[s] - m0) * math.copysign(1.0, jump) >= abs(jump) / 2:
                    reaction = {"steps": s - e, "seconds": (s - e) * dt}
                    break
        prints.append(
            {
                "print_index": i,
                "step": e,
                "jump_bp": jump,
                "position_into_print": int(held[e - 1]),
                "drift_bp": None if window is None else window.total_bp,
                "drift_agreed_with_print": None if window is None else (window.total_bp > 0) == (jump > 0),
                "position_with_print": int(np.sign(held[e - 1]) * np.sign(jump)),
                "two_sided_quotes_after_print": len(after),
                "quotes_reflected_print_after": reaction,
            }
        )

    # Clients.
    clients = {}
    for c in world.schedule.clients:
        clients[c.client_id] = {
            "category": c.category.value,
            "informed_type": c.informed_type,
            "rfqs": 0,
            "trades": 0,
            "lots": 0,
            "pnl_300s_usd": 0.0,
        }
    for ordinal, row in table.items():
        clients[row["rfq"]["client_id"]]["rfqs"] += 1
    for t in trades:
        if t.source is Source.CLIENT:
            c = clients[rfqs[t.rfq_ordinal].client_id]
            c["trades"] += 1
            c["lots"] += abs(t.quantity)
            cell = marks[t.seq].get("300s")
            if cell is not None and not cell["truncated"]:
                c["pnl_300s_usd"] += cell["pnl_bp"] * usd

    return {
        "outcomes": outcomes,
        "hit_rate": outcomes.get("won", 0) / quoted if quoted else None,
        "hit_rate_by_comp": {
            str(k): (v["won"] / v["quoted"] if v["quoted"] else None) for k, v in sorted(by_comp.items())
        },
        "cover": {
            "won_left_on_table_bp": _mean(left_on_table),
            "done_away_how_far_off_bp": _mean(how_far_off),
        },
        "flow": {
            **buckets,
            "informed_share_of_volume": buckets["informed"]["lots"] / total_lots if total_lots else None,
        },
        "quote_vs_truth": tracking(vs_truth),
        "quote_vs_screen": tracking(vs_screen),
        "position": {
            "max_abs": int(np.abs(held).max()) if held.size else 0,
            "time_weighted_mean_abs": float(np.abs(held).mean()) if held.size else 0.0,
            "into_prints": prints,
        },
        "clients": {str(k): v for k, v in clients.items()},
    }


def nan_to_none(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
