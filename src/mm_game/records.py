"""Session records: the replay file, the coaching record, the track record.

**Replay file** -- small: the provenance header (config, seed, scheme and
config-schema versions, numpy version, canary), the session settings, the
scoring and benchmark configs, flags, and every accepted input with the step
it was stamped at. Never wall time. Everything else regenerates from the
seed, so replaying the file reproduces the session -- and its coaching
record -- bit for bit. A loader refuses anything it cannot read faithfully:
an unknown format or config-schema version, a config whose fields differ, or
a canary showing numpy's transforms have moved.

**Coaching record** -- self-describing JSON for a review: the header, the
public event log, all four layers for every RFQ (what was observable, what
was done, what happened, the hidden truth), the hidden layer for screen and
print events, scores, the truth-based decomposition, diagnostics, and the
benchmark bots' results and action logs. Plain scalars and lists only, so it
round-trips through JSON exactly. Built after the session from the world;
nothing here runs during play.

**Track record** -- one line per session across sessions. A session on a
``(config, seed)`` already played is flagged as a replay and excluded from
the aggregates: you remember where the prints went.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mm_game import rng as rng_module
from mm_game import scoring as sc
from mm_game.book import Source
from mm_game.config import CONFIG_SCHEMA_VERSION, BenchmarkConfig, GameConfig, ScoringConfig
from mm_game.engine import Session, Settings, open_session
from mm_game.events import Kind
from mm_game.fills import from_ticks
from mm_game.protocol import PROTOCOL_VERSION
from mm_game.world import World

REPLAY_FORMAT = "mm_game.replay"
RECORD_FORMAT = "mm_game.coaching_record"
TRACK_FORMAT = "mm_game.track_record"
FORMAT_VERSION = 1


class RecordError(ValueError):
    """A record this code cannot read faithfully. Refused, never guessed."""


def plain(obj: Any) -> Any:
    """Plain JSON types only: numpy scalars to Python, tuples to lists,
    non-finite floats refused. A record that only serialises by luck is a
    record that does not round-trip."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            raise RecordError(f"non-finite number {x} in a record")
        return x
    if obj is None or isinstance(obj, str):
        return obj
    raise RecordError(f"{type(obj).__name__} is not a record type")


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, allow_nan=False, indent=1)


def config_digest(config: GameConfig) -> str:
    """Identity of a scenario: same digest, same world for the same seed."""
    text = json.dumps(config.to_record(), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Replay file
# --------------------------------------------------------------------------


def flags(session: Session, replay_of_played_seed: bool) -> dict[str, Any]:
    return {"replay_of_played_seed": replay_of_played_seed, "ended_early_at_step": session.quit_step}


def replay_file(
    world: World, session: Session, *, scoring: ScoringConfig, benchmark: BenchmarkConfig, flags: dict[str, Any]
) -> dict[str, Any]:
    return plain(
        {
            "format": REPLAY_FORMAT,
            "format_version": FORMAT_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "header": world.header,
            "settings": session.settings.to_record(),
            "scoring_config": scoring.to_record(),
            "benchmark_config": benchmark.to_record(),
            "flags": flags,
            "actions": session.actions,
        }
    )


@dataclass(frozen=True)
class ReplayFile:
    config: GameConfig
    seed: int
    settings: Settings
    scoring: ScoringConfig
    benchmark: BenchmarkConfig
    flags: dict[str, Any]
    actions: list[dict[str, Any]]


def check_header(header: dict[str, Any]) -> tuple[GameConfig, int]:
    """Refuse a header this code would misread: another stream scheme,
    another config schema, or numpy transforms that no longer give the
    recorded canary."""
    if header.get("scheme_version") != rng_module.SCHEME_VERSION:
        raise RecordError(f"stream scheme {header.get('scheme_version')!r}; this code is {rng_module.SCHEME_VERSION}")
    if header.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise RecordError(
            f"config schema {header.get('config_schema_version')!r}; this code reads {CONFIG_SCHEMA_VERSION}"
        )
    if header.get("canary") != rng_module.canary():
        moved = [k for k, v in rng_module.canary().items() if header.get("canary", {}).get(k) != v]
        raise RecordError(f"numpy's random transforms have changed since this was recorded: {moved}")
    try:
        config = GameConfig.from_record(header["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise RecordError(f"unreadable config: {error}") from None
    return config, int(header["seed"])


def load_replay(record: dict[str, Any]) -> ReplayFile:
    if record.get("format") != REPLAY_FORMAT or record.get("format_version") != FORMAT_VERSION:
        raise RecordError(f"not a version-{FORMAT_VERSION} replay file: {record.get('format')!r} {record.get('format_version')!r}")
    if record.get("protocol_version") != PROTOCOL_VERSION:
        raise RecordError(f"protocol {record.get('protocol_version')!r}; this engine speaks {PROTOCOL_VERSION}")
    config, seed = check_header(record["header"])
    try:
        settings = Settings(**record["settings"])
        scoring = ScoringConfig.from_record(record["scoring_config"])
        benchmark = BenchmarkConfig.from_record(record["benchmark_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise RecordError(f"unreadable settings: {error}") from None
    return ReplayFile(config, seed, settings, scoring, benchmark, dict(record["flags"]), list(record["actions"]))


def drive(session: Session, actions: list[dict[str, Any]]) -> None:
    """Re-apply recorded inputs at their steps, then run to the close. Any
    disagreement -- an RFQ the file never answered, a refused input -- is a
    replay that does not reproduce, and raises."""
    session.start()
    for a in actions:
        if session.ended:
            raise RecordError(f"session ended before action {a['seq']}")
        if session.step < a["step"]:
            _expect_ok(session.handle({"v": PROTOCOL_VERSION, "type": "advance", "to_step": a["step"]}), a)
        if session.step != a["step"]:
            raise RecordError(f"action {a['seq']} is at step {a['step']}, but the clock stopped at {session.step}")
        params = {k: v for k, v in a.items() if k not in ("seq", "step", "type")}
        _expect_ok(session.handle({"v": PROTOCOL_VERSION, "type": a["type"], **params}), a)
    if not session.ended:
        _expect_ok(
            session.handle({"v": PROTOCOL_VERSION, "type": "advance", "to_step": session.n_steps}),
            None,
        )
    if not session.ended:
        raise RecordError(f"replay stopped at step {session.step}: an RFQ the file never answered")


def _expect_ok(messages: list[dict[str, Any]], action: dict[str, Any] | None) -> None:
    for m in messages:
        if m["type"] == "error":
            raise RecordError(f"replay refused at {action}: {m['detail']}")


def replay(rf: ReplayFile, *, scoring: ScoringConfig | None = None) -> tuple[Session, World]:
    """Reproduce a recorded session. Pass ``scoring`` to re-score it."""
    session, world = open_session(
        rf.config,
        rf.seed,
        settings=rf.settings,
        scoring=scoring or rf.scoring,
        replay_of_played_seed=rf.flags["replay_of_played_seed"],
    )
    drive(session, rf.actions)
    return session, world


# --------------------------------------------------------------------------
# Coaching record
# --------------------------------------------------------------------------


def scores(world: World, session: Session, scoring: ScoringConfig) -> dict[str, Any]:
    """Headline P&L is realised cash after the closing liquidation; the risk
    penalty is recomputed from the blotter, so any kappa re-scores."""
    pnl = session.book.usd(session.book.cash_bp())
    penalty = sc.ex_ante_penalty_usd(world.config, scoring, world.path.event_steps, session.book.trades)
    return {
        "pnl_usd": pnl,
        "risk_penalty_usd": penalty["total_usd"],
        "risk_penalty_diffusion_usd": penalty["diffusion_usd"],
        "risk_penalty_prints_usd": penalty["prints_usd"],
        "risk_adjusted_usd": pnl - penalty["total_usd"],
    }


def _market_side(world: World, ordinal: int) -> dict[str, Any]:
    cfg = world.config
    rfq, side = world.schedule.rfqs[ordinal], world.sides[ordinal]
    dealers = [
        {
            "in_comp": asked,
            "bid": from_ticks(b, cfg),
            "ask": from_ticks(a, cfg),
            "mid": m,
        }
        for asked, b, a, m in zip(rfq.dealers_in_comp, side.competitor_bid_ticks, side.competitor_ask_ticks, side.competitor_mids)
    ]
    return {"dealers": dealers, "client_limit": side.client_limit, "tie_break_u": side.tie_break_u}


def rfq_layers(world: World, session: Session) -> list[dict[str, Any]]:
    """All four layers for every RFQ: observable state when it arrived, the
    action, the outcome, the hidden truth. RFQs after a quit were never shown
    and carry only the hidden layer."""
    cfg = world.config
    level = cfg.start_level
    usd = cfg.dollars_per_bp_per_lot
    path = world.path
    view = path.public
    trades_by_rfq = {t.rfq_ordinal: t for t in session.book.trades if t.source is Source.CLIENT}
    marks = sc.fill_markouts(cfg, path, session.book.trades)

    position, cash = 0, 0.0
    screen_mid, screen_step = None, None
    recent: list[dict[str, Any]] = []
    client_seen: dict[int, list[int]] = {}
    observable: dict[int, dict[str, Any]] = {}
    actions: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, dict[str, Any]] = {}
    for e in session.log:
        k, d = e.kind, e.data
        if k is Kind.SCREEN_UPDATE:
            screen_mid, screen_step = d["mid"], e.step
        elif k is Kind.RFQ_ARRIVED:
            r = d["rfq"]
            seen = client_seen.setdefault(r["client_id"], [0, 0])
            nxt = view.steps_to_next_event(e.step)
            observable[r["ordinal"]] = {
                "screen_mid": screen_mid,
                "screen_age_s": (e.step - screen_step) * cfg.dt,
                "seconds_to_next_print": None if nxt is None else nxt * cfg.dt,
                "position": position,
                "pnl_usd": (cash + position * (screen_mid - level)) * usd,
                "client": r,
                "client_history": {"rfqs": seen[0], "trades": seen[1]},
                "recent_outcomes": list(recent[-5:]),
            }
            seen[0] += 1
        elif k is Kind.QUOTE_SENT:
            actions[d["rfq_ordinal"]] = {**d, "blocked_sides": []}
        elif k is Kind.LIMIT_BLOCKED:
            pass  # attached below from the book's block list
        elif k is Kind.RFQ_RESOLVED:
            outcomes[d["rfq_ordinal"]] = d
            recent.append({"rfq_ordinal": d["rfq_ordinal"], "outcome": d["outcome"], "side": d["side"], "level": d["cover"]})
            if d["outcome"] == "won":
                position = d["position_after"]
                cash -= d["quantity"] * (d["price"] - level)
                client_seen[world.schedule.rfqs[d["rfq_ordinal"]].client_id][1] += 1
        elif k in (Kind.HEDGE_EXECUTED, Kind.CLOSE_LIQUIDATION) and d["quantity"]:
            position += d["quantity"]
            cash -= d["quantity"] * (d["price"] - level)
    for b in session.book.blocks:
        actions[b.rfq_ordinal]["blocked_sides"] = list(b.sides)

    layers = []
    for rfq in world.schedule.rfqs:
        t = trades_by_rfq.get(rfq.ordinal)
        layers.append(
            {
                "ordinal": rfq.ordinal,
                "step": rfq.step,
                "shown": rfq.ordinal in observable,
                "observable": observable.get(rfq.ordinal),
                "action": actions.get(rfq.ordinal),
                "outcome": outcomes.get(rfq.ordinal),
                "hidden": {
                    "true_fair_value": float(path.values[rfq.step]),
                    "side": int(rfq.side),
                    "informed": rfq.informed,
                    "signal_bp": rfq.signal_bp,
                    "print_index": rfq.event_index,
                    "client_informed_type": world.schedule.clients[rfq.client_id].informed_type,
                    "market": _market_side(world, rfq.ordinal),
                    "markouts": None if t is None else marks[t.seq],
                },
            }
        )
    return layers


def hidden_layer(world: World, session: Session) -> dict[str, Any]:
    """Truth for the events that are not RFQs, keyed by event sequence
    number: what the screen was hiding, what each print was, where the truth
    was when each hedge traded."""
    path = world.path
    obs_index = {int(s): i for i, s in enumerate(path.public.screen_steps)}
    out: dict[str, Any] = {}
    for e in session.log:
        if e.kind is Kind.SCREEN_UPDATE:
            i = obs_index[e.step]
            out[str(e.seq)] = {"true_fair_value": float(path.values[e.step]), "screen_error_bp": float(path.screen_error[i])}
        elif e.kind is Kind.PRINT_RELEASED:
            i = e.data["print_index"]
            out[str(e.seq)] = {
                "jump_bp": float(path.jump_sizes[i]),
                "true_fair_value_before": float(path.values[e.step - 1]),
                "true_fair_value_after": float(path.values[e.step]),
            }
        elif e.kind in (Kind.HEDGE_EXECUTED, Kind.CLOSE_LIQUIDATION):
            out[str(e.seq)] = {"true_fair_value": float(path.values[e.step])}
    return out


def analyse(world: World, session: Session, scoring: ScoringConfig) -> dict[str, Any]:
    """Scores, decomposition and diagnostics for one finished session."""
    trades = session.book.trades
    return {
        "scores": scores(world, session, scoring),
        "decomposition": sc.decomposition(world.config, world.path, world.path.public.screen_at, trades),
        "diagnostics": sc.diagnostics(world, session.log, trades),
    }


def bot_result(world: World, session: Session, scoring: ScoringConfig) -> dict[str, Any]:
    a = analyse(world, session, scoring)
    d = a["diagnostics"]
    return {
        "scores": a["scores"],
        "decomposition": a["decomposition"],
        "summary": {
            "hit_rate": d["hit_rate"],
            "outcomes": d["outcomes"],
            "informed_fills": d["flow"]["informed"]["fills"],
            "informed_lots": d["flow"]["informed"]["lots"],
            "uninformed_fills": d["flow"]["uninformed"]["fills"],
            "uninformed_lots": d["flow"]["uninformed"]["lots"],
            "max_abs_position": d["position"]["max_abs"],
            "time_weighted_mean_abs_position": d["position"]["time_weighted_mean_abs"],
        },
        "actions": session.actions,
    }


def capture(player: dict[str, Any], bots: dict[str, dict[str, Any]], scoring: ScoringConfig) -> dict[str, Any]:
    out: dict[str, Any] = {"min_gap_usd": scoring.capture_ratio_min_usd}
    for key, score in (("pnl", "pnl_usd"), ("risk_adjusted", "risk_adjusted_usd")):
        disciplined = bots["disciplined"]["scores"][score]
        oracle = bots["oracle"]["scores"][score]
        out[key] = {
            "ratio": sc.capture_ratio(player["scores"][score], disciplined, oracle, scoring.capture_ratio_min_usd),
            "oracle_minus_disciplined_usd": oracle - disciplined,
        }
    return out


def coaching_record(
    world: World,
    session: Session,
    *,
    scoring: ScoringConfig,
    benchmark: BenchmarkConfig,
    bots: dict[str, Session],
    flags: dict[str, Any],
) -> dict[str, Any]:
    player = analyse(world, session, scoring)
    bot_results = {name: bot_result(world, s, scoring) for name, s in bots.items()}
    return plain(
        {
            "format": RECORD_FORMAT,
            "format_version": FORMAT_VERSION,
            "header": world.header,
            "settings": session.settings.to_record(),
            "scoring_config": scoring.to_record(),
            "benchmark_config": benchmark.to_record(),
            "flags": flags,
            "scores": player["scores"],
            "decomposition": player["decomposition"],
            "diagnostics": player["diagnostics"],
            "benchmarks": bot_results,
            "capture_ratio": capture(player, bot_results, scoring),
            "world": {
                "path": world.path.to_record(),
                "clients": world.schedule.to_record()["clients"],
                "truth_at_screen_steps": [float(world.path.values[s]) for s in world.path.public.screen_steps],
            },
            "rfqs": rfq_layers(world, session),
            "blotter": [t.to_record() for t in session.book.trades],
            "blocks": [b.to_record() for b in session.book.blocks],
            "events": session.log.to_record(),
            "hidden_layer": hidden_layer(world, session),
            "actions": session.actions,
        }
    )


# --------------------------------------------------------------------------
# Markdown summary
# --------------------------------------------------------------------------


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def _ratio(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def markdown_summary(record: dict[str, Any]) -> str:
    h, s, dec, d = record["header"], record["scores"], record["decomposition"], record["diagnostics"]
    bots = record["benchmarks"]
    cfg = h["config"]
    minutes = cfg["session_length"] / 60
    preset = record["settings"]["preset"]
    flags = record["flags"]
    horizon = f"{record['scoring_config']['decomposition_horizon_s']:g}s"
    lines = [
        f"# Session summary — {minutes:g} minutes{' (preset)' if preset else ''}, seed {h['seed']}",
        "",
    ]
    if flags["replay_of_played_seed"]:
        lines += ["**Replay of a seed already played** — excluded from the track record.", ""]
    if flags["ended_early_at_step"] is not None:
        lines += [
            f"**Ended early at step {flags['ended_early_at_step']}** "
            f"({flags['ended_early_at_step'] * cfg['dt'] / 60:.1f} min); the position rode to the close.",
            "",
        ]
    lines += [
        "## Score",
        "",
        "| | You | Naive | Disciplined | Flow oracle |",
        "|---|---|---|---|---|",
    ]
    for label, key in (("P&L", "pnl_usd"), ("Risk penalty", "risk_penalty_usd"), ("Risk-adjusted", "risk_adjusted_usd")):
        lines.append(
            f"| {label} | {_usd(s[key])} | "
            + " | ".join(_usd(bots[b]["scores"][key]) for b in ("naive", "disciplined", "oracle"))
            + " |"
        )
    cr = record["capture_ratio"]
    lines += [
        "",
        f"Flow-reading capture ratio: P&L {_ratio(cr['pnl']['ratio'])}, risk-adjusted "
        f"{_ratio(cr['risk_adjusted']['ratio'])} (n/a unless the oracle beats the disciplined bot by at least "
        f"{_usd(cr['min_gap_usd'])}; oracle minus disciplined this session: "
        f"{_usd(cr['pnl']['oracle_minus_disciplined_usd'])} on P&L, "
        f"{_usd(cr['risk_adjusted']['oracle_minus_disciplined_usd'])} risk-adjusted).",
        "",
        "## Where the P&L came from, against true fair value",
        "",
        f"| | {horizon} |",
        "|---|---|",
        f"| Spread captured | {_usd(dec['spread_captured_usd'])} |",
        f"| − Adverse selection ({horizon} mark-outs) | {_usd(dec['by_horizon'][horizon]['adverse_selection_usd'])} |",
        f"| + Inventory beyond the mark-outs | {_usd(dec['by_horizon'][horizon]['inventory_beyond_usd'])} |",
        f"| − Hedge and closing costs | {_usd(dec['hedge_and_close_costs_usd'])} |",
        f"| **Total** | **{_usd(dec['total_usd'])}** |",
        "",
        "Adverse selection is money the flow took after you traded; negative means it moved your way.",
        "",
        "Adverse selection at each horizon: "
        + ", ".join(f"{k} {_usd(v['adverse_selection_usd'])}" for k, v in dec["by_horizon"].items())
        + ". Hedge and closing costs split: venue charge "
        + f"{_usd(dec['venue_charge_usd'])}, screen error at the hedge {_usd(dec['screen_error_on_hedges_usd'])}.",
        "",
        "## Flow",
        "",
    ]
    f = d["flow"]
    hit = d["hit_rate"]
    lines += [
        f"- Hit rate {('n/a' if hit is None else f'{hit:.0%}')} of quoted RFQs; by count in competition: "
        + ", ".join(f"{k}: {('n/a' if v is None else f'{v:.0%}')}" for k, v in d["hit_rate_by_comp"].items())
        + ".",
        f"- Fills: {f['uninformed']['fills']} uninformed ({f['uninformed']['lots']} lots), "
        f"{f['informed']['fills']} informed ({f['informed']['lots']} lots)"
        + (
            f" — {f['informed_share_of_volume']:.0%} of volume against informed clients."
            if f["informed_share_of_volume"] is not None
            else "."
        ),
    ]
    for bucket in ("uninformed", "informed"):
        m = f[bucket]["markouts"]
        if m:
            lines.append(
                f"- {bucket.capitalize()} mark-outs: "
                + ", ".join(
                    f"{k} {_usd(v['pnl_usd'])}" + (f" ({v['excluded']} excluded)" if v["excluded"] else "")
                    for k, v in m.items()
                )
                + "."
            )
    pos = d["position"]
    lines += [
        "",
        "## Position and prints",
        "",
        f"- Max |position| {pos['max_abs']} lots; time-weighted mean |position| {pos['time_weighted_mean_abs']:.0f}.",
    ]
    for p in pos["into_prints"]:
        drift = "none" if p["drift_bp"] is None else f"{p['drift_bp']:+.2f}bp"
        react = p["quotes_reflected_print_after"]
        lines.append(
            f"- Print {p['print_index'] + 1} at {p['step'] * cfg['dt'] / 60:.1f} min: {p['jump_bp']:+.2f}bp, "
            f"drift into it {drift}; you carried {p['position_into_print']:+d} lots"
            + (
                f"; quotes reflected it after {react['seconds']:.0f}s."
                if react
                else ("; no quotes after it." if not p["two_sided_quotes_after_print"] else "; quotes never moved half the print.")
            )
        )
    qt, qs = d["quote_vs_truth"], d["quote_vs_screen"]
    lines += ["", "## Quotes against the truth", ""]
    if qt["quotes"]:
        lines.append(
            f"- Quote mid minus truth: mean {qt['mean_bp']:+.2f}bp, mean |error| {qt['mean_abs_bp']:.2f}bp "
            f"over {qt['quotes']} two-sided quotes; against the screen {qs['mean_abs_bp']:.2f}bp."
        )
        ex = qt["worst_sustained_excursion"]
        if ex:
            lines.append(
                f"- Worst sustained excursion: {ex['quotes']} quotes over {ex['duration_s']:.0f}s from "
                f"{ex['start_step'] * cfg['dt'] / 60:.1f} min, {ex['mean_error_bp']:+.2f}bp off the truth on average."
            )
    informed_clients = [k for k, v in d["clients"].items() if v["informed_type"]]
    lines += [
        "",
        "## Clients",
        "",
        "Informed type this session: " + (", ".join(f"client {k}" for k in informed_clients) or "none") + ".",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Track record
# --------------------------------------------------------------------------


class TrackRecord:
    """Sessions across time. Replays of a played seed are kept but flagged
    and excluded from the aggregates."""

    NAME = "track_record.json"

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self.entries = list(entries or [])

    @classmethod
    def load(cls, storage) -> TrackRecord:
        if not storage.exists(cls.NAME):
            return cls()
        data = json.loads(storage.read_text(cls.NAME))
        if data.get("format") != TRACK_FORMAT or data.get("format_version") != FORMAT_VERSION:
            raise RecordError("not a readable track record")
        return cls(data["entries"])

    def save(self, storage) -> None:
        storage.write_text(
            self.NAME, dumps({"format": TRACK_FORMAT, "format_version": FORMAT_VERSION, "entries": self.entries})
        )

    def played(self, config: GameConfig, seed: int) -> bool:
        digest = config_digest(config)
        return any(e["config_digest"] == digest and e["seed"] == seed for e in self.entries)

    def next_id(self, settings: Settings, seed: int) -> str:
        prefix = f"{settings.preset}m" if settings.preset else "custom"
        return f"{prefix}-{seed}-{len(self.entries) + 1:04d}"

    def add(self, session_id: str, config: GameConfig, seed: int, record: dict[str, Any]) -> dict[str, Any]:
        entry = plain(
            {
                "session_id": session_id,
                "config_digest": config_digest(config),
                "seed": seed,
                "preset": record["settings"]["preset"],
                "replay_of_played_seed": record["flags"]["replay_of_played_seed"],
                "ended_early_at_step": record["flags"]["ended_early_at_step"],
                "pnl_usd": record["scores"]["pnl_usd"],
                "risk_adjusted_usd": record["scores"]["risk_adjusted_usd"],
                "capture_ratio_pnl": record["capture_ratio"]["pnl"]["ratio"],
                "capture_ratio_risk_adjusted": record["capture_ratio"]["risk_adjusted"]["ratio"],
                "disciplined_pnl_usd": record["benchmarks"]["disciplined"]["scores"]["pnl_usd"],
                "disciplined_risk_adjusted_usd": record["benchmarks"]["disciplined"]["scores"]["risk_adjusted_usd"],
                "oracle_pnl_usd": record["benchmarks"]["oracle"]["scores"]["pnl_usd"],
                "oracle_risk_adjusted_usd": record["benchmarks"]["oracle"]["scores"]["risk_adjusted_usd"],
            }
        )
        self.entries.append(entry)
        return entry

    def aggregates(self, min_gap_usd: float) -> dict[str, Any]:
        """Over scored sessions only -- replays excluded: mean scores, and
        the capture ratio pooled across sessions, sum(you - disciplined) /
        sum(oracle - disciplined). Pooled rather than a mean of per-session
        ratios: one session's oracle-minus-disciplined gap is dominated by
        where the two inventories happened to be at the prints, and a mean
        of ratios of noisy denominators is not a measurement. n/a until the
        pooled gap clears the per-session threshold."""
        scored = [e for e in self.entries if not e["replay_of_played_seed"]]
        out: dict[str, Any] = {"sessions": len(scored), "replays_excluded": len(self.entries) - len(scored)}
        for key in ("pnl_usd", "risk_adjusted_usd"):
            xs = [e[key] for e in scored]
            out[key] = float(np.mean(xs)) if xs else None
        for key, you, disc, orc in (
            ("pooled_capture_ratio_pnl", "pnl_usd", "disciplined_pnl_usd", "oracle_pnl_usd"),
            ("pooled_capture_ratio_risk_adjusted", "risk_adjusted_usd", "disciplined_risk_adjusted_usd", "oracle_risk_adjusted_usd"),
        ):
            out[key] = sc.capture_ratio(
                sum(e[you] for e in scored), sum(e[disc] for e in scored), sum(e[orc] for e in scored), min_gap_usd
            )
        return out
