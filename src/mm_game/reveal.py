"""What the end-of-session reveal draws, built once from the world after the
session and kept in the coaching record, so the page can draw the reveal
from the record alone, with no world and no engine.

The reveal block carries: the truth (sampled every second) and the screen
as paths; each print with its jump and run-up; each unannounced move with
its leak window; every fill and hedge with who was on the other side; the
disciplined bot's and the flow oracle's fills, positions and what each did
on every RFQ; the three moments; and the bots' rules.

**The three moments** are the three largest P&L events by absolute size,
whichever kind they are:

- a print: the position you carried into it times its jump;
- an unannounced move: the move's own contribution to your P&L -- the
  position you held over each step of its ramp times the move's increment
  on that step;
- a single ticket (a client fill), scored by what the ticket was. A fill on
  an informed RFQ: its price against the truth at its next event -- the
  edge plus the mark-out through the next print or move end -- or at the
  close when no event follows. Any other fill: the edge plus the mark-out
  over ``UNINFORMED_TICKET_S``, stopped short of the next print's jump or
  the next move's ramp. The print and move moments already carry the
  position through those, so a sale ahead of a print that cut risk is not
  scored as the print's loss.

Ties go to the earlier one. Each comes with one sentence saying what the
flow oracle did instead.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mm_game import handbook
from mm_game.book import Source, Trade
from mm_game.config import BenchmarkConfig
from mm_game.engine import Session
from mm_game.events import Kind
from mm_game.scoring import fill_type, position_path
from mm_game.world import World

TRUTH_EVERY_S = 1.0
MOMENTS = 3
UNINFORMED_TICKET_S = 300.0
BOTS = ("disciplined", "oracle")
_KIND_ORDER = {"print": 0, "move": 1, "ticket": 2}


# --------------------------------------------------------------------------
# Formatting for the sentences
# --------------------------------------------------------------------------


def _usd(x: float) -> str:
    a = round(abs(x))
    return f"{'−' if x < 0 and a else '+' if a else ''}${a:,.0f}"


def _price(bp: float | None) -> str:
    return "—" if bp is None else f"{bp / 100:.4f}"


def _clock(step: int, dt: float) -> str:
    s = int(math.floor(step * dt + 1e-9))
    return f"{s // 60:02d}:{s % 60:02d}"


def _lots(q: int) -> str:
    return f"{q:+d}"


# --------------------------------------------------------------------------
# Paths, prints, moves, fills, positions
# --------------------------------------------------------------------------


def _positions(trades: list[Trade]) -> dict[str, list[int]]:
    """The position as a step function: where it changes, and to what."""
    steps, positions = [0], [0]
    for t in trades:
        if t.step == steps[-1]:
            positions[-1] = t.position_after
        else:
            steps.append(t.step)
            positions.append(t.position_after)
    return {"steps": steps, "positions": positions}


def _fills(world: World, trades: list[Trade]) -> list[dict[str, Any]]:
    """Every fill and hedge, with who was on the other side and, for a
    client fill, its edge and mark-outs against the truth."""
    config, path = world.config, world.path
    usd = config.dollars_per_bp_per_lot
    rfqs = world.schedule.rfqs
    out = []
    for t in trades:
        truth = float(path.values[t.step])
        row: dict[str, Any] = {
            "seq": t.seq,
            "step": t.step,
            "source": t.source.value,
            "quantity": t.quantity,
            "price": t.price,
            "edge_usd": t.quantity * (truth - t.price) * usd,
        }
        if t.source is Source.CLIENT:
            q = rfqs[t.rfq_ordinal]
            row.update(
                rfq_ordinal=t.rfq_ordinal,
                client_id=q.client_id,
                fill_type=fill_type(q),
                informed_about=None if q.informed_about is None else q.informed_about.to_record(),
                ticket_usd=_ticket_usd(world, t),
            )
        out.append(row)
    return out


def _ticket_usd(world: World, t: Trade) -> float:
    """A client fill's P&L against the truth at its next event, or at the
    close when nothing follows it."""
    path = world.path
    ev = path.next_event_markout(t.step)
    end = ev.event_step if ev.event_step is not None else path.n_steps
    return t.quantity * (float(path.values[end]) - t.price) * world.config.dollars_per_bp_per_lot


def ticket_end(world: World, t: Trade) -> tuple[int, str, int | None]:
    """Where a client fill's moment score stops: the step whose truth it is
    marked to, why (``print``, ``move``, ``close`` or ``horizon``), and which
    print or move. An informed fill runs through its next event. Any other
    fill runs ``UNINFORMED_TICKET_S`` and stops short of the next print's
    jump (the step before it lands) or the next move's ramp (its start; a
    fill inside a ramp stops where it is)."""
    path = world.path
    n = path.n_steps
    if fill_type(world.schedule.rfqs[t.rfq_ordinal]) != "uninformed":
        ev = path.next_event_markout(t.step)
        return (n, "close", None) if ev.event_step is None else (ev.event_step, ev.kind, ev.index)
    horizon = t.step + round(UNINFORMED_TICKET_S / world.config.dt)
    end = (horizon, "horizon", None) if horizon <= n else (n, "close", None)
    cuts = [(int(e) - 1, "print", i) for i, e in enumerate(path.event_steps) if e > t.step]
    cuts += [(max(m.start_step, t.step), "move", m.index) for m in path.move_list if m.end_step > t.step]
    return min([end, *cuts], key=lambda c: c[0])


def _bot_rfqs(session: Session) -> list[dict[str, Any]]:
    """What a bot did on each RFQ it was shown: its quote and the outcome."""
    quotes = {e.data["rfq_ordinal"]: e.data for e in session.log.of(Kind.QUOTE_SENT)}
    out = []
    for e in session.log.of(Kind.RFQ_RESOLVED):
        d = e.data
        q = quotes.get(d["rfq_ordinal"], {})
        out.append(
            {
                "rfq_ordinal": d["rfq_ordinal"],
                "bid": q.get("bid"),
                "ask": q.get("ask"),
                "outcome": d["outcome"],
                "quantity": d["quantity"],
                "price": d["price"],
            }
        )
    return out


# --------------------------------------------------------------------------
# The moments
# --------------------------------------------------------------------------


def candidates(world: World, trades: list[Trade], oracle: list[Trade]) -> list[dict[str, Any]]:
    """Every print, move and client fill, with your P&L on it and the
    oracle's on the same event (for a ticket: what it did on that RFQ)."""
    config, path = world.config, world.path
    n, usd, dt = config.n_steps, config.dollars_per_bp_per_lot, config.dt
    held = position_path(trades, n)
    held_oracle = position_path(oracle, n)
    move_step = np.diff(path.moves)
    out: list[dict[str, Any]] = []
    for i, e in enumerate(int(x) for x in path.event_steps):
        jump = float(path.jump_sizes[i])
        window = next((w for w in path.drift_windows if w.event_index == i), None)
        out.append(
            {
                "kind": "print",
                "index": i,
                "step": e,
                "from_step": max(0, (window.start_step if window else e - round(600 / dt)) - round(60 / dt)),
                "to_step": min(n, e + round(300 / dt)),
                "pnl_usd": int(held[e - 1]) * jump * usd,
                "oracle_pnl_usd": int(held_oracle[e - 1]) * jump * usd,
                "you_lots": int(held[e - 1]),
                "oracle_lots": int(held_oracle[e - 1]),
                "size_bp": jump,
            }
        )
    for m in path.move_list:
        ramp = slice(m.start_step, m.end_step)
        out.append(
            {
                "kind": "move",
                "index": m.index,
                "step": m.start_step,
                "from_step": max(0, m.inform_step - round(60 / dt)),
                "to_step": min(n, m.end_step + round(300 / dt)),
                "pnl_usd": float(np.sum(held[ramp] * move_step[ramp])) * usd,
                "oracle_pnl_usd": float(np.sum(held_oracle[ramp] * move_step[ramp])) * usd,
                "you_lots": int(held[m.start_step]),
                "oracle_lots": int(held_oracle[m.start_step]),
                "size_bp": m.total_bp,
                "lead_s": (m.start_step - m.inform_step) * dt,
                "duration_s": (m.end_step - m.start_step) * dt,
            }
        )
    for t in trades:
        if t.source is not Source.CLIENT:
            continue
        end, why, which = ticket_end(world, t)
        out.append(
            {
                "kind": "ticket",
                "index": t.rfq_ordinal,
                "step": t.step,
                "from_step": max(0, t.step - round(300 / dt)),
                "to_step": min(n, max(end, t.step) + round(120 / dt)),
                "pnl_usd": t.quantity * (float(path.values[end]) - t.price) * usd,
                "end_step": end,
                "end": why,
                "end_index": which,
                "quantity": t.quantity,
                "price": t.price,
            }
        )
    return out


def _ticket_basis(world: World, c: dict[str, Any]) -> str:
    """What a ticket moment's dollars measure, in words."""
    informed = fill_type(world.schedule.rfqs[c["index"]]) != "uninformed"
    which = None if c["end_index"] is None else c["end_index"] + 1
    if informed:
        return {
            "print": f"against the truth through print {which}",
            "move": f"against the truth through the end of unannounced move {which}",
            "close": "against the truth at the close",
        }[c["end"]]
    after = f"{(c['end_step'] - c['step']) * world.config.dt:.0f}s on"
    return {
        "horizon": f"its spread plus its {UNINFORMED_TICKET_S:.0f}s mark-out against the truth",
        "print": f"its spread plus its mark-out against the truth, stopped before print {which} ({after})",
        "move": f"its spread plus its mark-out against the truth, stopped before unannounced move {which} ({after})",
        "close": "its spread plus its mark-out against the truth up to the close",
    }[c["end"]]


def choose(cands: list[dict[str, Any]], k: int = MOMENTS) -> list[dict[str, Any]]:
    """The k largest by absolute P&L; ties to the earlier, then print, move, ticket."""
    return sorted(cands, key=lambda c: (-abs(c["pnl_usd"]), c["step"], _KIND_ORDER[c["kind"]], c["index"]))[:k]


def _oracle_on(rfq: dict[str, Any] | None) -> str:
    if rfq is None:
        return "never saw it"
    if rfq["outcome"] == "passed":
        return "passed it"
    quoted = f"quoted {_price(rfq['bid'])} / {_price(rfq['ask'])}"
    if rfq["outcome"] == "won":
        return f"{quoted} and traded {_lots(rfq['quantity'])} at {_price(rfq['price'])}"
    if rfq["outcome"] == "done_away":
        return f"{quoted} and it traded elsewhere"
    return f"{quoted} and the client walked"


def _sentence(world: World, c: dict[str, Any], oracle_rfqs: dict[int, dict[str, Any]]) -> tuple[str, str]:
    dt = world.config.dt
    when = _clock(c["step"], dt)
    if c["kind"] == "print":
        title = f"{when} · print {c['index'] + 1}, {c['size_bp']:+.2f}bp"
        text = (
            f"You carried {_lots(c['you_lots'])} lots into it: {_usd(c['pnl_usd'])}; "
            f"the flow oracle carried {_lots(c['oracle_lots'])}: {_usd(c['oracle_pnl_usd'])}."
        )
    elif c["kind"] == "move":
        title = f"{when} · unannounced move {c['index'] + 1}, {c['size_bp']:+.2f}bp"
        text = (
            f"It ran over {c['duration_s']:.0f}s, {c['lead_s'] / 60:.1f} minutes after the first informed client "
            f"heard; you held {_lots(c['you_lots'])} lots: {_usd(c['pnl_usd'])}; "
            f"the flow oracle held {_lots(c['oracle_lots'])}: {_usd(c['oracle_pnl_usd'])}."
        )
    else:
        q = world.schedule.rfqs[c["index"]]
        client = world.schedule.clients[q.client_id]
        category = client.category.value.replace("_", " ")
        if q.informed_about is not None:
            about = q.informed_about
            truth = (float(world.path.jump_sizes[about.index]) if about.kind == "print"
                     else world.path.move_list[about.index].total_bp)
            right = (q.signal_bp > 0) == (truth > 0)
            who = f"{category}, informed about {about.kind} {about.index + 1}, {'right' if right else 'wrong'}"
        else:
            who = f"{category}, {'informed type' if client.informed_type else 'uninformed'}"
        title = f"{when} · RFQ {c['index']}, client {q.client_id}"
        text = (
            f"You {'bought' if c['quantity'] > 0 else 'sold'} {abs(c['quantity'])} "
            f"{'from' if c['quantity'] > 0 else 'to'} client {q.client_id} ({who}) "
            f"at {_price(c['price'])}: {_usd(c['pnl_usd'])}, {_ticket_basis(world, c)}; "
            f"the flow oracle {_oracle_on(oracle_rfqs.get(c['index']))}."
        )
    return title, text


def moments(world: World, trades: list[Trade], oracle: Session) -> list[dict[str, Any]]:
    oracle_rfqs = {r["rfq_ordinal"]: r for r in _bot_rfqs(oracle)}
    chosen = choose(candidates(world, trades, list(oracle.book.trades)))
    out = []
    for c in chosen:
        title, text = _sentence(world, c, oracle_rfqs)
        out.append(
            {
                "kind": c["kind"],
                "index": c["index"],
                "step": c["step"],
                "from_step": c["from_step"],
                "to_step": c["to_step"],
                "pnl_usd": c["pnl_usd"],
                "title": title,
                "sentence": text,
            }
        )
    return out


# --------------------------------------------------------------------------
# The block
# --------------------------------------------------------------------------


def build(
    world: World, session: Session, bots: dict[str, Session], *, benchmark: BenchmarkConfig
) -> dict[str, Any]:
    config, path = world.config, world.path
    view = path.public
    every = max(1, round(TRUTH_EVERY_S / config.dt))
    trades = list(session.book.trades)
    windows = {w.event_index: w for w in path.drift_windows}
    return {
        "truth": {"every_steps": every, "values": [round(float(v), 4) for v in path.values[::every]]},
        "screen": {"steps": [int(s) for s in view.screen_steps], "mids": [float(m) for m in view.screen]},
        "prints": [
            {
                "index": i,
                "step": int(e),
                "jump_bp": float(path.jump_sizes[i]),
                "runup": None if i not in windows else {
                    "start_step": windows[i].start_step,
                    "end_step": windows[i].end_step,
                    "total_bp": windows[i].total_bp,
                },
            }
            for i, e in enumerate(path.event_steps)
        ],
        "moves": [
            {
                "index": m.index,
                "inform_step": m.inform_step,
                "start_step": m.start_step,
                "end_step": m.end_step,
                "size_bp": m.total_bp,
            }
            for m in path.move_list
        ],
        "fills": _fills(world, trades),
        "positions": {"you": _positions(trades), **{b: _positions(list(bots[b].book.trades)) for b in BOTS}},
        "bots": {
            b: {
                "fills": [
                    {"step": t.step, "source": t.source.value, "quantity": t.quantity, "price": t.price,
                     "rfq_ordinal": t.rfq_ordinal}
                    for t in bots[b].book.trades
                ],
                "rfqs": _bot_rfqs(bots[b]),
            }
            for b in BOTS
        },
        "moments": moments(world, trades, bots["oracle"]),
        "rules": {
            "disciplined": [i.to_record() for i in handbook.disciplined_rules(benchmark, config)],
            "oracle": [i.to_record() for i in handbook.oracle_rules(benchmark)],
        },
    }
