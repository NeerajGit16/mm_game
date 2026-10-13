"""The desk's own records, built from what the player has been shown.

A trader keeps numbers of their own alongside the market: a blotter that
says what each outcome revealed, and per client the inquiries, the trades,
the net position and how those trades marked out afterwards. They are
derived here, in one place, from the outbound protocol messages -- events
and snapshots -- and nothing else. This module imports nothing that can
reach the world, so it cannot contain anything the player could not have
worked out by hand from their own screen.

Mark-outs are against the screen, never the truth. For a fill of q lots at
step t (q > 0 when you bought) the mark-out over h seconds is
q x (screen at t + h - screen at t), in bp-lots: positive when the market
moved your way after you traded. It becomes available once the clock has
passed t + h. Per lot is the sum over a client's matured fills divided by
their lots. The edge you charged is deliberately left out: a client you
quote wide can still be the one the market moves against you after.

P&L is against the screen too: every trade, hedges and the close included,
marked to the screen showing now -- the snapshot's, as the top bar marks
it. A trade of q lots at price p, when the screen read s, splits exactly
into its **edge** q x (s - p) -- what you charged a client, or paid the
venue -- and its **move since** q x (screen now - s), what the market did
to it. It is a linear product, so the rows add up to the book's P&L with no
matching of buys to sells. A venue trade's screen is read off its own fill
(price less the side times the per-lot cost), which also covers the close
after a quit, when no screen readings are shown in between.

Each blotter row also says how your price compared with the best other
dealer's on the side the client traded (``vs_best_dealer_bp``): what a win
gave away, or what a done-away missed by. The cover and the done-away level
are already on the blotter, so it is arithmetic on what the player saw.
The record and summary.md compute it with the same function.
"""

from __future__ import annotations

import bisect
from typing import Any

from mm_game.protocol import PROTOCOL_VERSION

MARKOUT_HORIZONS_S: tuple[int, ...] = (60, 300)
RECENT_INTERACTIONS = 3


def _horizon_key(h: int) -> str:
    return f"{h}s"


def vs_best_dealer_bp(
    outcome: str | None, client_side: int | None, bid: float | None, ask: float | None, other: float | None
) -> float | None:
    """Your price on the side the client traded against the best other
    dealer's, in bp, signed from your side: ``client_side`` x (other - yours),
    where yours is your ask when the client bought and your bid when it sold.

    Won: ``other`` is the cover, so it is never negative -- +0.75 means you
    beat the best other dealer by 0.75bp, about what you gave away. Done
    away: ``other`` is the level it traded at, so it is never positive, and
    0.0 on a tie you lost. None when there is nothing to compare: a win with
    nobody else in (no cover), not done, passed, or a done-away on a side
    you did not quote."""
    if outcome not in ("won", "done_away") or client_side is None or other is None:
        return None
    yours = ask if client_side == 1 else bid
    if yours is None:
        return None
    # Rounded off the float noise of an off-grid tick size, and + 0.0 so a
    # tie is 0.0: a tie on the bid side would otherwise be -0.0, and print so.
    return round(client_side * (other - yours), 9) + 0.0


class Desk:
    """Feed it every outbound message, in order; ``message()`` is the
    current desk, as one protocol message."""

    def __init__(self) -> None:
        self.dt: float | None = None
        self.step = 0
        self._screen_steps: list[int] = []
        self._screen_mids: list[float] = []
        self._rows: dict[int, dict[str, Any]] = {}  # by RFQ ordinal, in arrival order
        self._clients: dict[int, dict[str, Any]] = {}
        self._fills: list[dict[str, Any]] = []  # client fills: step, client, quantity
        self._usd_per_bp: float | None = None
        self._trades: list[dict[str, Any]] = []  # every trade: the P&L rows before marking
        self._position = 0
        self._cash_bp = 0.0  # minus the sum of quantity x price, relative to the start level
        self._level = 0.0
        self._series: dict[str, list] = {"step": [], "pnl_usd": [], "position": []}
        self._book: dict[str, float] = {"realised_usd": 0.0, "unrealised_usd": 0.0}
        self._now_mid: float | None = None  # the latest snapshot's screen

    # -- input ----------------------------------------------------------------

    def feed(self, message: dict[str, Any]) -> None:
        if message.get("type") == "event":
            self._event(message["kind"], message["step"], message["data"])
            self.step = max(self.step, message["step"])
        elif message.get("type") == "state":
            self.step = message["step"]
            self._book = {k: message["book"][k] for k in ("realised_usd", "unrealised_usd")}
            self._now_mid = message["screen"]["mid"]

    def _event(self, kind: str, step: int, d: dict[str, Any]) -> None:
        if kind == "session_start":
            self.dt = d["config"]["dt"]
            self._usd_per_bp = d["config"]["tick_value"] / d["config"]["tick_size"]
            self._level = d["config"]["start_level"]
            for c in d["roster"]:
                self._clients[c["client_id"]] = {
                    "client_id": c["client_id"],
                    "category": c["category"],
                    "rfqs": 0,
                    "quoted": 0,
                    "traded": 0,
                    "lots": 0,
                    "net": 0,
                    "last_trade_step": None,
                }
        elif kind == "screen_update":
            self._screen_steps.append(step)
            self._screen_mids.append(d["mid"])
            self._mark(step, d["mid"])
        elif kind == "rfq_arrived":
            r = d["rfq"]
            self._rows[r["ordinal"]] = {
                "ordinal": r["ordinal"],
                "step": step,
                "client_id": r["client_id"],
                "category": r["category"],
                "size": r["size"],
                "one_way": r["one_way"],
                "side": r["side"],
                "in_comp": r["in_comp"],
                "bid": None,
                "ask": None,
                "outcome": None,
                "quantity": 0,
                "price": None,
                "revealed_side": None,
                "learned": None,
                "vs_best_dealer_bp": None,
                "position_after": None,
            }
            self._clients[r["client_id"]]["rfqs"] += 1
        elif kind == "quote_sent":
            row = self._rows[d["rfq_ordinal"]]
            row["bid"], row["ask"] = d["bid"], d["ask"]
        elif kind == "rfq_resolved":
            row = self._rows[d["rfq_ordinal"]]
            row.update(
                outcome=d["outcome"],
                quantity=d["quantity"],
                price=d["price"],
                revealed_side=d["side"],
                vs_best_dealer_bp=vs_best_dealer_bp(d["outcome"], d["side"], row["bid"], row["ask"], d["cover"]),
                position_after=d["position_after"],
            )
            if d["outcome"] == "won":
                row["learned"] = {"cover": d["cover"]}
            elif d["outcome"] == "done_away":
                row["learned"] = {"side": d["side"], "level": d["cover"]}
            client = self._clients[row["client_id"]]
            if d["outcome"] != "passed":
                client["quoted"] += 1
            if d["outcome"] == "won":
                client["traded"] += 1
                client["lots"] += abs(d["quantity"])
                client["net"] += d["quantity"]
                client["last_trade_step"] = step
                self._fills.append({"step": step, "client_id": row["client_id"], "quantity": d["quantity"]})
                self._trade(step, "client", d["quantity"], d["price"], row, self._screen_at(step))
        elif kind in ("hedge_executed", "close_liquidation") and d["quantity"]:
            screen = d["price"] - (1 if d["quantity"] > 0 else -1) * d["cost_bp_per_lot"]
            self._trade(step, "hedge" if kind == "hedge_executed" else "close", d["quantity"], d["price"], None, screen)

    def _trade(
        self, step: int, kind: str, quantity: int, price: float, row: dict[str, Any] | None, screen: float
    ) -> None:
        self._trades.append(
            {
                "step": step,
                "kind": kind,
                "ordinal": None if row is None else row["ordinal"],
                "client_id": None if row is None else row["client_id"],
                "category": None if row is None else row["category"],
                "quantity": quantity,
                "price": price,
                "screen_then": screen,
            }
        )
        self._position += quantity
        self._cash_bp -= quantity * (price - self._level)
        self._mark(step, screen)

    def _mark(self, step: int, screen: float) -> None:
        """A point on the P&L chart: everything held, at the screen then."""
        pnl_bp = self._cash_bp + self._position * (screen - self._level)
        self._series["step"].append(step)
        self._series["pnl_usd"].append(pnl_bp * self._usd_per_bp)
        self._series["position"].append(self._position)

    # -- derived --------------------------------------------------------------

    def _screen_at(self, step: int) -> float:
        return self._screen_mids[bisect.bisect_right(self._screen_steps, step) - 1]

    def _markout_bp(self, fill_step: int, quantity: int, horizon_s: int) -> float | None:
        """A fill's mark-out over ``horizon_s`` in bp-lots, or None if the
        clock has not got there yet."""
        later = fill_step + round(horizon_s / self.dt)
        if later > self.step:
            return None
        return quantity * (self._screen_at(later) - self._screen_at(fill_step))

    def _client_markouts(self, client_id: int) -> tuple[dict[str, float | None], dict[str, int]]:
        per_lot, lots_in = {}, {}
        for h in MARKOUT_HORIZONS_S:
            total, lots = 0.0, 0
            for f in self._fills:
                if f["client_id"] != client_id:
                    continue
                m = self._markout_bp(f["step"], f["quantity"], h)
                if m is not None:
                    total += m
                    lots += abs(f["quantity"])
            per_lot[_horizon_key(h)] = total / lots if lots else None
            lots_in[_horizon_key(h)] = lots
        return per_lot, lots_in

    def _recent(self, client_id: int) -> list[dict[str, Any]]:
        done = [r for r in self._rows.values() if r["client_id"] == client_id and r["outcome"] is not None]
        out = []
        for r in reversed(done[-RECENT_INTERACTIONS:]):
            won = r["outcome"] == "won"
            out.append(
                {
                    "ordinal": r["ordinal"],
                    "step": r["step"],
                    "size": r["size"],
                    "outcome": r["outcome"],
                    "quantity": r["quantity"],
                    "level": r["learned"]["level"] if r["outcome"] == "done_away" else None,
                    "client_side": r["revealed_side"],
                    "markout_bp_per_lot": {
                        _horizon_key(h): (
                            None if not won or (m := self._markout_bp(r["step"], r["quantity"], h)) is None
                            else m / abs(r["quantity"])
                        )
                        for h in MARKOUT_HORIZONS_S
                    },
                }
            )
        return out

    def _pnl(self) -> dict[str, Any]:
        """Every trade marked to the screen now, the tiles they add up to,
        and the P&L and position through the session."""
        usd = self._usd_per_bp
        now = self._now_mid if self._now_mid is not None else self._screen_at(self.step)
        rows = []
        for i, t in enumerate(self._trades):
            edge = t["quantity"] * (t["screen_then"] - t["price"]) * usd
            move = t["quantity"] * (now - t["screen_then"]) * usd
            rows.append({"seq": i, **t, "edge_usd": edge, "move_usd": move, "total_usd": edge + move})
        edge_client = sum(r["edge_usd"] for r in rows if r["kind"] == "client")
        venue = sum(r["edge_usd"] for r in rows if r["kind"] != "client")
        move = sum(r["move_usd"] for r in rows)
        return {
            "rows": rows,
            "tiles": {
                "edge_client_usd": edge_client,
                "venue_charge_usd": venue,
                "move_usd": move,
                "total_usd": edge_client + venue + move,
                **self._book,
            },
            "series": {k: list(v) for k, v in self._series.items()},
        }

    def message(self) -> dict[str, Any]:
        clients = []
        for cid, c in self._clients.items():
            per_lot, lots = self._client_markouts(cid)
            clients.append(
                {
                    **c,
                    "hit_rate": c["traded"] / c["quoted"] if c["quoted"] else None,
                    "markout_bp_per_lot": per_lot,
                    "markout_lots": lots,
                    "recent": self._recent(cid),
                }
            )
        return {
            "v": PROTOCOL_VERSION,
            "type": "desk",
            "step": self.step,
            "blotter": [dict(r) for r in reversed(self._rows.values())],
            "clients": clients,
            "pnl": self._pnl(),
        }
