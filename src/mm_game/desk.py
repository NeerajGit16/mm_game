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
"""

from __future__ import annotations

import bisect
from typing import Any

from mm_game.protocol import PROTOCOL_VERSION

MARKOUT_HORIZONS_S: tuple[int, ...] = (60, 300)
RECENT_INTERACTIONS = 3


def _horizon_key(h: int) -> str:
    return f"{h}s"


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

    # -- input ----------------------------------------------------------------

    def feed(self, message: dict[str, Any]) -> None:
        if message.get("type") == "event":
            self._event(message["kind"], message["step"], message["data"])
            self.step = max(self.step, message["step"])
        elif message.get("type") == "state":
            self.step = message["step"]

    def _event(self, kind: str, step: int, d: dict[str, Any]) -> None:
        if kind == "session_start":
            self.dt = d["config"]["dt"]
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

    def message(self) -> dict[str, Any]:
        clients = []
        for cid, c in self._clients.items():
            per_lot, lots = self._client_markouts(cid)
            clients.append({**c, "markout_bp_per_lot": per_lot, "markout_lots": lots, "recent": self._recent(cid)})
        return {
            "v": PROTOCOL_VERSION,
            "type": "desk",
            "step": self.step,
            "blotter": [dict(r) for r in reversed(self._rows.values())],
            "clients": clients,
        }
