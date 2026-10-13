"""What every column means: the one place the definitions live.

The page's (i) panels and summary.md both show these, so the two can never
explain a number differently. Each table is keyed by where it appears
(``play.*`` during the session, ``reveal.*`` after it, ``summary.*`` in
summary.md); each entry is a column header -- or, for a table read by rows,
a row label -- exactly as it is printed, with a one-line meaning that says
which way the sign runs. A test checks every header on the page and in
summary.md has an entry here.

During play every number is against the screen: the truth stays hidden
until the reveal. After it, mark-outs and the breakdown are against the
true value.

Pure: built from a few numbers -- the mark-out horizons, the two
capture-ratio thresholds, the tick and the half-bp line the wins table is
split at -- and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from mm_game.protocol import PROTOCOL_VERSION


@dataclass(frozen=True)
class Entry:
    label: str
    meaning: str


@dataclass(frozen=True)
class Table:
    title: str
    intro: str
    entries: tuple[Entry, ...]

    def labels(self) -> list[str]:
        return [e.label for e in self.entries]

    def to_record(self) -> dict[str, object]:
        return {
            "title": self.title,
            "intro": self.intro,
            "entries": [{"label": e.label, "meaning": e.meaning} for e in self.entries],
        }


NEXT = "next_event"
NEXT_LABEL = "Mark-out to next print/move"


def horizon_key(seconds: float) -> str:
    return f"{seconds:g}s"


def markout_label(key: str) -> str:
    """A mark-out horizon's column header: ``60s`` -> Mark-out 60s."""
    return NEXT_LABEL if key == NEXT else f"Mark-out {key}"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------
# Meanings used in more than one table
# --------------------------------------------------------------------------

CLIENT = Entry("Client", "The client's number this session, with its category.")
CATEGORY = Entry("Category", "Hedge fund, real money or corporate: shown on every ticket.")
TRUE_TYPE = Entry(
    "True type",
    "Revealed now. Informed: some of its RFQs came knowing about a print or an unannounced move. "
    "Uninformed: none did.",
)
RFQS = Entry("RFQs", "RFQs the client sent you.")
INFORMED_RFQS = Entry(
    "Informed RFQs (filled)",
    "Of its RFQs, how many came inside its information window, knowing about a print or a move; in brackets, "
    "how many of those you filled. An informed client's RFQs outside its window count as uninformed.",
)
QUOTED = Entry("Quoted", "RFQs you quoted rather than passed (a pricing timer that ran out counts as a pass).")
WON = Entry("Won", "RFQs you traded: the client dealt on your price.")
LOTS_GROSS = Entry("Lots traded (gross)", "Your buys plus your sells with the client, in lots: never negative.")
NET = Entry(
    "Your net (+ = you bought)",
    "Your net position from trades with the client, in lots: + when on balance you bought from them, − when you sold.",
)
KNEW_ABOUT = Entry(
    "Knew about",
    "For an informed client: each print or move it knew about, the way it was told the market would go, "
    "and whether that was right.",
)


def _truth_markout(key: str) -> str:
    if key == NEXT:
        return (
            "The same as the other mark-outs, measured from each fill to the next print or the end of the next "
            "unannounced move, whichever comes first. + means it moved your way. Fills with neither after them "
            "are left out."
        )
    return (
        f"For each fill, the size times how far the true value moved from the fill to {key} later, in dollars "
        "(+ size when you bought). The edge you charged is left out: that is the spread line. + means the "
        f"market moved your way. Fills whose {key} ran past the close are left out."
    )


def _truth_markout_entry(key: str, label: str | None = None) -> Entry:
    return Entry(label or markout_label(key), _truth_markout(key))


INFORMED_FILLS = (
    "The mark-out to the next print or move from your fills on its informed RFQs only. "
    "— when you filled none of them."
)
OTHER_FILLS = (
    "The same from all your other fills with the client, including an informed client's RFQs outside "
    "its window. — when there were none."
)


def _screen_markout_per_lot(key: str) -> Entry:
    return Entry(
        f"Mark-out {key} / lot",
        f"Over your fills with the client at least {key} old: the size times how far the screen moved from the "
        f"fill to {key} later, summed and divided by those lots, in bp a lot. The edge you charged is left out. "
        "+ means the market moved your way. Against the screen: the truth stays hidden until the reveal.",
    )


def _lots_in(key: str) -> Entry:
    return Entry(f"Lots in {key}", f"The lots in the {key} average: your fills with the client at least {key} old.")


VS_BEST = Entry(
    "vs best dealer",
    "Your price against the best other dealer's on the side the client traded, in bp: + when you beat it. Won: "
    "the cover minus your ask when the client bought, your bid minus the cover when it sold — never −; +0.75 "
    "means you beat the best other dealer by 0.75bp, roughly what you gave away. Done away: the same with the "
    "level it traded at — −, or 0.00 on a tie you lost. Blank: won with nobody else in, not done, passed, or "
    "done away on a side you didn't quote.",
)

LEAN_BEFORE_MOVE = (
    "Lean before a move: the mean of (quote mid − screen) over the two-sided quotes from the leak to the start of "
    "its ramp, in bp, + toward the move. Yours includes your inventory skew: it is your net lean, the one that made "
    "or lost money on the move. The flow oracle's includes its lean on informed flow; the disciplined bot's is "
    "inventory skew only."
)

def win_gap_labels(tick_bp: float, half_bp: float) -> dict[str, str]:
    """Row labels for the wins table, keyed by scoring's win buckets, and the
    total row under them. The middle bucket -- above a tie, under ``half_bp``
    -- holds whole ticks only, so it is named by them: 0.25bp at the
    quarter-bp tick."""
    ticks = [k * tick_bp for k in range(1, int(half_bp / tick_bp) + 1) if k * tick_bp < half_bp - 1e-9]
    if not ticks:
        under = f"under {half_bp:.2f}bp"
    elif len(ticks) == 1:
        under = f"{ticks[0]:.2f}bp"
    else:
        under = f"{ticks[0]:.2f}–{ticks[-1]:.2f}bp"
    return {
        "tie": "0.00bp (a tie)",
        "under_half_bp": under,
        "half_bp_or_more": f"{half_bp:.2f}bp or more",
        "nobody_else_in": "Nobody else in",
        "all": "All wins",
    }


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------


def tables(
    *,
    horizons_s: tuple[float, ...],
    per_session_min_usd: float,
    pooled_min_usd: float,
    tick_bp: float,
    win_half_bp: float,
) -> dict[str, Table]:
    """Every table's definitions. ``horizons_s`` are the mark-out horizons
    of the breakdown; the thresholds are the capture ratio's; the tick and
    the half-bp line name the buckets of the wins table."""
    keys = [horizon_key(h) for h in horizons_s] + [NEXT]
    gaps = win_gap_labels(tick_bp, win_half_bp)
    capture_rule = (
        f"(you − disciplined) ÷ (flow oracle − disciplined) for the session; n/a unless the oracle beat the "
        f"disciplined bot by at least {_usd(per_session_min_usd)}."
    )
    pooled_rule = (
        f"The pooled ratio is the same over all scored sessions summed; it stays n/a until the oracle's pooled "
        f"edge over the disciplined bot reaches {_usd(pooled_min_usd)}. Practice replays are shown, never counted."
    )
    breakdown_lines = (
        Entry("Spread captured", "Each client fill's edge against the true value when you traded: size × (truth − price). + is in your favour."),
        Entry(
            "Adverse selection",
            "Your client fills' mark-outs at that horizon: size × how far the truth moved from the fill. − when the flow "
            "moved against you.",
        ),
        Entry(
            "Inventory beyond the mark-outs",
            "What the truth did to your position after each fill's mark-out had run: + when it moved your way.",
        ),
        Entry(
            "Hedging and closing",
            "Your hedges and the close against the truth: the broker's charge plus the screen's error when you traded. "
            "Usually −: they fill off the screen, not the true value.",
        ),
        Entry("Total", "Your P&L. Every column adds up to it."),
    )
    by_type = (
        ("uninformed", "Uninformed", "fills on uninformed RFQs (an informed client's RFQs outside its window included)"),
        ("informed_print", "Informed about a print", "fills on RFQs from a client that knew about a print"),
        ("informed_move", "Informed about a move", "fills on RFQs from a client that knew about an unannounced move"),
    )
    horizon_entries = tuple(
        Entry(
            markout_label(k),
            f"Adverse selection measured over {k}: inventory beyond takes the rest, so the column still adds up."
            if k != NEXT
            else "Adverse selection measured to the next print or the end of the next unannounced move, whichever "
            "comes first.",
        )
        for k in keys
    )
    score_cols = (
        Entry("P&L", "Realised P&L in dollars after the close, on the same world and seed for everyone."),
        Entry(
            "Risk-adj.",
            "P&L minus the risk penalty: a charge for each print held through, about κ × (dollar position)², and a "
            "small one for holding between prints.",
        ),
    )
    t: dict[str, Table] = {}

    # — play ------------------------------------------------------------------
    t["play.blotter"] = Table(
        "Blotter",
        "Every RFQ you were shown, most recent first, with what its outcome told you.",
        (
            Entry("Time", "When the RFQ arrived (session clock)."),
            Entry("RFQ", "The RFQ's number in the session."),
            CLIENT,
            Entry("Size", "Lots the client asked for."),
            Entry("Type", "Two-way, or the side a one-way client asked to trade: client buys or client sells."),
            Entry("Your bid / ask", "What you quoted; — when you passed."),
            Entry("Outcome", "Won (you traded), done away (another dealer did), not done (the client walked) or passed."),
            Entry(
                "What you learned",
                "Won: the cover, the next-best price. Done away: the client's side and the level it traded at. "
                "Otherwise nothing.",
            ),
            VS_BEST,
            Entry("Pos", "Your position after the ticket, in lots: + long, − short."),
        ),
    )
    t["play.pnl_tiles"] = Table(
        "Session P&L",
        "Every trade marked to the screen showing now; the three tiles add up to the P&L.",
        (
            Entry(
                "Edge · client fills",
                "Each client fill's size × (screen then − price): what you charged clients against the screen. + is in "
                "your favour.",
            ),
            Entry(
                "Broker charge · hedges + close",
                "The same for your hedges and the close: what the broker charged against the screen. Never +.",
            ),
            Entry(
                "Moves since the trade",
                "Every trade's size × (screen now − screen then): what the market did to what you held. + when it "
                "moved your way.",
            ),
            Entry("P&L", "The three added up: your P&L marked to the screen now, as in the top bar."),
        ),
    )
    t["play.pnl"] = Table(
        "Ticket by ticket",
        "One row per trade; the rows add up to the P&L above. Against the screen: the truth stays hidden until the "
        "reveal.",
        (
            Entry("Time", "When the trade happened (session clock)."),
            Entry("Ticket", "RFQ n for a client fill, Hedge, or Close (the liquidation at the end)."),
            Entry("With", "The client (number · category), the broker for a hedge, or the close."),
            Entry("You", "What you did: bought or sold, and how many lots."),
            Entry("Price", "The price you traded at."),
            Entry("Screen then", "The screen when you traded; for a hedge or the close, the screen the broker priced off."),
            Entry("Edge", "Size × (screen then − price), in dollars: + when you sold above or bought below the screen."),
            Entry("Move since", "Size × (screen now − screen then): + when the market has moved your way since."),
            Entry("Total", "Edge + move since: the trade's part of your P&L now. The column adds up to the P&L."),
        ),
    )
    t["play.clients"] = Table(
        "Clients",
        "Your trades with each client so far, marked against the screen.",
        (
            CLIENT,
            CATEGORY,
            RFQS,
            QUOTED,
            Entry("Traded", "RFQs you traded: the client dealt on your price."),
            Entry("Hit rate", "Traded ÷ quoted."),
            LOTS_GROSS,
            NET,
            Entry("Last trade", "How long ago you last traded with the client."),
            _screen_markout_per_lot("60s"),
            _lots_in("60s"),
            _screen_markout_per_lot("300s"),
            _lots_in("300s"),
        ),
    )
    t["play.roster"] = Table(
        "Clients",
        "The roster at a glance, against the screen; the ticket's client is highlighted.",
        (
            CLIENT,
            RFQS,
            Entry("Traded", "RFQs you traded: the client dealt on your price."),
            NET,
            _screen_markout_per_lot("60s"),
            _screen_markout_per_lot("300s"),
        ),
    )
    t["play.calendar"] = Table(
        "Print calendar",
        "The scheduled prints: times only, never outcomes.",
        (
            Entry("Print", "The print's number."),
            Entry("Time", "When it is released (session clock)."),
            Entry("Status", "Done, next (with the time to go) or later."),
        ),
    )
    t["play.log"] = Table(
        "Event log",
        "Every public event in order. Tick the kinds to show.",
        (
            Entry("Time", "When it happened (session clock)."),
            Entry("#", "The event's sequence number."),
            Entry("Kind", "What kind of event."),
            Entry("What", "What happened, in words."),
        ),
    )

    # — the reveal --------------------------------------------------------------
    t["reveal.chart"] = Table(
        "What really happened",
        "The session against the true value. Drag to zoom, double-click to reset, hover a fill for its ticket.",
        (
            Entry("True fair value", "What the market was really worth, every second; hidden until now."),
            Entry("Screen", "What you saw: the market's noisy best estimate."),
            Entry("Print run-up", "The drift into a print, shaded from its start to the print."),
            Entry(
                "Unannounced move",
                "Shaded from the leak — when informed clients first heard — to the end of its ramp.",
            ),
            Entry("Your fills", "▲ you bought, ▼ you sold: blue buys, orange sells."),
            Entry("Ring", "A fill on an informed RFQ."),
            Entry("Hedge", "■ a hedge with the broker, or the close."),
            Entry("Print", "Dashed: each print, with its jump in bp."),
            Entry("Bots' fills", "Outlined, when ticked: the disciplined bot's (grey) and the flow oracle's (purple)."),
            Entry("Moment", "Solid, labelled: the moment you clicked. Click it again to clear it."),
        ),
    )
    t["reveal.position"] = Table(
        "Position",
        "Your position against the benchmarks, on the same time axis as the chart above.",
        (
            Entry("You", "Your position in lots: + long, − short."),
            Entry("Disciplined bot", "The disciplined bot's position on the same world (public information, fixed rules)."),
            Entry("Flow oracle", "The flow oracle's: the disciplined bot, told which RFQs are informed."),
        ),
    )
    t["reveal.score"] = Table(
        "Score",
        "Everyone on the same world and seed. Flow-reading capture: " + capture_rule + " " + pooled_rule,
        (
            *score_cols,
            Entry("You", "Your session."),
            Entry("Naive", "A fixed half-width around the screen; never hedges."),
            Entry("Disciplined bot", "Public information only, fixed rules (listed below)."),
            Entry("Flow oracle", "The disciplined bot, told which RFQs are informed — not what about or which way."),
        ),
    )
    t["reveal.breakdown"] = Table(
        "Where it came from",
        "Your P&L split against the true value; every line is a signed contribution and the lines add up to the "
        "total. The switch picks the mark-out horizon adverse selection is measured at. The same numbers as "
        "summary.md.",
        (
            breakdown_lines[0],
            *(
                Entry(
                    f"Adverse selection · {label}",
                    f"The mark-outs of {what} at that horizon. − when the flow moved against you.",
                )
                for _, label, what in (
                    ("uninformed", "uninformed", by_type[0][2]),
                    ("informed_print", "informed, prints", by_type[1][2]),
                    ("informed_move", "informed, moves", by_type[2][2]),
                )
            ),
            *breakdown_lines[2:],
        ),
    )
    clients_top = (
        CLIENT,
        TRUE_TYPE,
        KNEW_ABOUT,
        INFORMED_RFQS,
        LOTS_GROSS,
        NET,
        _truth_markout_entry("300s"),
        _truth_markout_entry(NEXT),
        Entry("Informed fills", INFORMED_FILLS),
        Entry("Other fills", OTHER_FILLS),
    )
    t["reveal.clients_top"] = Table(
        "Clients, revealed",
        "The five clients that cost or made you the most through the next print or move. Mark-outs are against "
        "the true value, in dollars.",
        clients_top,
    )
    t["reveal.moments"] = Table(
        "Three moments",
        "The three largest by size, whichever kind: every print, every move and every ticket is a candidate. Ties "
        "go to the earlier. Click one to mark it on the charts; click it again to clear it.",
        (
            Entry("Print", "The position you carried into it × its jump."),
            Entry("Unannounced move", "The position you held over each step of its ramp × the move on that step."),
            Entry("Informed ticket", "A fill on an informed RFQ: its price against the truth through its next print or move."),
            Entry(
                "Other ticket",
                "Any other fill: its spread plus its 300s mark-out, stopped short of a print or a move — those are "
                "their own moments.",
            ),
        ),
    )
    t["reveal.clients"] = Table(
        "Clients, revealed",
        "Every client, with its true type. Mark-outs are against the true value, in dollars.",
        (
            CLIENT,
            CATEGORY,
            TRUE_TYPE,
            RFQS,
            INFORMED_RFQS,
            QUOTED,
            WON,
            LOTS_GROSS,
            NET,
            _truth_markout_entry("60s"),
            _truth_markout_entry("300s"),
            _truth_markout_entry(NEXT),
            Entry("All fills", "The mark-out to the next print or move over all your fills with the client: the two after it add up to this."),
            Entry("Informed fills", INFORMED_FILLS),
            Entry("Other fills", OTHER_FILLS),
            KNEW_ABOUT,
        ),
    )
    tickets = (
        Entry("Time", "When the RFQ arrived (session clock)."),
        Entry("RFQ", "The RFQ's number in the session."),
        CLIENT,
        Entry(
            "Who it was",
            "Revealed: uninformed, or informed about a print or a move, with the way it was told and whether "
            "that was right.",
        ),
        Entry("Size", "Lots the client asked for."),
        Entry("Type", "Two-way or one-way, and the client's true side."),
        Entry("Your quote", "Your bid / ask; passed when you passed."),
        Entry(
            "Outcome",
            "Won (your size and price), done away (the level), not done or passed; on the reveal, not shown for "
            "the RFQs after you quit.",
        ),
        VS_BEST,
        Entry("Your position after", "Your position once the ticket was done, in lots: + long, − short."),
        _truth_markout_entry("60s"),
        _truth_markout_entry("300s"),
        _truth_markout_entry(NEXT),
        Entry(
            "Flow oracle",
            "What the flow oracle quoted on the same RFQ and what happened: traded (its size), done away, not "
            "done, or passed.",
        ),
    )
    t["reveal.tickets"] = Table(
        "Tickets",
        "Every RFQ of the session: what you saw, what you did, what was true, what the flow oracle did. Mark-outs "
        "are against the true value, in dollars, for the RFQs you traded.",
        tickets,
    )
    t["reveal.wins"] = Table(
        "Wins vs best dealer, by side",
        "Each win's vs best dealer at its time, beside the screen's error. A run of wide wins on one side while the "
        "screen was off one way is that error, priced into your quotes and picked off by the clients. Wins with "
        "nobody else in have no cover and are not drawn. Drag to zoom, double-click to reset, hover a point for "
        "its ticket.",
        (
            Entry(
                "Client sold to you",
                "● a win where the client sold to you (you bought): how far your bid beat the best other bid, in bp.",
            ),
            Entry(
                "Client bought from you",
                "● a win where the client bought from you (you sold): how far your offer beat the best other offer, "
                "in bp.",
            ),
            Entry(
                "Screen − truth",
                "The screen you priced from minus the true value, every second, in bp: + when the screen read rich. "
                "Rich, clients sell to you at a bid that is too high; cheap, they buy at an offer that is too low.",
            ),
        ),
    )
    t["reveal.moves"] = Table(
        "Unannounced moves",
        "Each unannounced move, and how you and the benchmarks were leaning while it leaked.",
        (
            Entry("Move", "The move's number and when its ramp started (session clock)."),
            Entry("Size", "How far it moved the true value, in bp."),
            Entry("Leaked", "How long before the ramp the first informed client heard: the window the lean is over."),
            Entry("Lean before the move", LEAN_BEFORE_MOVE + " In brackets, how many quotes it covers; — with none."),
            Entry("You", "Your lean before the move, in bp: + toward it. It includes your inventory skew."),
            Entry("Flow oracle", "The flow oracle's lean, the same way: it includes its lean on informed flow."),
            Entry("Disciplined bot", "The disciplined bot's lean, the same way: inventory skew only."),
        ),
    )
    t["reveal.track_gaps"] = Table(
        "Track record, pooled",
        "Over scored sessions only. " + pooled_rule,
        (
            Entry(
                "Against the disciplined bot",
                "Your score, or the flow oracle's, minus the disciplined bot's on the same world: + when ahead of it.",
            ),
            Entry("Pooled", "Summed over every scored session, whatever the oracle's result."),
            Entry("Mean per session", "The pooled figure ÷ the number of scored sessions."),
            Entry(
                "Standard error",
                "Of that mean: the sessions' standard deviation (n − 1) ÷ √n. n/a with one session.",
            ),
        ),
    )
    t["reveal.track"] = Table(
        "Track record",
        "Every session, newest last. " + pooled_rule,
        (
            Entry("P&L", "The columns under it compare P&L."),
            Entry("Risk-adjusted", "The columns under it compare risk-adjusted P&L."),
            Entry("Session", "The session's name: preset, seed, number."),
            Entry("Preset", "Its length in minutes."),
            Entry("Seed", "The world it was played on."),
            Entry("You", "Your score."),
            Entry("Disciplined", "The disciplined bot's score on the same world."),
            Entry("Flow oracle", "The flow oracle's score on the same world."),
            Entry("You − disc.", "Your score minus the disciplined bot's: + when you beat it."),
            Entry("Oracle − disc.", "The flow oracle's minus the disciplined bot's: the flow-reading edge on offer."),
            Entry("Capture", "The capture ratio: " + capture_rule),
            Entry("Status", "Scored, scored · ended early, or a practice replay — shown, never counted."),
        ),
    )

    # — summary.md ----------------------------------------------------------------
    t["summary.score"] = Table(
        "Score",
        "Everyone on the same world and seed.",
        (
            Entry("You", "Your session."),
            Entry("Naive", "A fixed half-width around the screen; never hedges."),
            Entry("Disciplined", "The disciplined bot: public information only, fixed rules."),
            Entry("Flow oracle", "The disciplined bot, told which RFQs are informed — not what about or which way."),
            Entry("Passing oracle*", "Diagnostic: declines every informed RFQ instead of reading it."),
            Entry("P&L", score_cols[0].meaning),
            Entry("Risk penalty", "The charge for the risk carried: about κ × (dollar position)² per print held through, plus a small one between prints."),
            Entry("Risk-adjusted", "P&L minus the risk penalty."),
        ),
    )
    t["summary.track"] = Table(
        "Track record",
        "After this session, over scored sessions only. " + pooled_rule,
        t["reveal.track_gaps"].entries,
    )
    t["summary.breakdown"] = Table(
        "Where the P&L came from",
        "Against the true value; every line is a signed contribution and each column adds up to the total.",
        (*breakdown_lines, *horizon_entries),
    )
    t["summary.by_type"] = Table(
        "Adverse selection by who was on the other side",
        "The adverse selection line split by the RFQ, not the client: these add up to it.",
        (
            *(Entry(label, f"The mark-outs of {what}. − when the flow moved against you.") for _, label, what in by_type),
            *horizon_entries,
        ),
    )
    t["summary.wins"] = Table(
        "Wins against the best other dealer",
        "Your wins by how far your price beat the best other dealer's on the side the client traded (the vs best "
        "dealer column of the tickets table). The rows add up to every win.",
        (
            Entry(gaps["tie"], "Won at the best other dealer's price: a tie that went your way."),
            Entry(gaps["under_half_bp"], f"Beat the best other dealer, by less than {win_half_bp:.2f}bp."),
            Entry(
                gaps["half_bp_or_more"],
                f"Beat it by {win_half_bp:.2f}bp or more: priced well through the street, usually because the "
                "screen was off.",
            ),
            Entry(gaps["nobody_else_in"], "Won with no other dealer in competition: no cover to compare with."),
            Entry(gaps["all"], "Every win; its spread captured is the breakdown's spread line."),
            Entry("Fills", "How many of your wins."),
            Entry("Lots", "Their lots."),
            Entry(
                "Spread captured",
                "Their edge against the true value when you traded: size × (truth − price), in dollars. + is in "
                "your favour.",
            ),
        ),
    )
    t["summary.tickets"] = Table(
        "Tickets",
        "Every RFQ you were shown; the reveal's Tickets tab. Mark-outs are against the true value, in dollars, for "
        "the RFQs you traded.",
        tickets,
    )
    t["summary.clients"] = Table(
        "Clients",
        "Mark-outs are against the true value, in dollars: the P&L from your fills with that client.",
        (
            CLIENT,
            CATEGORY,
            TRUE_TYPE,
            RFQS,
            INFORMED_RFQS,
            QUOTED,
            WON,
            LOTS_GROSS,
            NET,
            _truth_markout_entry("60s"),
            _truth_markout_entry("300s"),
            Entry(f"{NEXT_LABEL}, informed fills", INFORMED_FILLS),
            Entry(f"{NEXT_LABEL}, other fills", OTHER_FILLS),
            KNEW_ABOUT,
        ),
    )
    return t


def md_escape(label: str) -> str:
    """A label inside markdown bold: "Passing oracle*" must not close it."""
    return label.replace("*", r"\*")


# The order summary.md explains its tables in.
SUMMARY_TABLES = (
    "summary.score",
    "summary.track",
    "summary.breakdown",
    "summary.by_type",
    "summary.wins",
    "summary.clients",
    "summary.tickets",
)


def markdown(t: dict[str, Table], keys: tuple[str, ...] = SUMMARY_TABLES) -> list[str]:
    """The definitions for summary.md: one list per table it prints."""
    lines = ["## What the columns mean", ""]
    for key in keys:
        table = t[key]
        lines += [f"**{table.title}.** {table.intro}", ""]
        lines += [f"- **{md_escape(e.label)}**: {e.meaning}" for e in table.entries]
        lines.append("")
    return lines


def message(t: dict[str, Table]) -> dict[str, object]:
    """Every table's definitions, for the page."""
    return {"v": PROTOCOL_VERSION, "type": "glossary", "tables": {k: v.to_record() for k, v in t.items()}}
