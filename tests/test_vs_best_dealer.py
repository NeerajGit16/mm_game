"""Tests for your price against the best other dealer ("vs best dealer"):
the play blotter's column, the record behind the reveal's Tickets tab and
summary.md's tickets table, summary.md's wins bucketed by it, and the
screen's error the reveal draws beside it.

What can go wrong silently: a sign flipped on one side (a bid compared as if
it were an ask), a lost tie printed as -0.00, a win with nobody else in
printed as 0.00 as if it had tied, a done-away on a side you never quoted
compared with a price you never showed, the record computing it differently
from the play screen, buckets that do not add up to the spread line, and a
screen error drawn with its sign the wrong way round. Each docstring says
which.

The worked examples are from session 2 as played (30 minutes, seed
1793618369), replayed from its saved replay file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mm_game import agents, desk, protocol, records
from mm_game.book import Source

SESSION_2 = Path(__file__).resolve().parent / "fixtures" / "play" / "30m-1793618369-0002"


@pytest.fixture(scope="module")
def session_2():
    """Session 2 rebuilt from its replay file: the world, the session, its
    coaching record, and the desk fed every public event the page was sent."""
    rf = records.load_replay(json.loads((SESSION_2 / "replay.json").read_text(encoding="utf-8")))
    session, w = records.replay(rf)
    bots = {b: agents.run_bot(b, w, settings=rf.settings, scoring=rf.scoring, benchmark=rf.benchmark) for b in agents.BOTS}
    record = records.coaching_record(w, session, scoring=rf.scoring, benchmark=rf.benchmark, bots=bots, flags=rf.flags)
    d = desk.Desk()
    for e in session.log:
        d.feed(protocol.event_message(e))
    return w, session, record, d.message()


def md_table(md: str, header: str) -> list[dict[str, str]]:
    """A summary.md table's rows, cell by header, from its header line."""
    lines = md.splitlines()
    i = lines.index(header)
    heads = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        rows.append(dict(zip(heads, [c.strip() for c in ln.strip("|").split("|")], strict=True)))
    return rows


def test_the_rule_case_by_case():
    """Each case of the definition on hand-made numbers. Won: the cover
    minus your ask when the client bought, your bid minus the cover when it
    sold. Done away: the same with the level. A tie is 0.0 on either side --
    on the bid side the arithmetic gives -0.0, which would print as -0.00.
    Blank (None) when there is nothing to compare: a win with nobody else
    in, not done, passed, and a done-away on the side you did not quote,
    which must not be compared with the price you never showed."""
    f = desk.vs_best_dealer_bp
    assert f("won", 1, 9549.25, 9549.75, 9550.50) == 0.75  # client bought: 9550.50 - 9549.75
    assert f("won", -1, 9549.50, 9550.00, 9549.00) == 0.50  # client sold: 9549.50 - 9549.00
    assert f("done_away", 1, 9551.25, 9552.75, 9552.50) == -0.25  # 9552.50 - 9552.75
    assert f("done_away", -1, 9550.50, 9551.00, 9550.75) == -0.25  # 9550.50 - 9550.75
    for side, bid, ask, level in ((1, 9549.75, 9550.25, 9550.25), (-1, 9550.50, 9551.00, 9550.50)):
        tie = f("done_away", side, bid, ask, level)
        assert tie == 0.0 and math.copysign(1.0, tie) == 1.0 and records.gap_text(tie) == "0.00"
    assert f("won", 1, 9549.25, 9549.75, None) is None  # nobody else in: no cover
    assert f("not_done", None, 9549.25, 9549.75, None) is None
    assert f("passed", None, None, None, None) is None
    assert f("done_away", -1, None, 9551.75, 9551.25) is None  # client sold; you quoted only the offer
    assert f("done_away", 1, None, 9551.75, 9551.50) == -0.25  # client bought on the side you quoted
    assert [records.gap_text(x) for x in (0.75, -0.25, -1.0, None)] == ["+0.75", "−0.25", "−1.00", ""]


# Session 2, worked by hand from the blotter: (your bid, your ask, cover or
# level) and the result, signed from your side.
SESSION_2_BY_HAND = {
    0: ("won", "client sold: bid 9549.50 - cover 9549.00", 0.50),
    1: ("won", "nobody else in (0 in competition): no cover", None),
    2: ("done_away", "client bought: level 9550.25 - ask 9550.25, a lost tie", 0.0),
    3: ("won", "client bought: cover 9550.50 - ask 9549.75", 0.75),
    4: ("won", "client sold: bid 9548.75 - cover 9548.75, a tie you won", 0.0),
    7: ("done_away", "client sold: bid 9549.25 - level 9549.50", -0.25),
    9: ("done_away", "client sold: bid 9550.50 - level 9550.50, a lost tie on the bid", 0.0),
    12: ("done_away", "one-way, client bought: level 9551.50 - ask 9551.75", -0.25),
    15: ("not_done", "the client walked", None),
    42: ("done_away", "client bought: level 9555.25 - ask 9556.25", -1.00),
    48: ("won", "client sold: bid 9554.50 - cover 9554.00", 0.50),
}


def test_session_2s_blotter_against_a_hand_working(session_2):
    """The play screen's blotter column, from the desk fed only the public
    events, for session 2's tickets worked by hand: a win by half a bp each
    way, a win with nobody else in (blank), a lost tie on each side (0.00,
    not -0.00), a tie won, a done-away a full bp through you, a one-way
    done-away on the side you quoted, and a client who walked (blank)."""
    _, _, _, msg = session_2
    rows = {r["ordinal"]: r for r in msg["blotter"]}
    for ordinal, (outcome, why, expected) in SESSION_2_BY_HAND.items():
        row = rows[ordinal]
        assert row["outcome"] == outcome, ordinal
        assert row["vs_best_dealer_bp"] == expected, (ordinal, why)
        if expected == 0.0:
            assert math.copysign(1.0, row["vs_best_dealer_bp"]) == 1.0, (ordinal, "a tie must not be -0.0")


def test_the_record_and_summary_md_agree_with_the_play_screen(session_2):
    """Every RFQ of session 2: the coaching record's figure (behind the
    reveal's Tickets tab) is the one the desk showed in play, and summary.md's
    tickets table prints it as the blotter does. The hand-worked tickets
    read +0.50, blank, 0.00, ... cell by cell. A second implementation in the
    record would drift from the screen the first time either changed."""
    _, _, record, msg = session_2
    shown = {r["ordinal"]: r["vs_best_dealer_bp"] for r in msg["blotter"]}
    layers = {r["ordinal"]: r["vs_best_dealer_bp"] for r in record["rfqs"] if r["shown"]}
    assert layers == shown and len(shown) == 49
    tickets = {int(r["RFQ"]): r for r in md_table(records.markdown_summary(record), "| " + " | ".join(
        records.glossary_tables(record["header"]["config"], record["capture_ratio"]["min_gap_usd"])["summary.tickets"].labels()
    ) + " |")}
    assert set(tickets) == set(shown)
    for ordinal, value in shown.items():
        assert tickets[ordinal]["vs best dealer"] == records.gap_text(value), ordinal
    assert {o: tickets[o]["vs best dealer"] for o in SESSION_2_BY_HAND} == {
        0: "+0.50", 1: "", 2: "0.00", 3: "+0.75", 4: "0.00", 7: "−0.25", 9: "0.00", 12: "−0.25", 15: "", 42: "−1.00",
        48: "+0.50"}
    assert tickets[2]["Outcome"] == "done away at 95.5025" and tickets[1]["Outcome"] == "won · +11 at 95.4925"


# Session 2's wins by bucket, by hand from the blotter: RFQ (lots).
WINS_BY_HAND = {
    "tie": {4: 21, 35: 14, 40: 3},
    "under_half_bp": {6: 4, 14: 3, 25: 9, 33: 21, 38: 5, 39: 21, 43: 10, 47: 25},
    "half_bp_or_more": {0: 11, 3: 7, 8: 24, 10: 17, 19: 19, 21: 5, 27: 7, 28: 6, 48: 21},
    "nobody_else_in": {1: 11, 5: 3},
}


def test_session_2s_wins_bucketed_by_hand(session_2):
    """The review of session 2 counted nine wins that beat the best other
    dealer by 0.5bp or more: 117 lots and about nothing in spread. By hand:
    three ties won (38 lots), eight by one tick (98), those nine (117), and
    two with nobody else in (14) -- 22 wins, 267 lots. Each bucket's spread
    captured is worked out from the truth at those fills, and the rows add
    up to the breakdown's spread line. summary.md prints exactly these rows.
    A bucket boundary off by a tick, a win with no cover counted as a tie,
    or buckets that skip a win would all change a row here."""
    w, session, record, _ = session_2
    usd = w.config.dollars_per_bp_per_lot
    fills = {t.rfq_ordinal: t for t in session.book.trades if t.source is Source.CLIENT}
    assert set(fills) == {o for bucket in WINS_BY_HAND.values() for o in bucket}
    got = record["diagnostics"]["wins_vs_best_dealer"]
    for bucket, wins in WINS_BY_HAND.items():
        assert {o: abs(fills[o].quantity) for o in wins} == wins, bucket
        spread = sum(fills[o].quantity * (float(w.path.values[fills[o].step]) - fills[o].price) * usd for o in wins)
        assert got[bucket] == {"fills": len(wins), "lots": sum(wins.values()), "spread_captured_usd": pytest.approx(spread)}
    assert sum(b["spread_captured_usd"] for b in got.values()) == pytest.approx(record["decomposition"]["spread_captured_usd"])
    md = records.markdown_summary(record)
    rows = [ln for ln in md.splitlines()[md.splitlines().index("| | Fills | Lots | Spread captured |") + 2:]][:5]
    assert rows == [
        "| 0.00bp (a tie) | 3 | 38 | +$1,516 |",
        "| 0.25bp | 8 | 98 | +$392 |",
        "| 0.50bp or more | 9 | 117 | +$2 |",
        "| Nobody else in | 2 | 14 | +$370 |",
        "| **All wins** | **22** | **267** | **+$2,280** |",
    ]


def test_the_middle_bucket_is_named_by_the_ticks_it_holds():
    """At the quarter-bp tick the bucket between a tie and half a bp holds
    one tick, 0.25bp, and says so; at a tenth of a bp it holds four ticks.
    A label typed as "0.25bp" would misname the bucket on any other tick."""
    from mm_game import glossary

    assert glossary.win_gap_labels(0.25, 0.5)["under_half_bp"] == "0.25bp"
    assert glossary.win_gap_labels(0.1, 0.5)["under_half_bp"] == "0.10–0.40bp"
    assert glossary.win_gap_labels(0.5, 0.5)["under_half_bp"] == "under 0.50bp"


def test_the_screen_error_is_the_screen_minus_the_truth_every_second(session_2):
    """The line the wins chart draws is the screen showing minus the true
    value on the truth's own one-second grid, worked out from the public
    screen and the path. By hand at 14:55 and 15:21, where RFQs 27 and 28
    bought from clients 0.75bp and 0.5bp through the street: the screen read
    95.5575 against a truth of 95.545923 and 95.545468 -- 1.16bp and 1.20bp
    rich, so the bids were too high. And at each fill, the record's screen
    mid less its true value, which the chart's hover shows: RFQ 27, 9555.75 -
    9554.6043 = +1.1457. A sign the wrong way would show those wins as the
    screen reading cheap."""
    w, _, record, _ = session_2
    err = record["reveal"]["screen_error"]
    every = err["every_steps"]
    assert every == round(1.0 / w.config.dt) == 10
    assert len(err["bp"]) == len(record["reveal"]["truth"]["values"])
    view, path = w.path.public, w.path
    assert err["bp"] == [round(view.screen_at(k) - float(path.values[k]), 4) for k in range(0, path.n_steps + 1, every)]
    assert err["bp"][895] == 1.1577 and err["bp"][921] == 1.2032  # 9555.75 - 9554.5923; 9555.75 - 9554.5468
    rfq27 = record["rfqs"][27]
    assert rfq27["observable"]["screen_mid"] - rfq27["hidden"]["true_fair_value"] == pytest.approx(1.1457, abs=1e-4)
