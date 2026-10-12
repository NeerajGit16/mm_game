# Session summary — 30 minutes (preset), seed 3

**Ended early at step 15000** (25.0 min); the position rode to the close.

## Score

| | You | Naive | Disciplined | Flow oracle | Passing oracle* |
|---|---|---|---|---|---|
| P&L | -$4,667 | -$17,861 | -$4,711 | $1,731 | -$3,043 |
| Risk penalty | $128 | $3,200 | $458 | $6 | $337 |
| Risk-adjusted | -$4,795 | -$21,061 | -$5,169 | $1,725 | -$3,379 |

*Diagnostic: the passing oracle declines every informed RFQ instead of reading it.

Flow-reading capture ratio against the flow oracle: P&L 0.01, risk-adjusted 0.05 (n/a unless the oracle beats the disciplined bot by at least $2,000; this session it did by $6,442 on P&L, $6,895 risk-adjusted).

## Track record

After this session the track record holds 1 scored session.

| Against the disciplined bot | Pooled | Mean per session | Standard error |
|---|---|---|---|
| You, P&L | +$44 | +$44 | n/a |
| Flow oracle, P&L | +$6,442 | +$6,442 | n/a |
| You, risk-adjusted | +$374 | +$374 | n/a |
| Flow oracle, risk-adjusted | +$6,895 | +$6,895 | n/a |

Pooled capture ratio: P&L n/a — not enough flow-reading edge yet ($6,442 of $10,000); risk-adjusted n/a — not enough flow-reading edge yet ($6,895 of $10,000).

## Where the P&L came from, against true fair value

Every line is a contribution to P&L; each column adds up to the total. Adverse selection is your fills' mark-outs at that horizon — negative when the flow moved against you. The mark-out to next print/move runs each fill through the next print or the end of the next unannounced move, whichever comes first.

| | Mark-out 10s | Mark-out 60s | Mark-out 300s | Mark-out to next print/move |
|---|---|---|---|---|
| Spread captured | +$775 | +$775 | +$775 | +$775 |
| Adverse selection | −$94 | −$507 | −$3,431 | −$5,615 |
| Inventory beyond the mark-outs | −$3,254 | −$2,841 | +$83 | +$2,267 |
| Hedging and closing | −$2,094 | −$2,094 | −$2,094 | −$2,094 |
| **Total** | **−$4,667** | **−$4,667** | **−$4,667** | **−$4,667** |

Adverse selection by who was on the other side (these add up to the line above):

| | Mark-out 10s | Mark-out 60s | Mark-out 300s | Mark-out to next print/move |
|---|---|---|---|---|
| Uninformed | +$101 | −$269 | −$1,755 | +$2,631 |
| Informed about a print | −$195 | −$238 | −$1,676 | −$8,246 |
| Informed about a move | +$0 | +$0 | +$0 | +$0 |

Hedging and closing splits into the broker's charge −$4,017 and the screen's error when you hedged +$1,924. Fills left out of adverse selection (mark-out cut off by the close, or no print or move after): next print/move: 2; their moves stay in inventory.

## Flow

- Hit rate 29% of quoted RFQs; by count in competition: 0: 0%, 1: 62%, 2: 20%, 3: 0%.
- Fills: 7 uninformed (98 lots); 1 informed about a print (33 lots); 0 informed about a move (0 lots) — 25% of volume against informed clients.

## Clients

Mark-outs are P&L contributions from your fills with that client. True type and what each informed client knew are revealed now. The mark-out to next print/move is split by ticket: fills on informed RFQs, and the rest — an informed client asking outside its information window is an uninformed RFQ.

| Client | Category | True type | RFQs | Informed RFQs (filled) | Quoted | Won | Lots traded (gross) | Your net (+ = you bought) | Mark-out 60s | Mark-out 300s | Mark-out to next print/move, informed fills | Mark-out to next print/move, other fills | Knew about |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | hedge fund | informed | 7 | 3 (1) | 5 | 2 | 50 | +16 | −$139 | −$1,632 | −$8,246 | +$0 | print 1: down (right) |
| 1 | hedge fund | uninformed | 4 | 0 (0) | 4 | 0 | 0 | +0 | — | — | — | — | — |
| 2 | real money | informed | 4 | 2 (0) | 4 | 0 | 0 | +0 | — | — | — | — | print 1: down (right) |
| 3 | real money | uninformed | 4 | 0 (0) | 4 | 2 | 22 | -12 | −$745 | −$194 | — | +$3,198 | — |
| 4 | real money | uninformed | 8 | 0 (0) | 7 | 2 | 18 | -2 | −$65 | −$1,419 | — | +$952 | — |
| 5 | corporate | uninformed | 4 | 0 (0) | 4 | 2 | 41 | -29 | +$440 | −$186 | — | −$1,520 | — |

## Position, prints and moves

- Max |position| 37 lots; time-weighted mean |position| 18.
- Print 1 at 18.1 min: -3.84bp, drift into it -1.35bp; you carried +15 lots; quotes reflected it after 72s.
- No unannounced moves this session.

## Quotes against the truth

- Quote mid minus truth: mean -0.14bp, mean |error| 0.41bp over 28 two-sided quotes; against the screen 0.05bp.
- Worst sustained excursion: 2 quotes over 71s from 13.9 min, -1.23bp off the truth on average.

## What the columns mean

**Score.** Everyone on the same world and seed.

- **You**: Your session.
- **Naive**: A fixed half-width around the screen; never hedges.
- **Disciplined**: The disciplined bot: public information only, fixed rules.
- **Flow oracle**: The disciplined bot, told which RFQs are informed — not what about or which way.
- **Passing oracle\***: Diagnostic: declines every informed RFQ instead of reading it.
- **P&L**: Realised P&L in dollars after the close, on the same world and seed for everyone.
- **Risk penalty**: The charge for the risk carried: about κ × (dollar position)² per print held through, plus a small one between prints.
- **Risk-adjusted**: P&L minus the risk penalty.

**Track record.** After this session, over scored sessions only. The pooled ratio is the same over all scored sessions summed; it stays n/a until the oracle's pooled edge over the disciplined bot reaches $10,000. Practice replays are shown, never counted.

- **Against the disciplined bot**: Your score, or the flow oracle's, minus the disciplined bot's on the same world: + when ahead of it.
- **Pooled**: Summed over every scored session, whatever the oracle's result.
- **Mean per session**: The pooled figure ÷ the number of scored sessions.
- **Standard error**: Of that mean: the sessions' standard deviation (n − 1) ÷ √n. n/a with one session.

**Where the P&L came from.** Against the true value; every line is a signed contribution and each column adds up to the total.

- **Spread captured**: Each client fill's edge against the true value when you traded: size × (truth − price). + is in your favour.
- **Adverse selection**: Your client fills' mark-outs at that horizon: size × how far the truth moved from the fill. − when the flow moved against you.
- **Inventory beyond the mark-outs**: What the truth did to your position after each fill's mark-out had run: + when it moved your way.
- **Hedging and closing**: Your hedges and the close against the truth: the broker's charge plus the screen's error when you traded. Usually −: they fill off the screen, not the true value.
- **Total**: Your P&L. Every column adds up to it.
- **Mark-out 10s**: Adverse selection measured over 10s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out 60s**: Adverse selection measured over 60s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out 300s**: Adverse selection measured over 300s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out to next print/move**: Adverse selection measured to the next print or the end of the next unannounced move, whichever comes first.

**Adverse selection by who was on the other side.** The adverse selection line split by the RFQ, not the client: these add up to it.

- **Uninformed**: The mark-outs of fills on uninformed RFQs (an informed client's RFQs outside its window included). − when the flow moved against you.
- **Informed about a print**: The mark-outs of fills on RFQs from a client that knew about a print. − when the flow moved against you.
- **Informed about a move**: The mark-outs of fills on RFQs from a client that knew about an unannounced move. − when the flow moved against you.
- **Mark-out 10s**: Adverse selection measured over 10s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out 60s**: Adverse selection measured over 60s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out 300s**: Adverse selection measured over 300s: inventory beyond takes the rest, so the column still adds up.
- **Mark-out to next print/move**: Adverse selection measured to the next print or the end of the next unannounced move, whichever comes first.

**Clients.** Mark-outs are against the true value, in dollars: the P&L from your fills with that client.

- **Client**: The client's number this session, with its category.
- **Category**: Hedge fund, real money or corporate: shown on every ticket.
- **True type**: Revealed now. Informed: some of its RFQs came knowing about a print or an unannounced move. Uninformed: none did.
- **RFQs**: RFQs the client sent you.
- **Informed RFQs (filled)**: Of its RFQs, how many came inside its information window, knowing about a print or a move; in brackets, how many of those you filled. An informed client's RFQs outside its window count as uninformed.
- **Quoted**: RFQs you quoted rather than passed (a pricing timer that ran out counts as a pass).
- **Won**: RFQs you traded: the client dealt on your price.
- **Lots traded (gross)**: Your buys plus your sells with the client, in lots: never negative.
- **Your net (+ = you bought)**: Your net position from trades with the client, in lots: + when on balance you bought from them, − when you sold.
- **Mark-out 60s**: For each fill, the size times how far the true value moved from the fill to 60s later, in dollars (+ size when you bought). The edge you charged is left out: that is the spread line. + means the market moved your way. Fills whose 60s ran past the close are left out.
- **Mark-out 300s**: For each fill, the size times how far the true value moved from the fill to 300s later, in dollars (+ size when you bought). The edge you charged is left out: that is the spread line. + means the market moved your way. Fills whose 300s ran past the close are left out.
- **Mark-out to next print/move, informed fills**: The mark-out to the next print or move from your fills on its informed RFQs only. — when you filled none of them.
- **Mark-out to next print/move, other fills**: The same from all your other fills with the client, including an informed client's RFQs outside its window. — when there were none.
- **Knew about**: For an informed client: each print or move it knew about, the way it was told the market would go, and whether that was right.
