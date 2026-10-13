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

## Wins against the best other dealer

Each win by how far your price beat the best other dealer's on the side the client traded (the "vs best dealer" column below). Spread captured is against the true value, as in the breakdown; the rows add up to its spread line.

| | Fills | Lots | Spread captured |
|---|---|---|---|
| 0.00bp (a tie) | 3 | 47 | +$490 |
| 0.25bp | 3 | 32 | +$350 |
| 0.50bp or more | 2 | 52 | −$66 |
| Nobody else in | 0 | 0 | +$0 |
| **All wins** | **8** | **131** | **+$775** |

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
- Print 1 at 18.1 min: -3.84bp, drift into it -1.35bp; you carried +15 lots.
- No unannounced moves this session.

## Quotes against the truth

- Quote mid minus truth: mean -0.14bp, mean |error| 0.41bp over 28 two-sided quotes; against the screen 0.05bp.
- Worst sustained excursion: 2 quotes over 71s from 13.9 min, -1.23bp off the truth on average.

## Tickets

Every RFQ you were shown: who it really was, what you did, and against the true value how the fills marked out, in dollars. The 15 RFQs after you stopped are left out.

| Time | RFQ | Client | Who it was | Size | Type | Your quote | Outcome | vs best dealer | Your position after | Mark-out 60s | Mark-out 300s | Mark-out to next print/move | Flow oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00:16 | 0 | C4 · RM | uninformed | 5 | two-way · client buys | 95.4950 / 95.5050 | done away at 95.5050 | 0.00 | 0 | — | — | — | 95.4950 / 95.5050 → done away |
| 00:46 | 1 | C1 · HF | uninformed | 8 | two-way · client buys | 95.5000 / 95.5100 | done away at 95.5100 | 0.00 | 0 | — | — | — | 95.5000 / 95.5100 → done away |
| 01:00 | 2 | C4 · RM | uninformed | 26 | two-way · client buys | 95.5025 / 95.5125 | done away at 95.5125 | 0.00 | 0 | — | — | — | 95.5025 / 95.5125 → done away |
| 01:09 | 3 | C5 · Corp | uninformed | 6 | two-way · client sells | 95.5000 / 95.5100 | won · +6 at 95.5000 | 0.00 | +6 | −$18 | −$51 | −$1,520 | 95.5000 / 95.5100 → traded +6 |
| 01:12 | 4 | C2 · RM | uninformed | 23 | client buys (one-way) | 95.5000 / 95.5100 | done away at 95.5075 | −0.25 | +6 | — | — | — | 95.5000 / 95.5100 → done away |
| 01:57 | 5 | C0 · HF | uninformed | 8 | two-way · client sells | 95.4925 / 95.5025 | done away at 95.4925 | 0.00 | +6 | — | — | — | 95.4925 / 95.5025 → done away |
| 02:04 | 6 | C4 · RM | uninformed | 10 | two-way · client buys | 95.4925 / 95.5025 | won · −10 at 95.5025 | +0.25 | −4 | −$14 | +$91 | +$2,525 | 95.4925 / 95.5025 → traded −10 |
| 02:07 | 7 | C0 · HF | uninformed | 21 | two-way · client sells | 95.4900 / 95.5000 | done away at 95.4900 | 0.00 | −4 | — | — | — | 95.4900 / 95.5000 → done away |
| 03:11 | 8 | C4 · RM | uninformed | 22 | two-way · client sells | passed | passed |  | −4 | — | — | — | 95.4950 / 95.5050 → traded +22 |
| 06:19 | 9 | C0 · HF | informed · print 1, down (right) | 33 | two-way · client sells | 95.4975 / 95.5075 | won · +33 at 95.4975 | 0.00 | +29 | −$238 | −$1,676 | −$8,246 | 95.4900 / 95.5125 → done away |
| 06:43 | 10 | C3 · RM | uninformed | 17 | two-way · client buys | 95.4900 / 95.5000 | won · −17 at 95.5000 | +0.50 | +12 | +$167 | +$685 | +$4,141 | 95.4875 / 95.5000 → traded −17 |
| 06:48 | 11 | C2 · RM | informed · print 1, down (right) | 22 | two-way · client sells | 95.4900 / 95.5025 | done away at 95.4925 | −0.25 | +12 | — | — | — | 95.4850 / 95.5050 → done away |
| 07:08 | 12 | C3 · RM | uninformed | 7 | two-way · client sells | 95.4900 / 95.5000 | done away at 95.4925 | −0.25 | +12 | — | — | — | 95.4875 / 95.4975 → done away |
| 10:05 | 13 | C1 · HF | uninformed | 12 | two-way · client sells | 95.4800 / 95.4950 | done away at 95.4825 | −0.25 | +12 | — | — | — | 95.4800 / 95.4900 → done away |
| 12:03 | 14 | C3 · RM | uninformed | 9 | two-way · client sells | 95.4825 / 95.4975 | done away at 95.4875 | −0.50 | +12 | — | — | — | 95.4850 / 95.4950 → done away |
| 13:01 | 15 | C5 · Corp | uninformed | 7 | two-way · client sells | 95.4775 / 95.4925 | done away at 95.4850 | −0.75 | +12 | — | — | — | 95.4800 / 95.4900 → done away |
| 13:52 | 16 | C4 · RM | uninformed | 14 | two-way · client sells | 95.4700 / 95.4875 | done away at 95.4800 | −1.00 | +12 | — | — | — | 95.4750 / 95.4850 → done away |
| 14:08 | 17 | C0 · HF | informed · print 1, down (right) | 36 | two-way · client sells | passed | passed |  | +12 | — | — | — | 95.4725 / 95.4925 → done away |
| 15:03 | 18 | C2 · RM | informed · print 1, down (right) | 20 | two-way · client sells | 95.4675 / 95.4850 | done away at 95.4750 | −0.75 | +12 | — | — | — | 95.4650 / 95.4850 → done away |
| 16:36 | 19 | C4 · RM | uninformed | 8 | two-way · client sells | 95.4800 / 95.4950 | won · +8 at 95.4800 | 0.00 | +20 | −$50 | −$1,509 | −$1,573 | 95.4800 / 95.4900 → traded +8 |
| 16:50 | 20 | C0 · HF | informed · print 1, down (right) | 21 | two-way · client sells | 95.4800 / 95.5075 | not done |  | +20 | — | — | — | 95.4775 / 95.5075 → not done |
| 17:26 | 21 | C4 · RM | uninformed | 16 | two-way · client buys | 95.4750 / 95.4900 | not done |  | +10 | — | — | — | 95.4750 / 95.4850 → traded −16 |
| 17:43 | 22 | C3 · RM | uninformed | 5 | two-way · client sells | 95.4750 / 95.4900 | won · +5 at 95.4750 | +0.25 | +15 | −$912 | −$879 | −$943 | 95.4750 / 95.4850 → traded +5 |
| 19:16 | 23 | C1 · HF | uninformed | 18 | client buys (one-way) | 95.4350 / 95.4575 | not done |  | +15 | — | — | — | 95.4350 / 95.4550 → traded −18 |
| 20:18 | 24 | C0 · HF | uninformed | 17 | two-way · client buys | 95.4375 / 95.4500 | won · −17 at 95.4500 | +0.25 | −2 | +$100 | +$44 | — | 95.4400 / 95.4525 → done away |
| 20:20 | 25 | C4 · RM | uninformed | 7 | two-way · client sells | 95.4400 / 95.4500 | done away at 95.4425 | −0.25 | −2 | — | — | — | 95.4425 / 95.4525 → traded +7 |
| 20:50 | 26 | C0 · HF | uninformed | 4 | two-way · client buys | passed | passed |  | −2 | — | — | — | 95.4450 / 95.4550 → not done |
| 20:51 | 27 | C5 · Corp | uninformed | 14 | two-way · client buys | 95.4450 / 95.4550 | done away at 95.4550 | 0.00 | −2 | — | — | — | 95.4450 / 95.4575 → done away |
| 20:55 | 28 | C5 · Corp | uninformed | 35 | two-way · client buys | 95.4400 / 95.4500 | won · −35 at 95.4500 | +0.50 | −37 | +$459 | −$135 | — | 95.4400 / 95.4525 → not done |
| 22:54 | 29 | C2 · RM | uninformed | 26 | two-way · client buys | 95.4475 / 95.4575 | not done |  | −37 | — | — | — | 95.4450 / 95.4575 → not done |
| 23:52 | 30 | C1 · HF | uninformed | 2 | two-way · client buys | 95.4475 / 95.4575 | done away at 95.4550 | −0.25 | −37 | — | — | — | 95.4450 / 95.4550 → traded −2 |

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

**Wins against the best other dealer.** Your wins by how far your price beat the best other dealer's on the side the client traded (the vs best dealer column of the tickets table). The rows add up to every win.

- **0.00bp (a tie)**: Won at the best other dealer's price: a tie that went your way.
- **0.25bp**: Beat the best other dealer, by less than 0.50bp.
- **0.50bp or more**: Beat it by 0.50bp or more: priced well through the street, usually because the screen was off.
- **Nobody else in**: Won with no other dealer in competition: no cover to compare with.
- **All wins**: Every win; its spread captured is the breakdown's spread line.
- **Fills**: How many of your wins.
- **Lots**: Their lots.
- **Spread captured**: Their edge against the true value when you traded: size × (truth − price), in dollars. + is in your favour.

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

**Tickets.** Every RFQ you were shown; the reveal's Tickets tab. Mark-outs are against the true value, in dollars, for the RFQs you traded.

- **Time**: When the RFQ arrived (session clock).
- **RFQ**: The RFQ's number in the session.
- **Client**: The client's number this session, with its category.
- **Who it was**: Revealed: uninformed, or informed about a print or a move, with the way it was told and whether that was right.
- **Size**: Lots the client asked for.
- **Type**: Two-way or one-way, and the client's true side.
- **Your quote**: Your bid / ask; passed when you passed.
- **Outcome**: Won (your size and price), done away (the level), not done or passed; on the reveal, not shown for the RFQs after you quit.
- **vs best dealer**: Your price against the best other dealer's on the side the client traded, in bp: + when you beat it. Won: the cover minus your ask when the client bought, your bid minus the cover when it sold — never −; +0.75 means you beat the best other dealer by 0.75bp, roughly what you gave away. Done away: the same with the level it traded at — −, or 0.00 on a tie you lost. Blank: won with nobody else in, not done, passed, or done away on a side you didn't quote.
- **Your position after**: Your position once the ticket was done, in lots: + long, − short.
- **Mark-out 60s**: For each fill, the size times how far the true value moved from the fill to 60s later, in dollars (+ size when you bought). The edge you charged is left out: that is the spread line. + means the market moved your way. Fills whose 60s ran past the close are left out.
- **Mark-out 300s**: For each fill, the size times how far the true value moved from the fill to 300s later, in dollars (+ size when you bought). The edge you charged is left out: that is the spread line. + means the market moved your way. Fills whose 300s ran past the close are left out.
- **Mark-out to next print/move**: The same as the other mark-outs, measured from each fill to the next print or the end of the next unannounced move, whichever comes first. + means it moved your way. Fills with neither after them are left out.
- **Flow oracle**: What the flow oracle quoted on the same RFQ and what happened: traded (its size), done away, not done, or passed.
