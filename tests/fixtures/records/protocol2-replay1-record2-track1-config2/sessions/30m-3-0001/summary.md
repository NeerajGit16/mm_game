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

## Where the P&L came from, against true fair value

Every line is a contribution to P&L; each column adds up to the total. Adverse selection is your fills' mark-outs at that horizon — negative when the flow moved against you. Next event marks each fill through the next print or the end of the next unannounced move, whichever comes first.

| | 10s | 60s | 300s | next event |
|---|---|---|---|---|
| Spread captured | +$775 | +$775 | +$775 | +$775 |
| Adverse selection | −$94 | −$507 | −$3,431 | −$5,615 |
| Inventory beyond the mark-outs | −$3,254 | −$2,841 | +$83 | +$2,267 |
| Hedging and closing | −$2,094 | −$2,094 | −$2,094 | −$2,094 |
| **Total** | **−$4,667** | **−$4,667** | **−$4,667** | **−$4,667** |

Adverse selection by who was on the other side (these add up to the line above):

| | 10s | 60s | 300s | next event |
|---|---|---|---|---|
| Uninformed | +$101 | −$269 | −$1,755 | +$2,631 |
| Informed about a print | −$195 | −$238 | −$1,676 | −$8,246 |
| Informed about a move | +$0 | +$0 | +$0 | +$0 |

Hedging and closing splits into the venue's charge −$4,017 and the screen's error when you hedged +$1,924. Fills left out of adverse selection (mark-out cut off by the close, or no print or move after): next event: 2; their moves stay in inventory.

## Flow

- Hit rate 29% of quoted RFQs; by count in competition: 0: 0%, 1: 62%, 2: 20%, 3: 0%.
- Fills: 7 uninformed (98 lots); 1 informed about a print (33 lots); 0 informed about a move (0 lots) — 25% of volume against informed clients.

## Clients

Mark-outs are P&L contributions from your fills with that client. True type and what each informed client knew are revealed now.

| Client | Category | True type | RFQs | Quoted | Won | Lots | Your net lots | 60s | 300s | Next event | Knew about |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | hedge fund | informed | 7 | 5 | 2 | 50 | +16 | −$139 | −$1,632 | −$8,246 | print 1: down (right) |
| 1 | hedge fund | uninformed | 4 | 4 | 0 | 0 | +0 | — | — | — | — |
| 2 | real money | informed | 4 | 4 | 0 | 0 | +0 | — | — | — | print 1: down (right) |
| 3 | real money | uninformed | 4 | 4 | 2 | 22 | -12 | −$745 | −$194 | +$3,198 | — |
| 4 | real money | uninformed | 8 | 7 | 2 | 18 | -2 | −$65 | −$1,419 | +$952 | — |
| 5 | corporate | uninformed | 4 | 4 | 2 | 41 | -29 | +$440 | −$186 | −$1,520 | — |

## Position, prints and moves

- Max |position| 37 lots; time-weighted mean |position| 18.
- Print 1 at 18.1 min: -3.84bp, drift into it -1.35bp; you carried +15 lots; quotes reflected it after 72s.
- No unannounced moves this session.

## Quotes against the truth

- Quote mid minus truth: mean -0.14bp, mean |error| 0.41bp over 28 two-sided quotes; against the screen 0.05bp.
- Worst sustained excursion: 2 quotes over 71s from 13.9 min, -1.23bp off the truth on average.
