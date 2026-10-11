// @ts-check
// Messages between the page and the server, and the shapes of what comes
// back. The page speaks the server's protocol version: tests/test_server.py
// checks V against mm_game.protocol.PROTOCOL_VERSION. The shapes below are
// what the server's own tests pin key for key.

export const V = 2;

/** @typedef {1 | -1} Side  The client's side: 1 the client buys, -1 sells. */
/** @typedef {"60s" | "300s"} Horizon */

/**
 * An RFQ as shown. A two-way RFQ never carries the client's side.
 * @typedef {Object} RfqPublic
 * @property {number} ordinal
 * @property {number} client_id
 * @property {string} category
 * @property {number} size
 * @property {boolean} one_way
 * @property {Side | null} side
 * @property {number} in_comp
 */

/** @typedef {RfqPublic & {blocked_sides: ("bid" | "ask")[]}} Ticket */

/**
 * @typedef {Object} NextPrint
 * @property {number} print_index
 * @property {number} step
 * @property {number} in_s
 * @property {number} penalty_if_held_usd
 */

/**
 * @typedef {Object} Book
 * @property {number} position
 * @property {number} realised_usd
 * @property {number} unrealised_usd
 * @property {number} total_usd
 * @property {number} limit
 * @property {number} room_to_buy
 * @property {number} room_to_sell
 */

/**
 * The snapshot after every input or clock tick.
 * @typedef {Object} StateMsg
 * @property {number} v
 * @property {"state"} type
 * @property {number} step
 * @property {number} time_s
 * @property {"live" | "quit" | "ended"} status
 * @property {{speed: number, paused: boolean, awaiting_quote: number | null}} clock
 * @property {{mid: number, age_s: number}} screen
 * @property {Book} book
 * @property {{penalty_so_far_usd: number, next_print: NextPrint | null}} risk
 * @property {Ticket | null} ticket
 * @property {{quotes: {size: number, cost_bp_per_lot: number}[], near_print: boolean}} venue
 */

/**
 * One logged event, its public payload only (declared per kind in events.FIELDS).
 * @typedef {Object} EventMsg
 * @property {number} v
 * @property {"event"} type
 * @property {number} seq
 * @property {number} step
 * @property {string} kind
 * @property {Record<string, any>} data
 */

/**
 * @typedef {Object} BlotterRow
 * @property {number} ordinal
 * @property {number} step
 * @property {number} client_id
 * @property {string} category
 * @property {number} size
 * @property {boolean} one_way
 * @property {Side | null} side
 * @property {number} in_comp
 * @property {number | null} bid
 * @property {number | null} ask
 * @property {"won" | "done_away" | "not_done" | "passed" | null} outcome
 * @property {number} quantity
 * @property {number | null} price
 * @property {Side | null} revealed_side
 * @property {{cover?: number | null, side?: Side, level?: number} | null} learned
 * @property {number | null} position_after
 */

/**
 * @typedef {Object} Interaction
 * @property {number} ordinal
 * @property {number} step
 * @property {number} size
 * @property {string} outcome
 * @property {number} quantity
 * @property {number | null} level
 * @property {Side | null} client_side
 * @property {Record<Horizon, number | null>} markout_bp_per_lot
 */

/**
 * @typedef {Object} ClientDesk
 * @property {number} client_id
 * @property {string} category
 * @property {number} rfqs
 * @property {number} quoted
 * @property {number} traded
 * @property {number} lots
 * @property {number} net
 * @property {number | null} last_trade_step
 * @property {number | null} hit_rate
 * @property {Record<Horizon, number | null>} markout_bp_per_lot
 * @property {Record<Horizon, number>} markout_lots
 * @property {Interaction[]} recent
 */

/**
 * One trade marked to the screen now: edge + move since = total.
 * @typedef {Object} PnlRow
 * @property {number} seq
 * @property {number} step
 * @property {"client" | "hedge" | "close"} kind
 * @property {number | null} ordinal
 * @property {number | null} client_id
 * @property {string | null} category
 * @property {number} quantity
 * @property {number} price
 * @property {number} screen_then
 * @property {number} edge_usd
 * @property {number} move_usd
 * @property {number} total_usd
 */

/**
 * @typedef {Object} PnlTiles
 * @property {number} edge_client_usd
 * @property {number} venue_charge_usd
 * @property {number} move_usd
 * @property {number} total_usd
 * @property {number} realised_usd
 * @property {number} unrealised_usd
 */

/**
 * The player's own numbers, from desk.py.
 * @typedef {Object} DeskMsg
 * @property {number} v
 * @property {"desk"} type
 * @property {number} step
 * @property {BlotterRow[]} blotter
 * @property {ClientDesk[]} clients
 * @property {{rows: PnlRow[], tiles: PnlTiles, series: {step: number[], pnl_usd: number[], position: number[]}}} pnl
 */

/**
 * @typedef {Object} HostMsg
 * @property {number} v
 * @property {"host"} type
 * @property {boolean} begun
 */

/**
 * @typedef {Object} HandbookPart
 * @property {string} title
 * @property {string} subtitle
 * @property {string | null} note
 * @property {{head: string | null, text: string}[]} items
 */

/**
 * @typedef {Object} HandbookMsg
 * @property {number} v
 * @property {"handbook"} type
 * @property {number} scored_sessions
 * @property {number} unlock_after
 * @property {HandbookPart[]} parts
 */

/**
 * @typedef {Object} ErrorMsg
 * @property {number} v
 * @property {"error"} type
 * @property {number} step
 * @property {string} detail
 */

/**
 * A fill or hedge on the reveal, with who was on the other side.
 * @typedef {Object} RevealFill
 * @property {number} seq
 * @property {number} step
 * @property {"client" | "hedge" | "close"} source
 * @property {number} quantity
 * @property {number} price
 * @property {number} edge_usd  quantity x (truth at the fill - price), in dollars
 * @property {number} [rfq_ordinal]
 * @property {number} [client_id]
 * @property {"uninformed" | "informed_print" | "informed_move"} [fill_type]
 * @property {{kind: string, index: number} | null} [informed_about]
 * @property {number} [ticket_usd]  against the truth at the next event
 */

/** @typedef {{steps: number[], positions: number[]}} PositionPath */
/** @typedef {{rfq_ordinal: number, bid: number | null, ask: number | null, outcome: string, quantity: number, price: number | null}} BotRfq */
/** @typedef {{step: number, source: string, quantity: number, price: number, rfq_ordinal: number | null}} BotFill */

/**
 * @typedef {Object} Moment
 * @property {"print" | "move" | "ticket"} kind
 * @property {number} index
 * @property {number} step
 * @property {number} from_step
 * @property {number} to_step
 * @property {number} pnl_usd
 * @property {string} title
 * @property {string} sentence
 */

/** @typedef {{head: string | null, text: string}} RuleItem */

/**
 * @typedef {Object} RevealBlock
 * @property {{every_steps: number, values: number[]}} truth
 * @property {{steps: number[], mids: number[]}} screen
 * @property {{index: number, step: number, jump_bp: number, runup: {start_step: number, end_step: number, total_bp: number} | null}[]} prints
 * @property {{index: number, inform_step: number, start_step: number, end_step: number, size_bp: number}[]} moves
 * @property {RevealFill[]} fills
 * @property {{you: PositionPath, disciplined: PositionPath, oracle: PositionPath}} positions
 * @property {{disciplined: {fills: BotFill[], rfqs: BotRfq[]}, oracle: {fills: BotFill[], rfqs: BotRfq[]}}} bots
 * @property {Moment[]} moments
 * @property {{disciplined: RuleItem[], oracle: RuleItem[]}} rules
 */

/**
 * @typedef {Object} Horizon_
 * @property {number} adverse_selection_usd
 * @property {Record<string, number>} adverse_selection_by_type_usd
 * @property {number} inventory_beyond_usd
 * @property {number} fills_excluded
 */

/**
 * @typedef {Object} ClientReveal
 * @property {string} category
 * @property {boolean} informed_type
 * @property {number} rfqs
 * @property {number} quoted
 * @property {number} won
 * @property {number} lots
 * @property {number} net_lots
 * @property {Record<string, number>} markouts_usd
 * @property {{kind: string, index: number, signal_direction: number, actual_direction: number, right: boolean}[]} informed_about
 */

/**
 * @typedef {Object} RfqLayer
 * @property {number} ordinal
 * @property {number} step
 * @property {boolean} shown
 * @property {{bid: number | null, ask: number | null, by: string} | null} action
 * @property {{outcome: string, side: Side | null, price: number | null, cover: number | null, quantity: number} | null} outcome
 * @property {{side: Side, informed: boolean, signal_bp: number | null, informed_about: {kind: string, index: number} | null,
 *   markouts: Record<string, {pnl_bp: number | null, truncated?: boolean}> | null}} hidden
 */

/** @typedef {{ratio: number | null, oracle_minus_disciplined_usd: number}} CaptureLine */

/**
 * The coaching record, as far as the reveal reads it.
 * @typedef {Object} CoachingRecord
 * @property {number} format_version
 * @property {{seed: number, config: Record<string, any>}} header
 * @property {{preset: number | null}} settings
 * @property {{replay_of_played_seed: boolean, ended_early_at_step: number | null}} flags
 * @property {{pnl_usd: number, risk_adjusted_usd: number}} scores
 * @property {Record<string, {scores: {pnl_usd: number, risk_adjusted_usd: number}}>} benchmarks
 * @property {{min_gap_usd: number, pnl: CaptureLine, risk_adjusted: CaptureLine}} capture_ratio
 * @property {{total_usd: number, spread_captured_usd: number, hedging_and_closing_usd: number, by_horizon: Record<string, Horizon_>}} decomposition
 * @property {{clients: Record<string, ClientReveal>}} diagnostics
 * @property {RfqLayer[]} rfqs
 * @property {{clients: {client_id: number, category: string, informed_type: boolean}[]}} world
 * @property {RevealBlock} reveal
 */

/**
 * The coaching record, once the session has ended.
 * @typedef {Object} RevealMsg
 * @property {number} v
 * @property {"reveal"} type
 * @property {CoachingRecord} record
 */

/**
 * @typedef {Object} TrackEntry
 * @property {string} session_id
 * @property {number | null} preset
 * @property {number} seed
 * @property {boolean} replay_of_played_seed
 * @property {number | null} ended_early_at_step
 * @property {number} pnl_usd
 * @property {number} risk_adjusted_usd
 * @property {number} disciplined_pnl_usd
 * @property {number} oracle_pnl_usd
 * @property {number} disciplined_risk_adjusted_usd
 * @property {number} oracle_risk_adjusted_usd
 * @property {number} you_minus_disciplined_pnl_usd
 * @property {number} oracle_minus_disciplined_pnl_usd
 * @property {number} you_minus_disciplined_risk_adjusted_usd
 * @property {number} oracle_minus_disciplined_risk_adjusted_usd
 * @property {number | null} capture_ratio_pnl
 * @property {number | null} capture_ratio_risk_adjusted
 */

/**
 * A gap to the disciplined bot over scored sessions: the sum, the mean per
 * session and its standard error (null under two sessions).
 * @typedef {{total_usd: number, mean_usd: number | null, se_usd: number | null}} Pooled
 */

/**
 * One score pooled: your gap and the oracle's, and the pooled capture ratio
 * (null until the oracle's pooled gap reaches the threshold).
 * @typedef {{you_minus_disciplined: Pooled, oracle_minus_disciplined: Pooled, capture_ratio: number | null}} PooledScore
 */

/**
 * The track record, sent after the session: every session, and aggregates over scored ones.
 * @typedef {Object} TrackMsg
 * @property {number} v
 * @property {"track"} type
 * @property {TrackEntry[]} entries
 * @property {{sessions: number, replays_excluded: number, pooled_min_gap_usd: number,
 *   pnl_usd: number | null, risk_adjusted_usd: number | null,
 *   pnl: PooledScore, risk_adjusted: PooledScore}} aggregates
 */

/**
 * @typedef {Object} SummaryMsg
 * @property {number} v
 * @property {"summary"} type
 * @property {string} session_id
 * @property {string} saved_to
 * @property {string} markdown
 */

/** @typedef {EventMsg | StateMsg | DeskMsg | HostMsg | HandbookMsg | ErrorMsg | RevealMsg | SummaryMsg | TrackMsg} ServerMessage */

/** @typedef {{v: number, type: string, [key: string]: any}} Action */

/** @type {(ordinal: number, bid: number | null, ask: number | null) => Action} */
export const quote = (ordinal, bid, ask) => ({ v: V, type: "quote", rfq_ordinal: ordinal, bid, ask });
/** @type {(ordinal: number) => Action} */
export const pass = (ordinal) => ({ v: V, type: "pass", rfq_ordinal: ordinal });
/** @type {(quantity: number) => Action} */
export const hedge = (quantity) => ({ v: V, type: "hedge", quantity });
/** @type {(speed: number) => Action} */
export const setSpeed = (speed) => ({ v: V, type: "set_speed", speed });
/** @type {() => Action} */
export const pause = () => ({ v: V, type: "pause" });
/** @type {() => Action} */
export const resume = () => ({ v: V, type: "resume" });
/** @type {() => Action} */
export const quit = () => ({ v: V, type: "quit" });
/** The server's own message: start the clock the first time. @type {() => Action} */
export const begin = () => ({ v: V, type: "begin" });
