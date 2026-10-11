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
 * The coaching record, once the session has ended.
 * @typedef {Object} RevealMsg
 * @property {number} v
 * @property {"reveal"} type
 * @property {Record<string, any>} record
 */

/**
 * @typedef {Object} SummaryMsg
 * @property {number} v
 * @property {"summary"} type
 * @property {string} session_id
 * @property {string} saved_to
 * @property {string} markdown
 */

/** @typedef {EventMsg | StateMsg | DeskMsg | HostMsg | HandbookMsg | ErrorMsg | RevealMsg | SummaryMsg} ServerMessage */

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
