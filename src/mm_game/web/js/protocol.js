// Messages the front end sends. Must speak the server's protocol version:
// tests/test_server.py checks V against mm_game.protocol.PROTOCOL_VERSION.
export const V = 2;

export const quote = (ordinal, bid, ask) => ({ v: V, type: "quote", rfq_ordinal: ordinal, bid, ask });
export const pass = (ordinal) => ({ v: V, type: "pass", rfq_ordinal: ordinal });
export const hedge = (quantity) => ({ v: V, type: "hedge", quantity });
export const setSpeed = (speed) => ({ v: V, type: "set_speed", speed });
export const pause = () => ({ v: V, type: "pause" });
export const resume = () => ({ v: V, type: "resume" });
export const quit = () => ({ v: V, type: "quit" });
// The server's own message: start the clock the first time.
export const begin = () => ({ v: V, type: "begin" });
