// Draft engine: 8 seats, 3 packs, and the PickNet input tokens each seat has seen so far.
// Layout matches wheelnet's picks_all shards: 3 rounds x 15 pick slots (slots past a set's pack
// size stay empty), up to 15 distinct cards per pack token set, up to 8 "taken by the others" tokens.

import { openPack } from './packs.js';

export const SLOTS = 15, P = 15, K = 8, SEATS = 8, T_MAX = 3 * SLOTS;

function counts(cards) {
  const m = new Map();
  for (const c of cards) m.set(c, (m.get(c) || 0) + 1);
  return m;
}

function remove(m, c) { // one copy, clipped at zero
  const v = m.get(c);
  if (v > 1) m.set(c, v - 1);
  else m.delete(c);
}

function writeTokens(idx, cnt, off, n, m, pad) {
  const keys = [...m.keys()].sort((a, b) => a - b);
  for (let j = 0; j < n; j++) {
    idx[off + j] = j < keys.length ? keys[j] : pad;
    cnt[off + j] = j < keys.length ? m.get(keys[j]) : 0;
  }
}

export class Seat {
  constructor(pad) {
    this.pad = pad;
    this.idx = new Int32Array(T_MAX * P).fill(pad);
    this.cnt = new Int32Array(T_MAX * P);
    this.pick = new Int32Array(T_MAX).fill(pad);
    this.oidx = new Int32Array(T_MAX * K).fill(pad);
    this.ocnt = new Int32Array(T_MAX * K);
    this.seen = [];
    this.pool = [];
  }

  // The pack in hand at slot t (card ids, repeated per copy). From the 9th pick of a round the pack
  // has come back around, so also record what the other seven took from it since we last saw it (if we know).
  show(t, pack) {
    this.seen[t] = pack.slice();
    writeTokens(this.idx, this.cnt, t * P, P, counts(pack), this.pad);
    if (t % SLOTS >= SEATS && this.seen[t - SEATS]) {
      const m = counts(this.seen[t - SEATS]);
      remove(m, this.pick[t - SEATS]);
      for (const c of pack) remove(m, c);
      writeTokens(this.oidx, this.ocnt, t * K, K, m, this.pad);
    }
  }

  take(t, card) {
    this.pick[t] = card;
    this.pool.push(card);
  }

  // Pack tokens at slot t: [{card, copies, j}] for the real (non-padding) tokens.
  tokens(t) {
    const out = [];
    for (let j = 0; j < P; j++) if (this.cnt[t * P + j] > 0) out.push({ card: this.idx[t * P + j], copies: this.cnt[t * P + j], j });
    return out;
  }
}

// Model inputs for these seats at slot t: their drafts so far, the last slot being the pack in hand.
export function feeds(seats, t, ctxs, sid) {
  const B = seats.length, T = t + 1, nCtx = ctxs[0].length;
  const f = {
    B, T, idx: new Int32Array(B * T * P), cnt: new Int32Array(B * T * P), pick: new Int32Array(B * T),
    oidx: new Int32Array(B * T * K), ocnt: new Int32Array(B * T * K), ctx: new Float32Array(B * nCtx),
    sid: new Int32Array(B).fill(sid),
  };
  seats.forEach((s, b) => {
    f.idx.set(s.idx.subarray(0, T * P), b * T * P);
    f.cnt.set(s.cnt.subarray(0, T * P), b * T * P);
    f.pick.set(s.pick.subarray(0, T), b * T);
    f.pick[b * T + t] = s.pad; // the pick being made is not an input
    f.oidx.set(s.oidx.subarray(0, T * K), b * T * K);
    f.ocnt.set(s.ocnt.subarray(0, T * K), b * T * K);
    f.ctx.set(ctxs[b], b * nCtx);
  });
  return f;
}

// Seat b's choices at slot t from the model's [B, 15] logits: [{card, copies, logit, p}], best first.
export function choices(seat, t, logits, b) {
  const toks = seat.tokens(t).map((x) => ({ ...x, logit: logits[b * P + x.j] }));
  const mx = Math.max(...toks.map((x) => x.logit));
  const z = toks.reduce((a, x) => a + Math.exp(x.logit - mx), 0);
  for (const x of toks) x.p = Math.exp(x.logit - mx) / z;
  return toks.sort((a, b2) => b2.logit - a.logit);
}

export function sample(ch, temperature, rng = Math.random) {
  if (temperature <= 0) return ch[0];
  const w = ch.map((x) => Math.exp((x.logit - ch[0].logit) / temperature));
  let r = rng() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < ch.length; i++) if ((r -= w[i]) <= 0) return ch[i];
  return ch[ch.length - 1];
}

export class Draft {
  // set: data/{SET}.json; seat 0 is the human. openers: the 24 packs to open (3 rounds x 8 seats),
  // e.g. a previous draft's, to replay it; by default freshly opened from the set's collation.
  constructor(set, rng = Math.random, openers = null) {
    this.set = set;
    this.n = set.n_picks;
    this.rng = rng;
    this.seats = Array.from({ length: SEATS }, () => new Seat(set.pad));
    if (!openers) openers = Array.from({ length: 3 * SEATS }, () => openPack(set, rng));
    this.openers = openers.map((p) => p.slice()); // hands are cut from these as picks are made
    this.round = 0;
    this.k = 0;
    this.hands = this.openers.slice(0, SEATS);
    this.done = false;
    this.show();
  }

  get t() { return this.round * SLOTS + this.k; }

  show() { this.seats.forEach((s, i) => s.show(this.t, this.hands[i])); }

  // picks[i]: card id seat i takes from its hand. Passes left in packs 1 and 3, right in pack 2.
  apply(picks) {
    const t = this.t;
    this.seats.forEach((s, i) => {
      s.take(t, picks[i]);
      const h = this.hands[i], at = h.indexOf(picks[i]);
      if (at < 0) throw new Error(`seat ${i} picked ${picks[i]}, not in hand`);
      h.splice(at, 1);
    });
    this.k += 1;
    if (this.k === this.n) {
      this.round += 1;
      this.k = 0;
      if (this.round === 3) { this.done = true; return; }
      this.hands = this.openers.slice(this.round * SEATS, (this.round + 1) * SEATS);
    } else {
      const dir = this.round === 1 ? SEATS - 1 : 1;
      this.hands = this.hands.map((_, i) => this.hands[(i - dir + SEATS) % SEATS]);
    }
    this.show();
  }
}

// PickNet's first-pick preference for each of `ids` (distinct card ids): Map id -> score. The cards go
// in as first-pick packs of up to 15, each also holding one fixed anchor card, and each card's logit
// is read relative to the anchor's so scores from different packs compare. Averaged over `reps`
// random splits into packs, since a card's logit shifts a little with its packmates. run(feeds) -> logits.
export async function rateCards(set, ids, ctx, run, reps = 4, rng = Math.random) {
  const [anchor, ...rest] = ids;
  const chunks = [];
  for (let r = 0; r < reps; r++) {
    const order = rest.map((c) => [rng(), c]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    for (let i = 0; i < order.length || i === 0; i += P - 1) chunks.push([anchor, ...order.slice(i, i + P - 1)]);
  }
  const seats = chunks.map((c) => { const s = new Seat(set.pad); s.show(0, c); return s; });
  const logits = await run(feeds(seats, 0, seats.map(() => ctx), set.sid));
  const out = new Map(ids.map((c) => [c, 0]));
  seats.forEach((s, b) => {
    const ch = choices(s, 0, logits, b), a = ch.find((x) => x.card === anchor).logit;
    for (const x of ch) out.set(x.card, out.get(x.card) + (x.logit - a) / reps);
  });
  return out;
}

// How likely each card of a live pack is to come back 8 picks later, by sending the pack round the other seven
// seats many times over. Nobody knows the real table's drafts, so the other drafters are stand-ins: the seats of
// a few simulated tables (PickNet drafting against itself from freshly opened packs, picks sampled like a human),
// kept a few slots ahead of the live draft so there is a drafter with a history at every slot the pack visits.
// Each pass round the table draws a stand-in at random for each seat. A card's odds assume you take PickNet's
// first choice, or its second when the card is the first choice. run(feeds) -> logits.
export class WheelSim {
  constructor(set, run, { tables = 3, rollouts = 48, temperature = 1, ctx = set.ctx.typical, rng = Math.random } = {}) {
    Object.assign(this, { set, run, rollouts, temperature, ctx, rng });
    this.tables = Array.from({ length: tables }, () => new Draft(set, rng));
  }

  get t() { return this.tables[0].t; }

  // Play the simulated tables on to slot t (or the end of the draft). onStep(t) after each slot.
  async advanceTo(t, cancelled = () => false, onStep = null) {
    while (!this.tables[0].done && this.t < t) {
      if (cancelled()) return false;
      const seats = this.tables.flatMap((d) => d.seats), at = this.t;
      const logits = await this.run(feeds(seats, at, seats.map(() => this.ctx), this.set.sid));
      this.tables.forEach((d, i) => d.apply(d.seats.map((s, j) =>
        sample(choices(s, at, logits, i * SEATS + j), this.temperature, this.rng).card)));
      onStep?.(this.t);
    }
    return true;
  }

  // pack: the card ids in your hand at slot t (repeated per copy); mine: your likely picks, best first.
  // -> Map card -> P(at least one copy comes back), or null if cancelled or the pack won't come back this round.
  async odds(pack, t, mine, cancelled = () => false, onStep = null) {
    const k = t % SLOTS;
    if (k + SEATS >= this.set.n_picks) return null;
    if (!(await this.advanceTo(t + SEATS - 1, cancelled, onStep))) return null;
    const R = this.rollouts, takes = Array.from({ length: R }, (_, r) => mine[r % Math.min(2, mine.length)]);
    const hands = takes.map((c) => { const h = pack.slice(); h.splice(h.indexOf(c), 1); return h; });
    const pool = this.tables.flatMap((d) => d.seats);
    for (let j = 1; j < SEATS; j++) {
      if (cancelled()) return null;
      const at = t + j;
      const seats = hands.map((h) => {
        const src = pool[Math.floor(this.rng() * pool.length)];
        const s = Object.assign(new Seat(this.set.pad), {
          idx: src.idx.slice(), cnt: src.cnt.slice(), pick: src.pick.slice(), oidx: src.oidx.slice(),
          ocnt: src.ocnt.slice(), seen: src.seen.slice(),
        });
        // on its second lap the pack is one this seat saw before; what it held then is close to what you saw
        if (at % SLOTS >= SEATS) s.seen[at - SEATS] = pack;
        s.show(at, h);
        return s;
      });
      const logits = await this.run(feeds(seats, at, seats.map(() => this.ctx), this.set.sid));
      seats.forEach((s, b) => {
        const c = sample(choices(s, at, logits, b), this.temperature, this.rng).card, h = hands[b];
        h.splice(h.indexOf(c), 1);
      });
      onStep?.(at);
    }
    const out = new Map();
    for (const c of new Set(pack)) {
      let n = 0, back = 0;
      hands.forEach((h, r) => { if (takes[r] !== c) { n++; if (h.includes(c)) back++; } });
      out.set(c, n ? back / n : null);
    }
    return out;
  }
}
