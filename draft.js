// Draft engine: 8 seats, 3 packs, and the PickNet input tokens each seat has seen so far.
// Layout matches mtg-open-lane's picks_all shards: 3 rounds x 15 pick slots (slots past a set's pack
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
  // has come back around, so also record what the other seven took from it since we last saw it.
  show(t, pack) {
    this.seen[t] = pack.slice();
    writeTokens(this.idx, this.cnt, t * P, P, counts(pack), this.pad);
    if (t % SLOTS >= SEATS) {
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
