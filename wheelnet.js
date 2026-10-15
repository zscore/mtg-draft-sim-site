// WheelNet (../wheelnet, exported by tools/export_wheelnet.py): the chance each card you pass comes back 8 picks
// later, read from your whole draft so far. Its inputs come from a draft.js Seat, re-indexed to the set's local
// card order (data/wheel/{SET}.json "cards": PickNet id -> local index) and laid out by its 3 x n_picks positions.

import { SLOTS } from './draft.js';

const SEATS = 8;

// Distinct cards (local indices, ascending) and copies from a Map local -> copies, into k slots padded with pad.
function writeBag(idx, cnt, off, k, m, pad) {
  const keys = [...m.keys()].filter((c) => m.get(c) > 0).sort((a, b) => a - b);
  for (let j = 0; j < k; j++) {
    idx[off + j] = j < keys.length ? keys[j] : pad;
    cnt[off + j] = j < keys.length ? m.get(keys[j]) : 0;
  }
}

function localCounts(meta, ids) {
  const m = new Map();
  for (const id of ids) {
    const c = meta.cards[id];
    if (c !== undefined) m.set(c, (m.get(c) || 0) + 1);
  }
  return m;
}

const minus = (m, ids, meta) => { for (const id of ids) { const c = meta.cards[id]; if (m.get(c) > 0) m.set(c, m.get(c) - 1); } return m; };

// Model inputs for one batch: the seat's draft up to slot t (later positions are padding), where batch row b takes
// takes[b] (a PickNet card id) at slot t. ctx: the draft context, meta.ctx.typical or .strong.
export function wheelFeeds(meta, seat, t, takes, ctx) {
  const n = meta.n_picks, T = 3 * n, M = meta.M, W = meta.W, pad = meta.pad, B = takes.length;
  const one = {
    pack: new Int32Array(T * n), pack_n: new Int32Array(T * n), oth: new Int32Array(T * M), oth_n: new Int32Array(T * M),
    pick: new Int32Array(T).fill(pad), idx: new Int32Array(3 * W * M), copies: new Int32Array(3 * W * M),
  };
  one.pack.fill(pad);
  one.oth.fill(pad);
  one.idx.fill(pad);
  for (let r = 0; r < 3; r++) {
    for (let p = 0; p < n; p++) {
      const s = r * SLOTS + p, q = r * n + p;
      if (s > t || !seat.seen[s]) continue;
      writeBag(one.pack, one.pack_n, q * n, n, localCounts(meta, seat.seen[s]), pad);
      const took = seat.pick[s];
      if (s < t && took !== seat.pad && meta.cards[took] !== undefined) one.pick[q] = meta.cards[took];
      if (p >= SEATS && seat.seen[s - SEATS]) { // taken by the other seven since you saw this pack
        const gone = minus(minus(localCounts(meta, seat.seen[s - SEATS]), [seat.pick[s - SEATS]], meta), seat.seen[s], meta);
        writeBag(one.oth, one.oth_n, q * M, M, gone, pad);
      }
      if (p < W && s < t) writeBag(one.idx, one.copies, (r * W + p) * M, M, minus(localCounts(meta, seat.seen[s]), [took], meta), pad);
    }
  }
  const f = { B };
  for (const [k, a] of Object.entries(one)) {
    f[k] = new Int32Array(B * a.length);
    for (let b = 0; b < B; b++) f[k].set(a, b * a.length);
  }
  f.mask = new Int32Array(f.copies.length);
  f.ctx = new Float32Array(B * meta.n_ctx);
  const tr = Math.floor(t / SLOTS), tp = t % SLOTS, tq = tr * n + tp;
  takes.forEach((c, b) => {
    if (meta.cards[c] !== undefined) f.pick[b * T + tq] = meta.cards[c];
    if (tp < W) writeBag(f.idx, f.copies, b * 3 * W * M + (tr * W + tp) * M, M, minus(localCounts(meta, seat.seen[t]), [c], meta), pad);
    f.ctx.set(ctx, b * meta.n_ctx);
  });
  for (let i = 0; i < f.copies.length; i++) f.mask[i] = f.copies[i] > 0 ? 1 : 0;
  return f;
}

// The model's input tensors for the worker: [{name, type, data, dims}].
export function wheelTensors(meta, f) {
  const n = meta.n_picks, T = 3 * n, M = meta.M, W = meta.W;
  const dims = { pack: [T, n], pack_n: [T, n], oth: [T, M], oth_n: [T, M], pick: [T], idx: [3, W, M], copies: [3, W, M], mask: [3, W, M], ctx: [meta.n_ctx] };
  return Object.entries(dims).map(([name, d]) => ({ name, type: name === 'ctx' ? 'float32' : 'int32', data: f[name], dims: [f.B, ...d] }));
}

// P(wheel) for the cards passed at slot t in batch row b: Map PickNet card id -> probability.
export function wheelOdds(meta, f, p, t, b = 0) {
  const W = meta.W, M = meta.M, r = Math.floor(t / SLOTS), k = t % SLOTS, out = new Map();
  if (k >= W) return out;
  const local = meta.byLocal ??= Object.fromEntries(Object.entries(meta.cards).map(([id, c]) => [c, +id]));
  const off = b * 3 * W * M + (r * W + k) * M;
  for (let j = 0; j < M; j++) if (f.mask[off + j]) out.set(local[f.idx[off + j]], p[off + j]);
  return out;
}
