// Deck model and deck builder. The model is mtg-open-lane's game-level deck model (tools/export_deck_model.py):
//   logit P(win) = base + w.x + sum_{i<j} x_i x_j <v_i, v_j> + F(x).wF + aff(x) * wAff
// with x the deck's card counts (basics included), v rank-4 pair vectors between nonland cards, F the deck
// features of openlane.cardinfo.deck_features (land count, curve, colours, castability) and aff the deck's mean
// pairwise pick-model affinity. One model per event: Premier Draft games for draft pools, Sealed for sealed.
//
// Every term is a function of a few linear sums over the deck's cards (plus two sums of squares), so a deck is held
// as that aggregate and a swap is scored by adding one card's row and subtracting another's.
//
// The builder searches for the deck the model likes best, and a search finds a model's blind spots: a set of cards
// whose pair vectors line up far beyond any real deck, or one main colour with three splashes. So the pair term,
// each deck feature and the affinity are clamped to the range real training decks span (0.5% to 99.5%), and a deck's
// colours must each have the sources they need (manaBase). The search is over spells and utility lands only: the
// rest of the lands are the rule-based mana base the PickNet build uses too, so the builder can't farm the model's
// land terms with off-colour duals.

export const BASICS = { W: 'Plains', U: 'Island', B: 'Swamp', R: 'Mountain', G: 'Forest' };
const COLORS = 'WUBRG', SOURCE_NEED = [7, 10, 12]; // coloured sources a 40-card deck wants for 1/2/3 pips
export const DECK_SIZE = 40, LANDS = [16, 18];

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export class DeckModel {
  // json: data/deck/{SET}.json; event: 'draft' | 'sealed'; set: data/{SET}.json (card ids <-> names).
  constructor(json, event, set) {
    const m = json[event];
    this.event = event;
    this.set = set;
    this.base = m.base;
    this.typical = sigmoid(m.base + m.deck_logit_mean);
    this.k = m.v[0].length;
    this.f = m.features;
    this.ranges = m.ranges;
    this.aff = json.affinity && m.aff ? m.aff : null;
    this.dA = this.aff ? json.affinity.a[0].length : 0;
    // row layout of a card's contribution to the aggregate
    const L = {}, k = this.k, dA = this.dA;
    let o = 0;
    for (const [name, n] of [['w', 1], ['v', k], ['land', 1], ['spell', 1], ['curve', 6], ['mv', 1], ['creature', 1],
      ['removal', 1], ['uses', 5], ['src', 5], ['pips', 5], ['need', 15], ['a', dA], ['e', dA]]) { L[name] = o; o += n; }
    this.L = L;
    this.D = o;
    const byName = new Map(json.names.map((n, i) => [n, i]));
    const ws = m.w.filter((_, i) => !json.card.is_land[i] && m.games_with[i] >= 2000).sort((a, b) => a - b);
    this.unknownW = ws[Math.floor(ws.length * 0.05)] ?? 0; // a card the games never saw: a 5th-percentile playable
    this.rows = new Map(); // site card id -> {row, vv, ae, land, basic}
    this.basicIds = {};
    for (const [id, c] of Object.entries(set.cards)) {
      const i = byName.get(c.name);
      this.rows.set(+id, this.row(json, m, i, c));
      for (const [col, name] of Object.entries(BASICS)) if (c.name === name) this.basicIds[col] = +id;
    }
  }

  row(json, m, i, c) {
    const L = this.L, r = new Float64Array(this.D);
    if (i === undefined) { // not in the game data: a weak spell of its colours with no pair or affinity terms
      r[L.w] = this.unknownW;
      const land = /Land/.test(c.type) && !/Creature/.test(c.type);
      r[land ? L.land : L.spell] = 1;
      return { row: r, vv: 0, ae: 0, land, colors: c.colors };
    }
    const card = json.card, land = card.is_land[i];
    r[L.w] = m.w[i];
    for (let d = 0; d < this.k; d++) r[L.v + d] = m.v[i][d];
    r[land ? L.land : L.spell] = 1;
    if (!land) {
      r[L.curve + Math.min(6, Math.max(1, card.mv[i])) - 1] = 1;
      r[L.mv] = card.mv[i];
    }
    r[L.creature] = card.is_creature[i] ? 1 : 0;
    r[L.removal] = card.removal[i] ? 1 : 0;
    for (let c5 = 0; c5 < 5; c5++) {
      const p = card.pips[i][c5];
      r[L.uses + c5] = p > 0 ? 1 : 0;
      r[L.src + c5] = (land && card.makes[i][c5] ? 1 : 0) + (card.nonland_source[i] ? 0.5 : 0);
      r[L.pips + c5] = p;
      // need: the sources wanted for the card's pips of this colour (7 / 10 / 12), as a count per need level
      const n = Math.min(3, Math.ceil(p - 1e-9));
      if (n > 0) r[L.need + 3 * c5 + n - 1] = 1;
    }
    let vv = 0, ae = 0;
    for (let d = 0; d < this.k; d++) vv += m.v[i][d] ** 2;
    if (this.aff && !land) {
      const a = json.affinity.a[i], e = json.affinity.e[i];
      for (let d = 0; d < this.dA; d++) { r[L.a + d] = a[d]; r[L.e + d] = e[d]; ae += a[d] * e[d]; }
    }
    return { row: r, vv, ae, land, colors: c.colors };
  }

  // Aggregate of a deck given as Map(card id -> copies).
  aggregate(counts) {
    const g = { s: new Float64Array(this.D), vv: 0, ae: 0 };
    for (const [id, n] of counts) this.add(g, id, n);
    return g;
  }

  // Add n copies (negative: remove) of a card to aggregate g, in place.
  add(g, id, n, have = null) {
    const r = this.rows.get(id);
    for (let d = 0; d < this.D; d++) g.s[d] += n * r.row[d];
    // sums of squared counts: (h + n)^2 - h^2 for a card held h times before
    const h = have ?? 0;
    const dsq = (h + n) ** 2 - h ** 2;
    g.vv += dsq * r.vv;
    g.ae += dsq * r.ae;
  }

  // The deck's logit terms (everything but `base`) from its aggregate.
  logit(g) {
    const { L, f, k } = this, s = g.s, F = this._F || (this._F = new Float64Array(f.w.length));
    let z = s[L.w];
    let fm = 0;
    for (let d = 0; d < k; d++) fm += s[L.v + d] ** 2;
    const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, v));
    z += clamp(0.5 * (fm - g.vv), this.ranges.pair);
    const nLand = s[L.land], nSpell = s[L.spell], c = f.centres;
    let curveSq = 0;
    for (let b = 0; b < 6; b++) curveSq += (s[L.curve + b] - c.curve[b]) ** 2;
    const mvMean = s[L.mv] / Math.max(nSpell, 1);
    let main = 0, splash = 0, gap = 0, worst = 0, pipTot = 0, srcTot = 0, unused = 0;
    for (let col = 0; col < 5; col++) {
      const uses = s[L.uses + col], src = s[L.src + col];
      if (uses >= 4 - 1e-9) main++;
      else if (uses > 1e-9) splash++;
      for (let p = 0; p < 3; p++) {
        const n = s[L.need + 3 * col + p];
        if (n > 1e-9) {
          const short = Math.max(0, SOURCE_NEED[p] - src);
          gap += n * short;
          if (short > worst) worst = short;
        }
      }
      pipTot += s[L.pips + col];
      srcTot += src;
      if (uses <= 1e-9) unused += src;
    }
    let mismatch = 0;
    for (let col = 0; col < 5; col++) {
      mismatch += Math.abs(s[L.pips + col] / Math.max(pipTot, 1e-6) - s[L.src + col] / Math.max(srcTot, 1e-6));
    }
    const nl = Math.round(nLand);
    F[0] = nl <= 15; F[1] = nl === 16; F[2] = nl === 18; F[3] = nl >= 19;
    F[4] = main <= 1; F[5] = main >= 3; F[6] = splash > 0;
    F[7] = curveSq; F[8] = mvMean; F[9] = (mvMean - c.mv_mean) ** 2;
    F[10] = (s[L.creature] - c.creatures) ** 2; F[11] = (s[L.removal] - c.removal) ** 2;
    F[12] = Math.max(0, 4 - s[L.curve + 1]);
    F[13] = gap / Math.max(nSpell, 1); F[14] = worst; F[15] = mismatch; F[16] = unused;
    // a feature constant over training decks has sd 1e-6, which the export rounds to 0
    for (let j = 0; j < F.length; j++) z += ((clamp(F[j], this.ranges.features[j]) - f.mu[j]) / Math.max(f.sd[j], 1e-6)) * f.w[j];
    if (this.aff) {
      let tot = -g.ae;
      for (let d = 0; d < this.dA; d++) tot += s[L.a + d] * s[L.e + d];
      const raw = clamp(tot / Math.max(nSpell * (nSpell - 1), 1), this.ranges.aff);
      z += ((raw - this.aff.mu) / this.aff.sd) * this.aff.w;
    }
    return z;
  }

  // The deck's logit unclamped: what the model itself says, for checking against PyTorch.
  rawLogit(g) {
    const r = this.ranges;
    this.ranges = { pair: [-Infinity, Infinity], features: r.features.map(() => [-Infinity, Infinity]), aff: [-Infinity, Infinity] };
    try { return this.logit(g); } finally { this.ranges = r; }
  }

  winRate(logit) { return sigmoid(this.base + logit); }

  score(counts) { return this.logit(this.aggregate(counts)); }
}

// ---------------------------------------------------------------- mana base

// The lands for a deck's spells, shared by both builders. It reads the set data's mana facts (tools/mana_facts.py:
// pips, the colours a land makes, enters tapped, ramp, landcycling).
//   - Land count from the curve: 17 at a mean mana value of 2.9 (landcyclers left out), one more or fewer per 0.4,
//     half a land fewer per mana rock, dork or landcycler, held to 16-18.
//   - Nonbasics from the pool only when they help: a land making two of the deck's colours (main or splash), a
//     basic-fetcher only in three colours or more, a choose-a-colour land only with a splash, at most three that
//     enter tapped. A dual making one of the deck's colours and one it doesn't play is a tapped basic, and stays in
//     the sideboard.
//   - Utility lands (utilityLands: a nonbasic making one of the deck's colours or only colourless mana, Mutavault
//     or Hidden Necropolis) are the caller's choice, at most two, each in place of a basic: the deck model's build
//     plays one when the model rates the deck higher with it, the PickNet build when PickNet rates it as highly as
//     the deck's weakest spell.
//   - Basics to fill: a splash colour (one used by at most three cards) gets 3 sources (4 for a double-pip card),
//     the main colours the rest by their pips but at least 7 each, each basic going to the colour furthest below
//     its share. Fetch lands count as a source of every colour.
//   - Any number of colours up to four, as long as each gets the sources it needs: two colours always do, a third
//     main colour or a fourth colour only with enough dual lands.

export const isLand = (c) => { const t = c.type.split(' // ')[0]; return /Land/.test(t) && !/Creature/.test(t); };
const cardPips = (c) => c.pips ?? Object.fromEntries([...c.colors].map((k) => [k, 1]));
const MAX_TAPPED = 3, MAX_UTILITY = 2;
// Sources a colour needs (Frank Karsten's numbers for 40 cards, rounded down): 7 for a main colour (four cards or
// more), 3 for a splash (4 with a double-pip card). A deck short of them isn't one the builders make.
const MIN_MAIN = 7, MIN_SPLASH = 3, MAX_COLORS = 4;

// What the mana base needs from a deck's spells (Map id -> copies).
function spellStats(set, spells) {
  const zero = () => ({ W: 0, U: 0, B: 0, R: 0, G: 0 });
  const s = { mv: 0, nMv: 0, cheap: 0, pips: zero(), uses: zero(), heavy: zero() };
  for (const [id, n] of spells) {
    const c = set.cards[id];
    if (c.ramp || c.cycler) s.cheap += n;
    if (!c.cycler) { s.mv += n * c.cmc; s.nMv += n; }
    for (const [k, p] of Object.entries(cardPips(c))) {
      s.pips[k] += n * p;
      s.uses[k] += n;
      if (p >= 2) s.heavy[k] += n;
    }
  }
  return s;
}

const landsFor = (s) => Math.min(LANDS[1], Math.max(LANDS[0],
  Math.round(17 + 2.5 * (s.mv / Math.max(s.nMv, 1) - 2.9) - 0.5 * s.cheap)));

// The land count for a deck's spells (Map id -> copies).
export const landCount = (set, spells) => landsFor(spellStats(set, spells));

// The deck's colours: used (by its spells; with none, the pool's most common colour), mains and splash.
function deckColors(set, s, pool) {
  let used = [...COLORS].filter((k) => s.uses[k] > 0);
  if (!used.length) {
    const n = Object.fromEntries([...COLORS].map((k) => [k, 0]));
    for (const id of pool) for (const k of set.cards[id].colors) n[k] += 1;
    used = [[...COLORS].sort((a, b) => n[b] - n[a])[0]];
  }
  const main = used.filter((k) => s.uses[k] >= 4), mains = main.length ? main : used;
  return { used, mains, splash: used.filter((k) => !mains.includes(k)) };
}

const nonbasicLand = (c) => isLand(c) && !/Basic/.test(c.type);
// The colours a fixing land gives the deck, [] for a land that isn't one.
function fixes(c, { used, splash }) {
  if (!c.makes) return [];
  const on = c.makes === '*' ? (used.length >= 3 ? used : []) : c.makes === '?' ? splash.slice(0, 1)
    : used.filter((k) => c.makes.includes(k));
  return on.length >= 2 || (/[*?]/.test(c.makes) && on.length) ? on : [];
}
// A utility land for the deck: not a fixer, making only colourless mana or only the deck's colours (an off-colour
// dual is a tapped basic, not a utility land).
const utility = (c, cols) => !fixes(c, cols).length
  && (!c.makes || (/^[WUBRG]+$/.test(c.makes) && [...c.makes].every((k) => cols.used.includes(k))));

// The pool's utility lands for a deck's spells (Map id -> copies): distinct card ids.
export function utilityLands(set, spells, pool) {
  const cols = deckColors(set, spellStats(set, spells), pool);
  return [...new Set(pool)].filter((id) => nonbasicLand(set.cards[id]) && utility(set.cards[id], cols));
}

// The mana base for a deck's spells (Map id -> copies) from a pool (card ids, repeats for copies): {lands: the pool's
// nonbasic land ids it plays, basics: {W, U, B, R, G} counts, ok: every colour has the sources it needs (MIN_MAIN,
// MIN_SPLASH) in at most four colours}, nLands lands in all (default: landCount's). `util`: utility lands to play (the first MAX_UTILITY that are utility lands
// for these spells).
export function manaBase(set, spells, pool, nLands, util = []) {
  const s = spellStats(set, spells);
  nLands ??= landsFor(s);
  const cols = deckColors(set, s, pool), { used, mains, splash } = cols;
  const options = [];
  for (const id of pool) {
    const c = set.cards[id];
    if (!nonbasicLand(c)) continue;
    const on = fixes(c, cols);
    if (on.length) options.push({ id, on, tapped: !!c.tapped });
  }
  const splashes = (o) => o.on.some((k) => splash.includes(k));
  options.sort((a, b) => splashes(b) - splashes(a) || b.on.length - a.on.length || a.tapped - b.tapped);
  const lands = [], src = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  let tapped = 0;
  for (const id of util.filter((id) => utility(set.cards[id], cols)).slice(0, MAX_UTILITY)) {
    const c = set.cards[id];
    lands.push(id);
    tapped += !!c.tapped;
    for (const k of used) if (c.makes?.includes(k)) src[k] += 1;
  }
  for (const o of options) {
    if (lands.length >= nLands || (o.tapped && tapped >= MAX_TAPPED)) continue;
    lands.push(o.id);
    tapped += o.tapped;
    for (const k of o.on) src[k] += 1;
  }
  const need = {}, target = {};
  let spare = nLands;
  for (const k of splash) { need[k] = target[k] = MIN_SPLASH + (s.heavy[k] ? 1 : 0); spare -= target[k]; }
  const tot = mains.reduce((a, k) => a + s.pips[k], 0);
  for (const k of mains) {
    need[k] = MIN_MAIN;
    target[k] = Math.max(MIN_MAIN, tot ? (spare * s.pips[k]) / tot : spare / mains.length);
  }
  const basics = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  for (let i = lands.length; i < nLands; i++) {
    const gap = (k) => target[k] - src[k] - basics[k];
    basics[used.reduce((a, k) => (gap(k) > gap(a) + 1e-9 || (Math.abs(gap(k) - gap(a)) <= 1e-9 && s.pips[k] > s.pips[a]) ? k : a))] += 1;
  }
  const ok = used.length <= MAX_COLORS && used.every((k) => src[k] + basics[k] >= need[k]);
  return { lands, basics, ok };
}

// ---------------------------------------------------------------- builder

// The builder searches over spells and utility lands; the rest of a deck's lands are always manaBase's. A state is
// {spells: Map id -> copies, n: land count, util: utility land ids}.

// The pool's spells with how many copies are available, and its nonbasic lands.
function candidates(model, pool) {
  const avail = new Map(), lands = [];
  for (const id of pool) {
    if (isLand(model.set.cards[id])) lands.push(id);
    else avail.set(id, (avail.get(id) || 0) + 1);
  }
  return { avail, lands };
}

// The whole deck (Map id -> copies) for a state's spells, utility lands and manaBase's lands, with `ok` attached:
// whether its mana base gives every colour the sources it needs, which is all the builder asks of a deck.
function deckOf(model, st, lands) {
  const base = manaBase(model.set, st.spells, lands, st.n, st.util), x = new Map(st.spells);
  for (const id of base.lands) x.set(id, (x.get(id) || 0) + 1);
  for (const [k, v] of Object.entries(base.basics)) if (v) x.set(model.basicIds[k], (x.get(model.basicIds[k]) || 0) + v);
  x.ok = base.ok;
  return x;
}

// Score deck y against deck x (aggregate g): fills `tmp` with y's aggregate by adding the cards that differ, and
// returns y's logit, or null if y's mana isn't ok.
function rescore(model, g, x, y, tmp) {
  tmp.s.set(g.s);
  tmp.vv = g.vv;
  tmp.ae = g.ae;
  for (const [id, n] of y) { const h = x.get(id) || 0; if (n !== h) model.add(tmp, id, n - h, h); }
  for (const [id, h] of x) if (!y.has(id)) model.add(tmp, id, -h, h);
  return y.ok ? model.logit(tmp) : null;
}

const swapped = (spells, out, inn) => {
  const sp = new Map(spells);
  if (out !== null) { if (sp.get(out) > 1) sp.set(out, sp.get(out) - 1); else sp.delete(out); }
  if (inn !== null) sp.set(inn, (sp.get(inn) || 0) + 1);
  return sp;
};

// A colour target for the search: want = {main: 'WU' (the main colours), splash: '' (none), 'B' (that colour) or '*'
// (any one colour)}. misfit says how far a deck's spells are from it: 0 when they fit, one more per card a main colour
// is short of four and one for a splash asked for but missing; null when they can't fit (a colour the target doesn't
// allow, two splash colours, or a splash of more than three cards). The search climbs value = logit - PENALTY * misfit,
// so it walks into the target from a start outside it. want null: anything goes.
const PENALTY = 10;
function misfit(set, spells, want) {
  if (!want) return 0;
  const uses = {};
  for (const [id, n] of spells) for (const k of Object.keys(cardPips(set.cards[id]))) uses[k] = (uses[k] || 0) + n;
  const extra = Object.keys(uses).filter((k) => !want.main.includes(k));
  if (extra.length > (want.splash ? 1 : 0)) return null;
  if (extra.length && ((want.splash !== '*' && extra[0] !== want.splash) || uses[extra[0]] > 3)) return null;
  let v = 0;
  for (const k of want.main) v += Math.max(0, 4 - (uses[k] || 0));
  if (want.splash && !extra.length) v += 1;
  return v;
}

// The colours a target lets a spell use, or null for any.
const allowed = (want) => (!want || want.splash === '*' ? null : want.main + want.splash);

// The states one move from st: each spell swap (a deck spell out, a pool spell in), and each utility land in (in
// place of a basic) or out.
function* moves(model, st, avail, lands, want) {
  const uses = {}, colsOf = (id) => Object.keys(cardPips(model.set.cards[id])), ok = allowed(want);
  for (const [id, n] of st.spells) for (const k of colsOf(id)) uses[k] = (uses[k] || 0) + n;
  for (const out of st.spells.keys()) {
    const left = new Set(Object.keys(uses).filter((k) => uses[k] > (colsOf(out).includes(k) ? 1 : 0)));
    for (const [inn, a] of avail) {
      if (inn === out || (st.spells.get(inn) || 0) >= a) continue;
      if (ok && colsOf(inn).some((k) => !ok.includes(k))) continue;
      // a swap to five colours can't be ok: skip it before building its mana base
      if (new Set([...left, ...colsOf(inn)]).size > MAX_COLORS) continue;
      yield { ...st, spells: swapped(st.spells, out, inn) };
    }
  }
  for (const u of st.util) yield { ...st, util: st.util.filter((v) => v !== u) };
  if (st.util.length < MAX_UTILITY) {
    for (const u of utilityLands(model.set, st.spells, lands)) if (!st.util.includes(u)) yield { ...st, util: [...st.util, u] };
  }
}

// A state's value to the search from its logit z (null if z is): see misfit.
function valueOf(model, st, z, want) {
  const m = z === null ? null : misfit(model.set, st.spells, want);
  return m === null ? null : z - PENALTY * m;
}

// Best-improvement hill climbing over moves until none helps. st is updated in place; returns its value.
function climb(model, st, avail, lands, want) {
  // a start whose mana isn't ok (a three-colour start of basics) is worth nothing: any ok deck beats it
  let x = deckOf(model, st, lands), g = model.aggregate(x);
  let v = (x.ok ? valueOf(model, st, model.logit(g), want) : null) ?? -Infinity;
  const tmp = { s: new Float64Array(model.D), vv: 0, ae: 0 };
  for (let iter = 0; iter < 200; iter++) {
    let best = null;
    for (const next of moves(model, st, avail, lands, want)) {
      const vi = valueOf(model, next, rescore(model, g, x, deckOf(model, next, lands), tmp), want);
      if (vi !== null && vi > v + 1e-9 && (!best || vi > best.v)) best = { next, v: vi };
    }
    if (!best) break;
    Object.assign(st, best.next);
    x = deckOf(model, st, lands);
    g = model.aggregate(x);
    v = best.v;
  }
  return v;
}

// Climb, then bring the land count to what the curve asks (landCount) by the best spell added or dropped, and climb
// again, a few rounds at most. A curve on the edge (23 spells want 18 lands, the best 22 of them 17) settles on the
// more lands. Returns the value.
function settle(model, st, avail, lands, want) {
  let v = climb(model, st, avail, lands, want);
  const seen = new Set([st.n]), ok = allowed(want);
  for (let round = 0; round < 4; round++) {
    const target = landCount(model.set, st.spells);
    if (target === st.n) break;
    const n = st.n + Math.sign(target - st.n), back = seen.has(n);
    if (back && n < st.n) break;
    seen.add(n);
    const x = deckOf(model, st, lands), g = model.aggregate(x);
    const tmp = { s: new Float64Array(model.D), vv: 0, ae: 0 };
    const fits = (id) => !ok || Object.keys(cardPips(model.set.cards[id])).every((k) => ok.includes(k));
    const tries = n > st.n ? [...st.spells.keys()].map((out) => swapped(st.spells, out, null))
      : [...avail].filter(([id, a]) => (st.spells.get(id) || 0) < a && fits(id)).map(([id]) => swapped(st.spells, null, id));
    let best = null;
    for (const spells of tries) {
      const next = { ...st, spells, n };
      const vi = valueOf(model, next, rescore(model, g, x, deckOf(model, next, lands), tmp), want);
      if (vi !== null && (!best || vi > best.v)) best = { next, v: vi };
    }
    if (!best) break;
    Object.assign(st, best.next);
    if (back) return best.v;
    v = climb(model, st, avail, lands, want);
  }
  return v;
}

// A starting deck in colours `cols`: its best 23 spells there by power and 17 lands (more if it has fewer spells).
// null when the pool has fewer than `min` spells in those colours (colourless ones count for every colour).
function seed(model, avail, cols, min = DECK_SIZE - LANDS[1]) {
  const L = model.L, spells = [];
  for (const [id, a] of avail) {
    const r = model.rows.get(id);
    if (![...r.colors].every((c) => cols.includes(c))) continue;
    for (let n = 0; n < a; n++) spells.push({ id, w: r.row[L.w] - n * 1e-6 });
  }
  if (spells.length < min) return null;
  spells.sort((a, b) => b.w - a.w);
  const x = new Map();
  const top = spells.slice(0, DECK_SIZE - 17);
  for (const s of top) x.set(s.id, (x.get(s.id) || 0) + 1);
  return { spells: x, n: DECK_SIZE - top.length, util: [] };
}

// Search for the pool's best 40-card deck under the model: a climb from each one- and two-colour start (or from the
// target's main colours), then a few random kicks (three swaps) and re-climbs from the best deck found. Returns
// {counts, logit, winRate}; with a colour target (want, see misfit), null when the pool has no deck that fits it.
export function buildDeck(model, pool, { kicks = 8, rng = Math.random, want = null } = {}) {
  const { avail, lands } = candidates(model, pool);
  const starts = [];
  if (want) starts.push(want.main);
  else {
    for (let i = 0; i < 5; i++) {
      starts.push(COLORS[i]);
      for (let j = i + 1; j < 5; j++) starts.push(COLORS[i] + COLORS[j]);
    }
  }
  const done = (st) => {
    const counts = deckOf(model, st, lands), logit = model.logit(model.aggregate(counts));
    return { counts, logit, winRate: model.winRate(logit) };
  };
  let best = null;
  for (const cols of starts) {
    if (!cols.split('').every((c) => model.basicIds[c] !== undefined)) continue;
    const st = seed(model, avail, cols);
    // a target's start may be off (three colours on basics alone) and climb into shape
    if (!st || (!want && !deckOf(model, st, lands).ok)) continue;
    const v = settle(model, st, avail, lands, want);
    if (v > -Infinity && (!best || v > best.v)) best = { st, v };
  }
  if (!best) { // no legal start in this pool: none for a target, or for the open search its best spells, unsearched
    return want ? null : done(seed(model, avail, COLORS, 0));
  }
  const ids = [...avail.keys()];
  for (let t = 0; t < kicks; t++) {
    const st = { ...best.st, spells: new Map(best.st.spells) };
    for (let s = 0; s < 3; s++) { // three random legal swaps, better or not
      const outs = [...st.spells.keys()], out = outs[Math.floor(rng() * outs.length)];
      const inn = ids[Math.floor(rng() * ids.length)];
      if (inn === out || (st.spells.get(inn) || 0) >= avail.get(inn)) continue;
      const next = { ...st, spells: swapped(st.spells, out, inn) };
      if (deckOf(model, next, lands).ok && misfit(model.set, next.spells, want) !== null) Object.assign(st, next);
    }
    const v = settle(model, st, avail, lands, want);
    if (v > best.v + 1e-9) best = { st, v };
  }
  if (misfit(model.set, best.st.spells, want) !== 0 || !deckOf(model, best.st, lands).ok) return null;
  return done(best.st);
}

// The best deck for each colour combination: every pair of main colours with no splash and with its best splash,
// then every three main colours (which only make it with enough fixing). Yields {want, deck} as each is built, deck
// null for a combination the pool can't make. rng(key) gives each search its random numbers.
export function* deckPerColors(model, pool, rng) {
  const combos = [];
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) combos.push(COLORS[i] + COLORS[j]);
  const wants = combos.flatMap((main) => [{ main, splash: '' }, { main, splash: '*' }]);
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) for (let k = j + 1; k < 5; k++) {
    wants.push({ main: COLORS[i] + COLORS[j] + COLORS[k], splash: '' });
  }
  for (const want of wants) {
    const key = want.main + (want.splash ? `+${want.splash}` : '');
    yield { want, deck: buildDeck(model, pool, { kicks: 3, rng: rng(key), want }) };
  }
}

// What each card is worth to the deck, in win-rate points. A deck spell: how much the deck loses if it is swapped
// for the best replacement from the sideboard. A sideboard spell: how much the deck gains (usually negative) by
// swapping it in for the deck spell it best replaces. A utility land: what the deck loses by playing a basic in its
// place, or gains by playing it for a basic. Each move rebuilds the other lands as the builder does (keeping the
// deck's land count); other lands get no value, and a card with no legal move gets null. Returns Map(id -> points).
export function cardValues(model, deck, pool) {
  const { avail, lands } = candidates(model, pool), x = deck;
  const spells = new Map([...x].filter(([id]) => !isLand(model.set.cards[id])));
  const n = [...x].reduce((a, [id, k]) => a + (spells.has(id) ? 0 : k), 0);
  const utils = utilityLands(model.set, spells, lands);
  const st = { spells, n, util: utils.filter((id) => x.has(id)) };
  const g = model.aggregate(x), p = model.winRate(model.logit(g));
  const tmp = { s: new Float64Array(model.D), vv: 0, ae: 0 };
  const out = new Map();
  const score = (next) => {
    const z = rescore(model, g, x, deckOf(model, next, lands), tmp);
    return z === null ? null : model.winRate(z);
  };
  const tryMove = (o, i) => (o === i || (spells.get(i) || 0) >= avail.get(i) ? null
    : score({ ...st, spells: swapped(spells, o, i) }));
  for (const id of avail.keys()) {
    let best = null;
    if (spells.get(id)) {
      for (const i of avail.keys()) { const q = tryMove(id, i); if (q !== null && (best === null || q > best)) best = q; }
      out.set(id, best === null ? null : 100 * (p - best));
    } else {
      for (const o of spells.keys()) { const q = tryMove(o, id); if (q !== null && (best === null || q > best)) best = q; }
      out.set(id, best === null ? null : 100 * (best - p));
    }
  }
  for (const id of utils) {
    const inDeck = st.util.includes(id);
    const q = score({ ...st, util: inDeck ? st.util.filter((v) => v !== id) : [...st.util, id].slice(-MAX_UTILITY) });
    out.set(id, q === null ? null : 100 * (inDeck ? p - q : q - p));
  }
  return out;
}
