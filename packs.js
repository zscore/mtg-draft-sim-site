// Booster generation from a set's collation (data/{SET}.json "collation", built from
// tools/collation.py): pick a layout, pick each slot's sheet by its weights, then a card from that
// sheet. Cards within a slot are distinct; "dedupe" slots also avoid cards already in the pack, and
// sets flagged "unique" avoid repeats anywhere.
// Commons are colour-balanced (see balanced()). tools/verify_packs.py checks the output
// against real Arena packs.

function weighted(items, w, rng) {
  let r = rng() * items.reduce((a, x) => a + w(x), 0);
  for (const x of items) if ((r -= w(x)) <= 0) return x;
  return items[items.length - 1];
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// k distinct cards, each draw proportional to w among those not yet drawn.
function drawDistinct(sheet, k, w, rng) {
  const out = new Set();
  const total = w.reduce((a, b) => a + b, 0);
  while (out.size < Math.min(k, sheet.length)) {
    let r = rng() * total, i = 0;
    while ((r -= w[i]) > 0 && i < sheet.length - 1) i++;
    out.add(sheet[i]); // a repeat is just redrawn, which is the same as drawing from the rest
  }
  return [...out];
}

// Arena's colour rules for commons (not documented; measured from 17Lands packs): "cover" = the five
// colours all show up (a gold card counts for each of its colours), "cap" = at most that many
// commons of any one colour (mono-coloured cards only, or gold ones too with "capGold"); neither
// applies to fewer than "min" commons.
function acceptable(cards, colorOf, rule) {
  if (cards.length < (rule.min || 0)) return true;
  if (rule.cover && new Set(cards.map(colorOf).join('')).size < 5) return false;
  if (rule.cap) {
    const n = {};
    for (const c of cards) {
      const g = colorOf(c);
      if (g.length !== 1 && !rule.capGold) continue;
      for (const x of g) if ((n[x] = (n[x] || 0) + 1) > rule.cap) return false;
    }
  }
  return true;
}

// Weights that make balanced() pick every card equally often: redrawing until the colour rule holds
// favours some cards (gold ones cover two colours; colourless ones never break a cap). One weight
// per colour group, fitted by simulation. tools/fit_balance.mjs stores them in the set's data
// (collation.balance.weights[k]) because fitting takes seconds; otherwise they are fitted here.
export function fitBalanceWeights(sheet, k, colorOf, rule) {
  const groups = sheet.map(colorOf), gw = new Map(groups.map((g) => [g, 1]));
  const size = new Map();
  for (const g of groups) size.set(g, (size.get(g) || 0) + 1);
  const rng = mulberry32(12345), N = 4000;
  for (let it = 0; it < 30; it++) {
    const w = groups.map((g) => gw.get(g)), hits = new Map();
    for (let n = 0; n < N; n++) {
      for (const c of pickAcceptable(sheet, k, w, colorOf, rule, rng)) hits.set(colorOf(c), (hits.get(colorOf(c)) || 0) + 1);
    }
    for (const [g, s] of size) gw.set(g, gw.get(g) * ((k / sheet.length) / (((hits.get(g) || 0) + 1) / (N * s))) ** 0.7);
  }
  return Object.fromEntries(gw);
}

const fits = new WeakMap();
function balanceWeights(sheet, k, colorOf, rule) {
  if (!fits.has(sheet)) fits.set(sheet, new Map());
  const memo = fits.get(sheet);
  if (!memo.has(k)) {
    const gw = rule.weights?.[k] || fitBalanceWeights(sheet, k, colorOf, rule);
    memo.set(k, sheet.map((c) => gw[colorOf(c)]));
  }
  return memo.get(k);
}

function pickAcceptable(sheet, k, w, colorOf, rule, rng) {
  let cards;
  for (let tries = 0; tries < 2000; tries++) {
    cards = drawDistinct(sheet, k, w, rng);
    if (acceptable(cards, colorOf, rule)) break;
  }
  return cards;
}

// k distinct cards from `sheet` that satisfy the set's colour rule, every card equally likely.
// Redrawing until the rule holds also reproduces how often same-coloured commons share a pack.
export function balanced(sheet, k, colorOf, rule, rng = Math.random) {
  if (k <= 0) return [];
  return pickAcceptable(sheet, k, balanceWeights(sheet, k, colorOf, rule), colorOf, rule, rng);
}

// One pack: card ids, repeated per copy.
export function openPack(set, rng = Math.random) {
  const col = set.collation;
  const layout = weighted(col.layouts, (l) => l.w, rng);
  const colorOf = (c) => set.cards[c].colors;
  const slots = layout.slots.map((sl) => ({ ...sl, sheets: Array.from({ length: sl.n }, () => weighted(sl.w, (x) => x[1], rng)[0]) }));
  const bal = col.balance; // {sheet, cover, cap, all}: "all" also balances commons from the wildcard slots
  const isBal = (sl, s) => (sl.balance || bal.all) && s === bal.sheet;
  const k = slots.reduce((a, sl) => a + sl.sheets.filter((s) => isBal(sl, s)).length, 0);
  const pack = balanced(col.sheets[bal.sheet], k, colorOf, bal, rng);
  for (const sl of slots) {
    const mine = [];
    for (const s of sl.sheets) {
      if (isBal(sl, s)) continue;
      const sheet = col.sheets[s];
      let c, tries = 0;
      do c = sheet[Math.floor(rng() * sheet.length)];
      while ((mine.includes(c) || ((col.unique || sl.dedupe) && pack.includes(c))) && ++tries < 100);
      mine.push(c);
      pack.push(c);
    }
  }
  return pack;
}
