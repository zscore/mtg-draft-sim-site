// Fit packs.js's colour-balance weights for each possible number of balanced commons and store them
// in data/{SET}.json (collation.balance.weights), so the page doesn't spend seconds fitting them.
// Run by tools/build_data.py.   node tools/fit_balance.mjs SET ...
import { readFileSync, writeFileSync } from 'node:fs';
import { fitBalanceWeights } from '../packs.js';

for (const code of process.argv.slice(2)) {
  const path = new URL(`../data/${code}.json`, import.meta.url);
  const set = JSON.parse(readFileSync(path));
  const col = set.collation, bal = col.balance;
  const sheet = col.sheets[bal.sheet], colorOf = (c) => set.cards[c].colors;
  // The numbers of balanced commons a pack can have: each balanced slot always, sometimes or never
  // draws from the balanced sheet. (Fitting a k the rule can't satisfy would just spin.)
  const ks = new Set();
  for (const l of col.layouts) {
    let lo = 0, hi = 0;
    for (const sl of l.slots) {
      if (!(sl.balance || bal.all) || !sl.w.some(([s]) => s === bal.sheet)) continue;
      hi += sl.n;
      if (sl.w.every(([s]) => s === bal.sheet)) lo += sl.n;
    }
    for (let k = Math.max(lo, bal.min || 1); k <= hi; k++) ks.add(k);
  }
  bal.weights = {};
  for (const k of [...ks].sort((a, b) => a - b)) {
    const w = fitBalanceWeights(sheet, k, colorOf, bal);
    bal.weights[k] = Object.fromEntries(Object.entries(w).map(([g, x]) => [g, +x.toPrecision(4)]));
  }
  writeFileSync(path, JSON.stringify(set));
  console.log(`${code}: balance weights for ${Object.keys(bal.weights).length} pack sizes`);
}
