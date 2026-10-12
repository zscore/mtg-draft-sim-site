// Open N packs with packs.js and print them as JSON (arrays of card ids), for tools/verify_packs.py.
//   node tools/gen_packs.mjs SET N [seed]
import { readFileSync } from 'node:fs';
import { openPack } from '../packs.js';

const [code, n, seed = '1'] = process.argv.slice(2);
const set = JSON.parse(readFileSync(new URL(`../data/${code}.json`, import.meta.url)));
let a = +seed >>> 0; // mulberry32
const rng = () => {
  a = (a + 0x6d2b79f5) >>> 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
process.stdout.write(JSON.stringify(Array.from({ length: +n }, () => openPack(set, rng))));
