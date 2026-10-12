// The browser deck scorer (deck.js) against PyTorch's deck terms on training decks (tools/fixtures/deck_{SET}.json,
// written by tools/export_deck_model.py), then a timed deck build from a random sealed pool.
//     node tools/verify_deck.mjs [SET ...]
import { readFileSync, readdirSync } from 'node:fs';
import { DeckModel, buildDeck, cardValues } from '../deck.js';
import { openPack } from '../packs.js';

const sets = process.argv.slice(2).length ? process.argv.slice(2)
  : readdirSync('data/deck').map((f) => f.replace('.json', ''));
let worst = 0;
for (const code of sets) {
  const json = JSON.parse(readFileSync(`data/deck/${code}.json`)), set = JSON.parse(readFileSync(`data/${code}.json`));
  const idOf = new Map(Object.entries(set.cards).map(([id, c]) => [c.name, +id]));
  const fx = JSON.parse(readFileSync(`tools/fixtures/deck_${code}.json`));
  const line = [code];
  for (const event of ['draft', 'sealed']) {
    const model = new DeckModel(json, event, set);
    let err = 0;
    for (const { deck, logit } of fx[event]) {
      const counts = new Map(Object.entries(deck).map(([j, n]) => [idOf.get(json.names[+j]), n]));
      err = Math.max(err, Math.abs(model.rawLogit(model.aggregate(counts)) - logit));
    }
    worst = Math.max(worst, err);
    const pool = Array.from({ length: 6 }, () => openPack(set)).flat();
    const t0 = performance.now();
    const built = buildDeck(model, pool);
    const ms = performance.now() - t0;
    cardValues(model, built.counts, pool);
    const ms2 = performance.now() - t0 - ms;
    const desc = [...built.counts].map(([id, n]) => `${n} ${set.cards[id].name}`).join(', ');
    line.push(`${event}: max |err| ${err.toExponential(1)}, build ${ms.toFixed(0)} ms + values ${ms2.toFixed(0)} ms, ` +
      `${(100 * built.winRate).toFixed(1)}% (typical ${(100 * model.typical).toFixed(1)}%)`);
    if (sets.length === 1) line.push(`  ${desc}`);
  }
  console.log(line.join('\n  '));
}
console.log(`worst |logit error| ${worst.toExponential(2)}`);
process.exit(worst < 2e-3 ? 0 : 1);
