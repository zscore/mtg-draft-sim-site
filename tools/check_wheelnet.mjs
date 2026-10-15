// Checks the page's WheelNet inputs (wheelnet.js) against training's: replays tools/fixtures/wheel_SET.json's test
// drafts pick by pick, as the Live tab would (only the draft so far), and compares P(wheel) with the PyTorch model's.
//     node tools/check_wheelnet.mjs SET
import fs from 'fs';
import * as ort from '../node_modules/onnxruntime-web/dist/ort.node.min.mjs';
import { Seat, SLOTS } from '../draft.js';
import { wheelFeeds, wheelOdds, wheelTensors } from '../wheelnet.js';

const code = process.argv[2] ?? 'FRA', root = new URL('..', import.meta.url).pathname;
const read = (p) => JSON.parse(fs.readFileSync(root + p));
const set = read(`data/${code}.json`), meta = read(`data/wheel/${code}.json`), fx = read(`tools/fixtures/wheel_${code}.json`);
const idOf = Object.fromEntries(Object.entries(set.cards).map(([id, c]) => [c.name, +id]));
ort.env.wasm.numThreads = 1;
const sess = await ort.InferenceSession.create(fs.readFileSync(root + meta.model));

let worst = 0, n = 0;
for (const d of fx.drafts) {
  const seat = new Seat(set.pad), n_ = meta.n_picks;
  for (let r = 0; r < 3; r++) for (let p = 0; p < n_; p++) {
    const { pack, pick } = d.picks[r * n_ + p];
    seat.show(r * SLOTS + p, pack.map((x) => idOf[x]));
    seat.take(r * SLOTS + p, idOf[pick]);
  }
  for (const [key, want] of Object.entries(d.p)) {
    const [r, w] = key.split('.').map(Number), t = r * SLOTS + w;
    const f = wheelFeeds(meta, seat, t, [seat.pick[t]], d.ctx);
    const feeds = Object.fromEntries(wheelTensors(meta, f).map((x) => [x.name, new ort.Tensor(x.type, x.data, x.dims)]));
    const got = wheelOdds(meta, f, (await sess.run(feeds)).p.data, t);
    for (const [name, p] of Object.entries(want)) {
      const q = got.get(idOf[name]);
      if (q === undefined) throw new Error(`P${r + 1}p${w + 1}: ${name} missing from the page's passed cards`);
      worst = Math.max(worst, Math.abs(q - p));
      n++;
    }
  }
}
console.log(`${code}: ${n} passed cards in ${fx.drafts.length} drafts, max |page - training| P(wheel) = ${worst.toExponential(2)}`);
if (worst > 1e-4) process.exit(1);
