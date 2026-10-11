// Full 8-bot draft in Node: timing, and each seat's colours.   node tools/sim_js.mjs [SET] [threads]
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { Draft, feeds, choices, sample, P, K } from '../draft.js';

const code = process.argv[2] || 'BLB';
const set = JSON.parse(readFileSync(new URL(`../data/${code}.json`, import.meta.url)));
ort.env.wasm.numThreads = +(process.argv[3] || 1);
const sess = await ort.InferenceSession.create(readFileSync(new URL('../model/picknet.onnx', import.meta.url)));
const d = new Draft(set);
const ctxs = d.seats.map(() => set.ctx.typical);
let ms = 0, steps = 0;
while (!d.done) {
  const f = feeds(d.seats, d.t, ctxs, set.sid);
  const t0 = performance.now();
  const lg = (await sess.run({
    idx: new ort.Tensor('int32', f.idx, [f.B, f.T, P]), cnt: new ort.Tensor('int32', f.cnt, [f.B, f.T, P]),
    pick: new ort.Tensor('int32', f.pick, [f.B, f.T]), oidx: new ort.Tensor('int32', f.oidx, [f.B, f.T, K]),
    ocnt: new ort.Tensor('int32', f.ocnt, [f.B, f.T, K]), ctx: new ort.Tensor('float32', f.ctx, [f.B, 18]),
    sid: new ort.Tensor('int32', f.sid, [f.B]),
  })).logits.data;
  ms += performance.now() - t0; steps++;
  d.apply(d.seats.map((s, b) => sample(choices(s, d.t, lg, b), 0).card));
}
console.log(`${code}: ${steps} steps, ${(ms / steps).toFixed(0)} ms/step (B=8), ${(ms / 1000).toFixed(1)} s total`);
for (const [i, s] of d.seats.entries()) {
  const c = {};
  for (const id of s.pool) for (const ch of set.cards[id].colors || 'C') c[ch] = (c[ch] || 0) + 1;
  console.log(`seat ${i}: ${Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k, v]) => k + v).join(' ')}`);
}
