// Replay real drafts through draft.js's token building + the ONNX model (onnxruntime-web in Node) and
// compare with PyTorch's logits.   node tools/verify_js.mjs
import { readFileSync } from 'node:fs';
import * as ort from 'onnxruntime-web';
import { Seat, feeds, P } from '../draft.js';

const fx = JSON.parse(readFileSync(new URL('./fixtures/replay.json', import.meta.url)));
const pad = JSON.parse(readFileSync(new URL('../data/BLB.json', import.meta.url))).pad;
ort.env.wasm.numThreads = 1;
const sess = await ort.InferenceSession.create(readFileSync(new URL('../model/picknet.onnx', import.meta.url)));
const tensor = (f) => ({
  idx: new ort.Tensor('int32', f.idx, [f.B, f.T, P]), cnt: new ort.Tensor('int32', f.cnt, [f.B, f.T, P]),
  pick: new ort.Tensor('int32', f.pick, [f.B, f.T]), oidx: new ort.Tensor('int32', f.oidx, [f.B, f.T, 8]),
  ocnt: new ort.Tensor('int32', f.ocnt, [f.B, f.T, 8]), ctx: new ort.Tensor('float32', f.ctx, [f.B, f.ctx.length / f.B]),
  sid: new ort.Tensor('int32', f.sid, [f.B]),
});
let worst = 0, n = 0, ms = 0;
for (const d of fx) {
  const s = new Seat(pad);
  for (const sl of d.slots) {
    s.show(sl.t, sl.pack);
    if (sl.pack.length) {
      const t0 = performance.now();
      const out = (await sess.run(tensor(feeds([s], sl.t, [d.ctx], d.sid)))).logits.data;
      ms += performance.now() - t0;
      for (const tok of s.tokens(sl.t)) {
        worst = Math.max(worst, Math.abs(out[tok.j] - sl.logit[tok.card]));
        n++;
      }
    }
    s.take(sl.t, sl.pick);
  }
}
console.log(`${fx.length} drafts, ${n} card logits: max |js - torch| = ${worst.toExponential(2)}; ${(ms / fx.length / 39).toFixed(1)} ms/step (B=1, 1 thread)`);
