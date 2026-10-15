// Runs PickNet off the main thread. Messages: {type:'load', url} -> progress*, ready | error;
// {type:'run', id, feeds} -> {type:'result', id, logits, ms}. Other models (WheelNet) by name:
// {type:'load', name, url} -> {type:'loaded', name} | error; {type:'run', id, model, tensors: [{name, type, data, dims}]}
// -> {type:'result', id, out, ms} (the first output's data). Runs go one at a time: models share the threads.
import * as ort from './vendor/ort/ort.wasm.min.mjs';

let session, queue = Promise.resolve();
const sessions = {};

async function fetchWithProgress(url, progress = true) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  // a compressed response's content-length counts the compressed bytes (GitHub Pages gzips the models), while the
  // body comes unzipped, so it can't be the total
  const total = res.headers.get('content-encoding') ? 0 : +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    if (progress) self.postMessage({ type: 'progress', got, total });
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  return buf;
}

self.onmessage = ({ data: m }) => { queue = queue.then(() => handle(m)); };

async function handle(m) {
  try {
    if (m.type === 'load' && m.name) {
      sessions[m.name] = await ort.InferenceSession.create(await fetchWithProgress(m.url, false),
        { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      self.postMessage({ type: 'loaded', name: m.name });
    } else if (m.type === 'run' && m.model) {
      const t0 = performance.now();
      const out = await sessions[m.model].run(Object.fromEntries(m.tensors.map((x) => [x.name, new ort.Tensor(x.type, x.data, x.dims)])));
      self.postMessage({ type: 'result', id: m.id, out: Object.values(out)[0].data, ms: performance.now() - t0 });
    } else if (m.type === 'load') {
      ort.env.wasm.wasmPaths = new URL('./vendor/ort/', self.location.href).href;
      const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
      ort.env.wasm.numThreads = threads;
      const t0 = performance.now();
      const buf = await fetchWithProgress(m.url);
      session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      self.postMessage({ type: 'ready', threads, bytes: buf.length, ms: performance.now() - t0 });
    } else if (m.type === 'run') {
      const f = m.feeds, P = 15, K = 8;
      const t0 = performance.now();
      const out = await session.run({
        idx: new ort.Tensor('int32', f.idx, [f.B, f.T, P]), cnt: new ort.Tensor('int32', f.cnt, [f.B, f.T, P]),
        pick: new ort.Tensor('int32', f.pick, [f.B, f.T]), oidx: new ort.Tensor('int32', f.oidx, [f.B, f.T, K]),
        ocnt: new ort.Tensor('int32', f.ocnt, [f.B, f.T, K]),
        ctx: new ort.Tensor('float32', f.ctx, [f.B, f.ctx.length / f.B]), sid: new ort.Tensor('int32', f.sid, [f.B]),
      });
      self.postMessage({ type: 'result', id: m.id, logits: out.logits.data, ms: performance.now() - t0 });
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: m.id, name: m.name, message: String(e && e.message || e) });
  }
}
