import { Draft, Seat, WheelSim, feeds, choices, sample, rateCards, SEATS, SLOTS } from './draft.js';
import { newSeed, openPack, seededRng } from './packs.js';
import { wheelFeeds, wheelOdds, wheelTensors } from './wheelnet.js';
import { ArenaDrafts, eventSet, isBotDraft, isSealed, mergeDrafts, parseEntry, placeSteps } from './arena.js';
import { DeckModel, buildDeck, cardValues, deckPerColors, isLand, landCount, manaBase, utilityLands, BASICS } from './deck.js';

const $ = (s) => document.querySelector(s);
const COLORS = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };
const RARITY = { m: 0, r: 1, s: 1, u: 2, c: 3 };

// ---------------------------------------------------------------- model worker

const worker = new Worker('./worker.js', { type: 'module' });
const pending = new Map();
let nextId = 0, onReady;
const ready = new Promise((res) => { onReady = res; });

worker.onmessage = ({ data: m }) => {
  if (m.type === 'progress') {
    status(`Loading PickNet… ${(m.got / 1e6).toFixed(1)}${m.total ? ` / ${(m.total / 1e6).toFixed(1)}` : ''} MB`);
  } else if (m.type === 'ready') {
    model = m;
    status(`PickNet ready · ${(m.bytes / 1e6).toFixed(0)} MB loaded in ${(m.ms / 1000).toFixed(1)} s · ${m.threads} thread${m.threads > 1 ? 's' : ''}`);
    onReady();
  } else if (m.type === 'result') {
    pending.get(m.id).res(m);
    pending.delete(m.id);
  } else if (m.type === 'loaded') {
    modelLoads.get(m.name)?.res();
    modelLoads.delete(m.name);
  } else if (m.type === 'error') {
    if (m.name && modelLoads.has(m.name)) { modelLoads.get(m.name).rej(new Error(m.message)); modelLoads.delete(m.name); return; }
    if (pending.has(m.id)) { pending.get(m.id).rej(new Error(m.message)); pending.delete(m.id); }
    status(`Error: ${m.message}`);
  }
};
worker.postMessage({ type: 'load', url: new URL('./model/picknet.onnx', location.href).href });

function run(f) {
  return new Promise((res, rej) => {
    const id = ++nextId;
    pending.set(id, { res, rej });
    worker.postMessage({ type: 'run', id, feeds: f }, [f.idx.buffer, f.cnt.buffer, f.pick.buffer, f.oidx.buffer,
      f.ocnt.buffer, f.ctx.buffer, f.sid.buffer]);
  });
}

// Another model in the worker, by name: tensors [{name, type, data, dims}] -> {out, ms}.
function runModel(name, tensors) {
  return new Promise((res, rej) => {
    const id = ++nextId;
    pending.set(id, { res, rej });
    worker.postMessage({ type: 'run', id, model: name, tensors });
  });
}

// A set's WheelNet, once tools/export_wheelnet.py has made one: its data/wheel/{SET}.json, or null.
const modelLoads = new Map(), wheelNets = {};
function loadWheelNet(code) {
  return wheelNets[code] ??= fetch(`data/wheel/${code}.json`).then((r) => (r.ok ? r.json() : null))
    .then((meta) => meta && new Promise((res, rej) => {
      modelLoads.set(`wheel:${code}`, { res: () => res(meta), rej });
      worker.postMessage({ type: 'load', name: `wheel:${code}`, url: new URL(meta.model, location.href).href });
    }))
    .catch((e) => { console.warn(`WheelNet for ${code}: ${e.message}`); return null; });
}

// ---------------------------------------------------------------- state

// history: every step's picks (seat 0 = you), so undo can replay the draft from its opening packs.
let model, set, draft, openers, history, step, log, selected = null, busy = false, autodrafting = false;
let draftId, draftTime, poolSeat = 0, draftBuilds = [];
// set is whichever set is on screen: the draft's (draftSet) or the sealed pool's (sealed.set). sealed: the sealed
// pool's deck builder.
let mode = 'draft', draftSet, sealed, draftSeed;
const setCache = {};
const pref = {
  get(k, d) { try { return localStorage.getItem(`picknet.${k}`) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`picknet.${k}`, v); } catch { /* storage unavailable */ } },
};

function status(s) { $('#status').textContent = s; }

async function loadSet(code) {
  if (!setCache[code]) setCache[code] = await (await fetch(`data/${code}.json`)).json();
  return setCache[code];
}

// Run the model for all 8 seats at the current slot; the human's pack is shown meanwhile. The bots
// decide as soon as the logits arrive, so the god's-eye view shows what they are about to take.
function beginStep() {
  const t = draft.t, ctx = set.ctx[$('#skill').value], temp = +$('#style').value;
  const f = feeds(draft.seats, t, draft.seats.map(() => ctx), set.sid);
  const s = { t, choices: null };
  s.done = run(f).then((r) => {
    s.choices = draft.seats.map((seat, b) => choices(seat, t, r.logits, b));
    // each bot's draw comes from the seed and the pick, so an undone and redone pick draws the same
    s.botPicks = s.choices.map((ch, i) => (i === 0 ? null : sample(ch, temp, seededRng(`${draftSeed}:${t}:${i}`)).card));
    s.ms = r.ms;
    return s;
  });
  return s;
}

async function newDraft() {
  autodrafting = false;
  $('#new').disabled = true;
  set = draftSet = await loadSet($('#set').value);
  await ready;
  if (mode !== 'draft') return void ($('#new').disabled = false);
  draftSeed = takeSeed();
  draft = new Draft(set, seededRng(draftSeed));
  openers = draft.openers.map((p) => p.slice());
  showSeed();
  history = [];
  log = [];
  draftId = globalThis.crypto?.randomUUID?.().replace(/-/g, '') ??
    Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
  draftTime = new Date().toISOString().replace('T', ' ').slice(0, 19);
  $('#intro').hidden = true;
  $('#new').disabled = false;
  next();
}

function next() {
  selected = null;
  $('#undo').disabled = !history.length;
  if (draft.done) return mode === 'draft' && finish();
  step = beginStep();
  if (mode !== 'draft') return; // an autodraft carrying on while sealed is on screen
  $('#end-view').hidden = $('#log-view').hidden = $('#build-view').hidden = true;
  draftBuilds = [];
  $('#pick-view').hidden = $('#pool-view').hidden = false;
  renderPack();
  renderGod();
  renderPool(0);
  step.done.then((s) => {
    if (s !== step || mode !== 'draft') return;
    status(`PickNet: 8 seats in ${s.ms.toFixed(0)} ms · ${model.threads} thread${model.threads > 1 ? 's' : ''}`);
    renderPack();
    renderGod();
  });
}

// Clicking a card selects it; clicking it again, Enter, or the confirm button takes it.
function select(card) {
  if (busy || draft.done) return;
  if (selected === card) return commit(card);
  selected = card;
  renderPack();
}

// Take `card` (null: PickNet's top choice) and the bots' picks, then pass.
async function commit(card) {
  if (busy || draft.done) return;
  busy = true;
  $('#pack').classList.add('busy');
  try {
    const s = await step.done;
    if (s !== step) return;
    const mine = s.choices[0];
    const choice = card ?? mine[0].card;
    const rank = mine.findIndex((x) => x.card === choice);
    const picks = s.botPicks.map((c, i) => (i === 0 ? choice : c));
    log.push({ t: s.t, card: choice, p: mine[rank].p, rank, top: mine[0].card, topP: mine[0].p, n: mine.length });
    history.push(picks);
    draft.apply(picks);
  } finally {
    busy = false;
    $('#pack').classList.remove('busy');
  }
  next();
}

async function autodraftRest() {
  if (busy || autodrafting) return;
  autodrafting = true;
  while (autodrafting && !draft.done) await commit(null);
  autodrafting = false;
}

// Rebuild the draft from its opening packs and every pick but the last.
function undo() {
  if (busy || !history?.length) return;
  autodrafting = false;
  history.pop();
  log.pop();
  draft = new Draft(set, Math.random, openers);
  for (const picks of history) draft.apply(picks);
  next();
}

// ---------------------------------------------------------------- rendering

function cardEl(id, small = false) {
  const c = set.cards[id];
  const el = document.createElement('div');
  el.className = 'card';
  el.title = c.name;
  const name = document.createElement('div');
  name.className = 'name';
  name.textContent = c.name;
  el.append(name);
  if (c.img) {
    const img = document.createElement('img');
    img.src = small ? smallImg(c.img) : c.img;
    img.alt = c.name;
    img.loading = 'lazy';
    img.onerror = () => { img.remove(); el.classList.add('noimg'); };
    el.append(img);
  } else {
    el.classList.add('noimg');
  }
  return el;
}

function colorKey(c) {
  return c.colors.length === 0 ? 6 : c.colors.length > 1 ? 5 : 'WUBRG'.indexOf(c.colors);
}

// The pack's card elements are built once per step (so images don't reload); later calls only update
// the selection and PickNet's overlay.
let packStep = null;
const packEls = new Map();

function renderPack() {
  const t = draft.t, round = Math.floor(t / SLOTS), k = t % SLOTS;
  const pack = $('#pack');
  if (packStep !== step) {
    packStep = step;
    $('#pick-title').textContent = `Pack ${round + 1} · Pick ${k + 1} of ${set.n_picks}`;
    const toks = draft.seats[0].tokens(t).sort((a, b) => {
      const A = set.cards[a.card], B = set.cards[b.card];
      return (RARITY[A.rarity] ?? 4) - (RARITY[B.rarity] ?? 4) || colorKey(A) - colorKey(B) || A.cmc - B.cmc;
    });
    packEls.clear();
    pack.replaceChildren(...toks.map((tok) => {
      const el = cardEl(tok.card);
      el.onclick = () => select(tok.card);
      if (tok.copies > 1) {
        const b = document.createElement('span');
        b.className = 'copies';
        b.textContent = `×${tok.copies}`;
        el.append(b);
      }
      const o = document.createElement('div');
      o.className = 'model';
      el.append(o);
      packEls.set(tok.card, el);
      return el;
    }));
  }
  const ch = step.choices ? step.choices[0] : [];
  const rankOf = new Map(ch.map((x, i) => [x.card, i]));
  for (const [card, el] of packEls) {
    el.classList.toggle('selected', card === selected);
    el.classList.toggle('top1', rankOf.get(card) === 0);
    const r = rankOf.get(card), o = el.querySelector('.model');
    if (r === undefined) {
      o.textContent = '';
    } else {
      const p = ch[r].p;
      o.innerHTML = `#${r + 1} · ${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%<div class="meter"><i style="width:${(p * 100).toFixed(1)}%"></i></div>`;
    }
  }
  pack.classList.toggle('show-model', $('#show').checked);
  const btn = $('#confirm');
  btn.disabled = selected === null;
  btn.textContent = selected === null ? 'Select a card' : `Take ${set.cards[selected].name}`;
}

const passTo = (i, round) => (i + (round === 1 ? SEATS - 1 : 1)) % SEATS;
const seatName = (i) => (i === 0 ? 'You' : `Bot ${i}`);
const smallImg = (url) => url.replace('/normal/', '/small/');

// Every seat's pack in hand, PickNet's read of it, and what each bot is about to take. Built once per
// step like the pack; when the logits arrive, each pack is re-ordered by PickNet's probability.
let godStep = null;
const godSlots = []; // per seat: Map card -> slot element
const closedPiles = new Set(); // seats whose pile the viewer collapsed

function renderGod() {
  const view = $('#god-view');
  view.hidden = !$('#god').checked || draft.done;
  if (view.hidden) return;
  const t = draft.t, round = Math.floor(t / SLOTS);
  if (godStep !== step) {
    godStep = step;
    $('#god-flow').textContent = `Pack ${round + 1} passes ${round === 1 ? 'right: each seat passes to the row above'
      : 'left: each seat passes to the row below'}. Gold: what each bot is taking. Blue: PickNet's pick for you.`;
    godSlots.length = 0;
    $('#god-table').replaceChildren(...draft.seats.map((seat, i) => {
      const row = document.createElement('div');
      row.className = `seat-row${i === 0 ? ' you' : ''}`;
      const info = document.createElement('div');
      info.className = 'seat-info';
      const last = seat.pool.length ? set.cards[seat.pool[seat.pool.length - 1]].name : '—';
      const h = document.createElement('h3');
      h.textContent = seatName(i);
      const m1 = document.createElement('div');
      m1.className = 'meta';
      m1.textContent = `${seat.pool.length} picks · passes to ${seatName(passTo(i, round))}`;
      const m2 = document.createElement('div');
      m2.className = 'meta';
      m2.textContent = `Last took: ${last}`;
      const chips = document.createElement('div');
      chips.className = 'chips';
      chips.append(...colorChips(seat.pool));
      info.append(h, m1, m2, chips);
      const mini = document.createElement('div');
      mini.className = 'mini';
      const slots = new Map();
      for (const x of seat.tokens(t)) {
        const slot = document.createElement('div');
        slot.className = 'slot';
        const el = cardEl(x.card, true);
        if (x.copies > 1) {
          const b = document.createElement('span');
          b.className = 'copies';
          b.textContent = `×${x.copies}`;
          el.append(b);
        }
        const lab = document.createElement('span');
        lab.className = 'p';
        lab.textContent = '…';
        slot.append(el, lab);
        slots.set(x.card, slot);
        mini.append(slot);
      }
      godSlots.push(slots);
      const pile = document.createElement('details');
      pile.className = 'pile';
      pile.open = !closedPiles.has(i);
      pile.ontoggle = () => (pile.open ? closedPiles.delete(i) : closedPiles.add(i));
      const sum = document.createElement('summary');
      sum.textContent = seat.pool.length ? `Picks (${seat.pool.length})` : 'No picks yet';
      const cols = document.createElement('div');
      cols.className = 'pool mini-pool';
      cols.append(...pileCols(seat.pool, true));
      pile.append(sum, cols);
      row.append(info, mini, pile);
      return row;
    }));
  }
  if (!step.choices) return;
  step.choices.forEach((ch, i) => {
    const slots = godSlots[i], taking = i === 0 ? ch[0].card : step.botPicks[i];
    for (const x of ch) {
      const slot = slots.get(x.card);
      slot.classList.toggle('taking', i > 0 && x.card === taking);
      slot.classList.toggle('suggest', i === 0 && x.card === taking);
      slot.querySelector('.p').textContent = `${(x.p * 100).toFixed(x.p < 0.1 ? 1 : 0)}%`;
      slot.querySelector('.card').title = `${set.cards[x.card].name} · ${(x.p * 100).toFixed(1)}%`;
      slot.parentNode.append(slot); // best first
    }
  });
}

function colorChips(ids) {
  const n = {};
  for (const id of ids) for (const ch of set.cards[id].colors || 'C') n[ch] = (n[ch] || 0) + 1;
  return Object.entries(n).sort((a, b) => b[1] - a[1]).map(([k, v]) => {
    const s = document.createElement('span');
    s.className = 'chip';
    s.innerHTML = `<i style="background:var(--${k})"></i>${COLORS[k] || 'Colorless'} ${v}`;
    return s;
  });
}

// A pile of picks in mana-value columns (lands first), each column sorted by colour (or by `order`,
// a card id comparator). decorate(el, id) can add to each card. stack: one card per name with a ×n count.
function pileCols(ids, small = false, decorate = null, order = null, stack = false) {
  const cols = new Map([['Lands', []], ['0–1', []], ['2', []], ['3', []], ['4', []], ['5', []], ['6+', []]]);
  const copies = new Map();
  for (const id of ids) {
    copies.set(id, (copies.get(id) || 0) + 1);
    if (stack && copies.get(id) > 1) continue;
    const c = set.cards[id];
    const key = isLand(c) ? 'Lands' : c.cmc <= 1 ? '0–1' : c.cmc >= 6 ? '6+' : String(c.cmc);
    cols.get(key).push(id);
  }
  return [...cols].filter(([, v]) => v.length).map(([k, v]) => {
    const col = document.createElement('div');
    col.className = 'col';
    const h = document.createElement('h3');
    h.textContent = `${k} (${stack ? v.reduce((a, id) => a + copies.get(id), 0) : v.length})`;
    v.sort(order || ((a, b) => colorKey(set.cards[a]) - colorKey(set.cards[b])));
    col.append(h, ...v.map((id) => {
      const el = cardEl(id, small);
      if (stack && copies.get(id) > 1) {
        const n = document.createElement('span');
        n.className = 'copies';
        n.textContent = `×${copies.get(id)}`;
        el.append(n);
      }
      decorate?.(el, id);
      // the count and any labels side by side at the top right, clear of the card's name
      const tags = el.querySelectorAll(':scope > .copies, :scope > .rate, :scope > .basic-ctl');
      if (tags.length) {
        const box = document.createElement('span');
        box.className = 'tags';
        box.append(...tags);
        el.append(box);
      }
      return el;
    }));
    return col;
  });
}

function renderPool(seatIdx) {
  poolSeat = seatIdx;
  const ids = draft.seats[seatIdx].pool;
  $('#pool-title').textContent = seatIdx === 0 ? `Your picks (${ids.length})` : `Bot ${seatIdx}'s picks (${ids.length})`;
  $('#pool-colors').replaceChildren(...colorChips(ids));
  $('#pool').replaceChildren(...pileCols(ids));
}

function finish() {
  $('#pick-view').hidden = $('#god-view').hidden = true;
  $('#end-view').hidden = $('#log-view').hidden = $('#pool-view').hidden = false;
  const agree = log.filter((x) => x.rank === 0).length;
  const real = log.filter((x) => x.n > 1);
  $('#agree').textContent = `You took PickNet's top choice on ${agree} of ${log.length} picks ` +
    `(${real.filter((x) => x.rank === 0).length} of ${real.length} excluding forced last picks).`;
  const tabs = $('#seat-tabs');
  tabs.replaceChildren(...draft.seats.map((_, i) => {
    const b = document.createElement('button');
    b.textContent = i === 0 ? 'You' : `Bot ${i}`;
    b.onclick = () => { renderPool(i); showDraftBuild(i); [...tabs.children].forEach((x, j) => x.classList.toggle('on', j === i)); };
    if (i === 0) b.classList.add('on');
    return b;
  }));
  renderPool(0);
  showDraftBuild(0);
  const row = (cells, cls = '') => `<tr class="${cls}">${cells.map((c) => `<td>${c}</td>`).join('')}</tr>`;
  const esc = (s) => s.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
  $('#log').innerHTML = '<tr><th>Pick</th><th>You took</th><th>PickNet\'s rank</th><th>PickNet\'s top choice</th></tr>' +
    log.map((x) => row([
      `P${Math.floor(x.t / SLOTS) + 1}p${(x.t % SLOTS) + 1}`, esc(set.cards[x.card].name),
      `#${x.rank + 1} (${(x.p * 100).toFixed(0)}%)`, x.rank === 0 ? '—' : `${esc(set.cards[x.top].name)} (${(x.topP * 100).toFixed(0)}%)`,
    ], x.rank === 0 ? 'agree' : '')).join('');
  status(`Draft complete · PickNet ran ${log.length} steps for 8 seats`);
}

// ---------------------------------------------------------------- export

// One row per pick for every seat, in 17Lands' draft_data schema (pack/pick numbers from 0, a
// pack_card_ and pool_ count column for every card in the set). Each seat gets its own draft_id;
// table_id and seat tie the eight together.
function exportRows() {
  const ids = Object.keys(set.cards).map(Number);
  const rows = [];
  draft.seats.forEach((seat, s) => {
    const pool = new Map();
    for (let round = 0; round < 3; round++) {
      for (let k = 0; k < set.n_picks; k++) {
        const t = round * SLOTS + k;
        if (seat.pick[t] === set.pad) break; // not picked yet
        const pick = seat.pick[t], pack = new Map();
        for (const c of seat.seen[t]) pack.set(c, (pack.get(c) || 0) + 1);
        const row = {
          expansion: set.code, event_type: 'PickNetSim', draft_id: `${draftId}${s}`, table_id: draftId, seat: s,
          drafter: s === 0 ? 'human' : 'picknet', draft_time: draftTime, pack_number: round, pick_number: k,
          pick: set.cards[pick].name,
        };
        for (const c of ids) row[`pack_card_${set.cards[c].name}`] = pack.get(c) || 0;
        for (const c of ids) row[`pool_${set.cards[c].name}`] = pool.get(c) || 0;
        rows.push(row);
        pool.set(pick, (pool.get(pick) || 0) + 1);
      }
    }
  });
  return rows;
}

function download(name, type, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 0);
}

const fileBase = () => `picknet_${set.code}_${draftTime.replace(/[^0-9]/g, '').slice(0, 12)}`;

function exportCsv() {
  const rows = exportRows(), cols = Object.keys(rows[0]);
  const q = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  download(`${fileBase()}.csv`, 'text/csv', [cols.map(q).join(','), ...rows.map((r) => cols.map((c) => q(r[c])).join(','))].join('\n') + '\n');
}

// Same rows as the CSV, minus the zero card counts.
function exportJsonl() {
  const rows = exportRows().map((r) => Object.fromEntries(Object.entries(r).filter(([k, v]) => v !== 0 || !/^(pack_card|pool)_/.test(k))));
  download(`${fileBase()}.jsonl`, 'application/x-ndjson', rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
}

// The shown seat's picks as an Arena-style list ("2 Card Name"), in pick order.
async function copyDeck() {
  const n = new Map();
  for (const id of draft.seats[poolSeat].pool) n.set(id, (n.get(id) || 0) + 1);
  const text = [...n].map(([id, k]) => `${k} ${set.cards[id].name}`).join('\n');
  const btn = $('#copy-deck');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied!';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = 'Copy decklist'; }, 1500);
}

// ---------------------------------------------------------------- deck builder

// One builder per pool: the sealed pool, and each seat's picks once a draft is done. A builder holds the deck
// (card ids, basics apart), the sideboard, the deck model for the pool's event (null if the set has none) and
// PickNet's ratings. It renders into a copy of #builder-tpl.
const SEALED_PACKS = 6;
// The five basics are the deck's own (b.basics), never in the sideboard; ones opened in packs are left out.
const basicKey = (c) => (/Basic/.test(c.type) ? Object.keys(BASICS).find((k) => BASICS[k] === c.name) : undefined);
const basicId = (s, k) => (s.basicIds ??= Object.fromEntries(Object.entries(s.cards)
  .map(([id, c]) => [basicKey(c), +id]).filter(([key]) => key)))[k];
const notBasic = (s) => (id) => !basicKey(s.cards[id]);
const deckModels = {};

async function loadDeckModel(s, event) {
  const key = `${s.code}.${event}`;
  if (!(key in deckModels)) {
    try {
      const r = await fetch(`data/deck/${s.code}.json`);
      deckModels[key] = r.ok ? new DeckModel(await r.json(), event, s) : null;
    } catch {
      deckModels[key] = null;
    }
  }
  return deckModels[key];
}

// The seed box's seed, or a fresh one if it is empty.
const takeSeed = () => $('#seed').value.trim() || newSeed();

// The seed of the draft or sealed pool on screen, as a button that copies it into the seed box.
function showSeed() {
  const seed = mode === 'sealed' ? sealed?.seed : draft && draftSeed;
  const btn = $('#seed-used');
  btn.hidden = !seed;
  btn.textContent = `Seed ${seed}`;
  btn.onclick = () => { $('#seed').value = seed; };
}

async function makeBuilder(s, pool, event, root, seed) {
  if (!root.firstElementChild) root.append($('#builder-tpl').content.cloneNode(true));
  const b = { set: s, pool: pool.slice(), deck: [], side: pool.filter(notBasic(s)), basics: { W: 0, U: 0, B: 0, R: 0, G: 0 },
    event, root, model: await loadDeckModel(s, event), ratings: null, rank: null, values: null };
  root.builder = b; // a sealed pool reuses the root; the old pool's late ratings must not draw over the new one
  b.seed = seed;
  b.ratingsReady = ready.then(() => rateCards(s, [...new Set(pool)], s.ctx[$('#skill').value], (f) => run(f).then((r) => r.logits),
    4, seededRng(`${seed}:rate`)))
    .then((ratings) => {
      b.ratings = ratings;
      b.rank = new Map([...ratings].sort((a, c) => c[1] - a[1]).map(([c], i) => [c, i + 1]));
      renderBuilder(b);
    });
  const q = (sel) => root.querySelector(sel);
  q('.b-best').onclick = () => buildBest(b);
  q('.b-best').disabled = !b.model;
  q('.b-best').title = b.model
    ? `Search for the 40-card deck the ${event === 'sealed' ? 'Sealed' : 'Premier Draft'} deck model rates highest (card power, pair synergy, curve, colours and mana base)`
    : `No deck model for ${s.code}`;
  q('.b-picknet').onclick = () => b.ratings ? buildPickNet(b) : b.ratingsReady.then(() => buildPickNet(b));
  q('.b-clear').onclick = () => {
    b.side.push(...b.deck.splice(0));
    for (const k in b.basics) b.basics[k] = 0;
    b.comboKey = null;
    renderBuilder(b);
  };
  q('.b-copy').onclick = () => copyBuilder(b);
  q('.combos').hidden = true;
  if (b.model) setupCombos(b);
  const labels = q('.b-labels');
  labels.value = pref.get('labels', 'value');
  labels.onchange = () => { pref.set('labels', labels.value); renderBuilder(b); };
  return b;
}

// Deck counts for the model: deck cards plus basics.
function builderCounts(b) {
  const n = new Map();
  for (const id of b.deck) n.set(id, (n.get(id) || 0) + 1);
  for (const [k, v] of Object.entries(b.basics)) if (v) n.set(b.model.basicIds[k], (n.get(b.model.basicIds[k]) || 0) + v);
  return n;
}

function setDeck(b, counts) {
  b.deck = [];
  b.basics = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  const left = new Map();
  for (const id of b.pool.filter(notBasic(b.set))) left.set(id, (left.get(id) || 0) + 1);
  for (const [id, n] of counts) {
    const k = basicKey(b.set.cards[id]);
    if (k) { b.basics[k] += n; continue; }
    for (let i = 0; i < n; i++) b.deck.push(id);
    left.set(id, (left.get(id) || 0) - n);
  }
  b.side = [...left].flatMap(([id, n]) => Array(Math.max(0, n)).fill(id));
}

function buildBest(b) {
  const t0 = performance.now();
  const built = buildDeck(b.model, b.pool, { rng: seededRng(`${b.seed}:build`) });
  setDeck(b, built.counts);
  b.bestRow = { want: wantOf(b, built), deck: built, overall: true };
  b.comboKey = 'best';
  renderBuilder(b);
  status(`Deck model built the best deck it found in ${(performance.now() - t0).toFixed(0)} ms`);
}

// Best two colours by the sum of PickNet's top 23 nonland ratings there (colourless cards count for every pair),
// then its best spells there, as many as the land count for their curve leaves room for, and the same mana base
// the deck model's build gets (deck.js manaBase), with up to two utility lands PickNet rates as a playable.
function buildPickNet(b) {
  const all = b.pool, r = (c) => b.ratings.get(c);
  let best = null;
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      const pair = 'WUBRG'[i] + 'WUBRG'[j];
      const fits = all.filter((c) => !isLand(b.set.cards[c]) && [...b.set.cards[c].colors].every((x) => pair.includes(x)))
        .sort((a, c) => r(c) - r(a));
      const top = fits.slice(0, 23);
      const score = top.reduce((a, c) => a + r(c), 0) - 20 * (23 - top.length);
      if (!best || score > best.score) best = { fits, score };
    }
  }
  // a pair short of spells is topped up with the best of the rest
  const rest = all.filter((c) => !isLand(b.set.cards[c]) && !best.fits.includes(c)).sort((a, c) => r(c) - r(a));
  best.fits = best.fits.concat(rest);
  const counts = (ids) => ids.reduce((m, c) => m.set(c, (m.get(c) || 0) + 1), new Map());
  let deck = best.fits.slice(0, 23);
  for (let round = 0; round < 3; round++) { // the land count follows the curve, which follows the spell count
    const n = Math.min(40 - landCount(b.set, counts(deck)), best.fits.length);
    if (n === deck.length) break;
    deck = best.fits.slice(0, n);
  }
  // utility lands PickNet rates as highly as the deck's weakest spell, in place of basics
  const worst = Math.min(...deck.map(r));
  const util = utilityLands(b.set, counts(deck), all).filter((c) => r(c) >= worst).sort((a, c) => r(c) - r(a));
  const { lands, basics } = manaBase(b.set, counts(deck), all, 40 - deck.length, util);
  deck = deck.concat(lands);
  const side = all.filter(notBasic(b.set));
  for (const c of deck) side.splice(side.indexOf(c), 1);
  Object.assign(b, { deck, side, basics, comboKey: null });
  renderBuilder(b);
}

// ---------------------------------------------------------------- best deck by colours

// The deck model's best deck for each colour combination (deck.js deckPerColors), built a combination at a time in the
// background and listed by predicted win rate; clicking one loads it. Above the list, pickers build any main colours
// with or without a splash.
const comboKey = (want) => want.main + (want.splash ? `+${want.splash}` : '');
const COMBOS_SHOWN = 8;

// A built deck's colours as a target: main colours (four cards or more) and the rest as its splash.
function wantOf(b, deck) {
  const uses = {};
  for (const [id, n] of deck.counts) for (const k of Object.keys(b.set.cards[id].pips || {})) uses[k] = (uses[k] || 0) + n;
  const cols = [...'WUBRG'].filter((k) => uses[k]);
  const main = cols.filter((k) => uses[k] >= 4).join('');
  return { main, splash: cols.filter((k) => !main.includes(k)).join('') };
}

function setupCombos(b) {
  const q = (sel) => b.root.querySelector(sel);
  q('.combos').hidden = false;
  b.combos = [];
  b.pick = new Set();
  q('.combo-cols').replaceChildren(...Object.entries(COLORS).map(([k, name]) => {
    const btn = document.createElement('button');
    btn.className = 'ghost color-pick';
    btn.title = name;
    btn.innerHTML = `<i style="background:var(--${k})"></i>${name}`;
    btn.onclick = () => {
      b.pick.has(k) ? b.pick.delete(k) : b.pick.add(k);
      btn.classList.toggle('on', b.pick.has(k));
      q('.combo-build').disabled = !b.pick.size;
    };
    return btn;
  }));
  q('.combo-build').disabled = true;
  q('.combo-build').onclick = () => {
    const want = { main: [...'WUBRG'].filter((k) => b.pick.has(k)).join(''), splash: q('.combo-splash').value };
    if (want.splash && want.splash !== '*' && want.main.includes(want.splash)) want.splash = '';
    const key = comboKey(want);
    let row = b.combos.find((r) => comboKey(r.want) === key);
    if (!row) {
      row = { want, deck: buildDeck(b.model, b.pool, { kicks: 3, rng: seededRng(`${b.seed}:${key}`), want }) };
      b.combos.push(row);
    }
    if (!row.deck) {
      status(`No ${describeWant(want)} deck in this pool: too few playables there, or not enough fixing`);
      return renderCombos(b);
    }
    loadCombo(b, row);
  };
  fillCombos(b);
}

// Each combination in turn, yielding to the page between them.
async function fillCombos(b) {
  const gen = deckPerColors(b.model, b.pool, (key) => seededRng(`${b.seed}:${key}`));
  for (;;) {
    await new Promise((r) => setTimeout(r, 0));
    if (b.root.builder !== b) return; // a new sealed pool replaced this one
    const { value, done } = gen.next();
    if (done) break;
    if (!b.combos.some((r) => comboKey(r.want) === comboKey(value.want))) b.combos.push(value);
    renderCombos(b);
  }
  b.combosDone = true;
  renderCombos(b);
}

// The colours a deck's spells use beyond its main ones (a splash).
function splashOf(b, deck, main) {
  const out = new Set();
  for (const [id] of deck.counts) for (const k of Object.keys(b.set.cards[id].pips || {})) if (!main.includes(k)) out.add(k);
  return [...out].join('');
}

const describeWant = (want) => [...want.main].map((k) => COLORS[k]).join('-') +
  (want.splash === '*' ? ' with a splash' : want.splash ? ` with a ${COLORS[want.splash].toLowerCase()} splash` : '');

function loadCombo(b, row) {
  setDeck(b, row.deck.counts);
  b.comboKey = row.overall ? 'best' : comboKey(row.want);
  renderBuilder(b);
  status(`Deck model: ${describeWant(row.want)}, ${(100 * row.deck.winRate).toFixed(1)}% predicted win rate`);
}

const dots = (cols) => [...cols].map((k) => `<i style="background:var(--${k})" title="${COLORS[k]}"></i>`).join('');

function renderCombos(b) {
  if (!b.combos || b.root.builder !== b) return;
  const q = (sel) => b.root.querySelector(sel);
  const made = b.combos.filter((r) => r.deck).concat(b.bestRow ? [b.bestRow] : [])
    .sort((x, y) => y.deck.winRate - x.deck.winRate);
  const top = made[0]?.deck.winRate;
  const isOn = (row) => (row.overall ? 'best' : comboKey(row.want)) === b.comboKey;
  // the top few, plus the deck on screen; all of them on request
  const shown = b.allCombos ? made : made.filter((row, i) => i < COMBOS_SHOWN || isOn(row));
  q('.combo-table tbody').replaceChildren(...shown.map((row) => {
    const tr = document.createElement('tr');
    const splash = row.overall ? row.want.splash : row.want.splash ? splashOf(b, row.deck, row.want.main) : '';
    tr.className = isOn(row) ? 'on' : '';
    tr.title = row.overall ? 'Load the deck Build best deck found (any colours)' : `Load the best ${describeWant(row.want)} deck`;
    tr.innerHTML = `<td><span class="dots">${dots(row.want.main)}</span>${[...row.want.main].map((k) => COLORS[k]).join('-')}` +
      `${row.overall ? ' <span class="muted">(Build best deck)</span>' : ''}</td>` +
      `<td>${splash ? `<span class="dots">${dots(splash)}</span>${[...splash].map((k) => COLORS[k]).join(', ')}` : '—'}</td>` +
      `<td class="num">${(100 * row.deck.winRate).toFixed(1)}%</td>` +
      `<td class="num muted">${row.deck.winRate === top ? 'best' : `−${(100 * (top - row.deck.winRate)).toFixed(1)}`}</td>`;
    tr.onclick = () => loadCombo(b, row);
    return tr;
  }));
  const missing = b.combos.length - made.length + (b.bestRow ? 1 : 0);
  const note = q('.combo-note');
  note.textContent = (b.combosDone ? '' : `Building… ${b.combos.length} combinations so far. `) +
    (missing ? `${missing} combination${missing > 1 ? 's' : ''} had no deck (too few playables, or not enough fixing for three colours). ` : '');
  if (made.length > COMBOS_SHOWN) {
    const more = document.createElement('button');
    more.className = 'ghost';
    more.textContent = b.allCombos ? 'Show fewer' : `Show all ${made.length}`;
    more.onclick = () => { b.allCombos = !b.allCombos; renderCombos(b); };
    note.append(more);
  }
}

function moveCard(b, id, from, to) {
  b[from].splice(b[from].indexOf(id), 1);
  b[to].push(id);
  b.comboKey = null;
  renderBuilder(b);
}

const pts = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;

function renderBuilder(b) {
  if (!b || b.root.builder !== b || b.root.closest('[hidden]')) return;
  set = b.set;
  const q = (sel) => b.root.querySelector(sel);
  const n = Object.values(b.basics).reduce((a, c) => a + c, 0), size = b.deck.length + n;
  const lands = n + b.deck.filter((c) => isLand(set.cards[c])).length;
  q('.deck-title').textContent = `Deck: ${size} cards (${size - lands} spells, ${lands} lands)`;
  q('.deck-colors').replaceChildren(...colorChips(b.deck));
  q('.side-title').textContent = `Sideboard (${b.side.length})`;
  const label = q('.b-labels').value;
  b.values = null;
  if (b.model && size) {
    const counts = builderCounts(b), z = b.model.score(counts);
    q('.deck-score').textContent = `Deck model: ${(100 * b.model.winRate(z)).toFixed(1)}% predicted game win rate · ` +
      `a typical 17Lands ${b.event === 'sealed' ? 'Sealed' : 'Premier Draft'} deck: ${(100 * b.model.typical).toFixed(1)}%` +
      (size === 40 ? '' : ' · the model knows 40-card decks best');
    if (label === 'value') b.values = cardValues(b.model, counts, b.pool);
  } else {
    q('.deck-score').textContent = b.model ? '' : `No deck model for ${set.code}: PickNet build only.`;
  }
  const rating = (c) => b.ratings?.get(c) ?? -1e9;
  const byRating = (x, y) => rating(y) - rating(x);
  const deco = (from, to) => (el, id) => {
    const k = basicKey(set.cards[id]);
    if (k) { // a basic in the deck: − and + instead of moving to the sideboard
      el.title = `${BASICS[k]} · click or − for one fewer, + for one more`;
      const ctl = document.createElement('span');
      ctl.className = 'basic-ctl';
      for (const [sign, d] of [['−', -1], ['+', 1]]) {
        const btn = document.createElement('button');
        btn.textContent = sign;
        btn.title = `One ${d < 0 ? 'fewer' : 'more'} ${BASICS[k]}`;
        btn.onclick = (e) => { e.stopPropagation(); b.basics[k] = Math.max(0, b.basics[k] + d); renderBuilder(b); };
        ctl.append(btn);
      }
      el.append(ctl);
      el.onclick = () => { b.basics[k] -= 1; renderBuilder(b); };
      return;
    }
    const tag = document.createElement('span');
    tag.className = 'rate';
    const name = set.cards[id].name, v = b.values?.get(id);
    if (label === 'value' && v !== undefined && v !== null) {
      tag.textContent = pts(v);
      tag.classList.toggle('neg', v < 0);
      const land = isLand(set.cards[id]);
      el.title = from === 'deck'
        ? `${name} · the deck loses ${v.toFixed(1)} win-rate points if it is swapped for ` +
          (land ? 'a basic' : 'the best sideboard spell (lands rebuilt to match)')
        : `${name} · swapped in for ${land ? 'a basic' : 'the deck card it best replaces'}, the deck ` +
          `${v >= 0 ? 'gains' : 'loses'} ${Math.abs(v).toFixed(1)} win-rate points`;
    } else if (label === 'rank' && b.rank) {
      tag.textContent = `#${b.rank.get(id)}`;
      el.title = `${name} · PickNet's #${b.rank.get(id)} of ${b.rank.size} in this pool`;
    } else {
      tag.hidden = true;
      el.title = name;
    }
    el.append(tag);
    el.onclick = () => moveCard(b, id, from, to);
  };
  const basicIds = Object.keys(BASICS).flatMap((k) => Array(basicId(set, k) === undefined ? 0 : b.basics[k]).fill(basicId(set, k)));
  q('.deck').replaceChildren(...pileCols(b.deck.concat(basicIds), false, deco('deck', 'side'), byRating, true));
  q('.side').replaceChildren(...pileCols(b.side, false, deco('side', 'deck'), byRating, true));
  // adding a basic of a colour the deck has none of yet; the deck's basics have their own − and +
  const add = document.createElement('span');
  add.textContent = 'Add a basic:';
  q('.basics').replaceChildren(add, ...Object.entries(BASICS).map(([k, name]) => {
    const btn = document.createElement('button');
    btn.className = 'ghost';
    btn.title = `One more ${name}`;
    const dot = document.createElement('i');
    dot.style.background = `var(--${k})`;
    btn.append(dot, name);
    btn.onclick = () => { b.basics[k] += 1; renderBuilder(b); };
    return btn;
  }));
  for (const sel of ['.deck', '.side']) q(sel).classList.toggle('show-rate', label !== 'none');
  renderCombos(b);
}

function arenaList(ids) {
  const n = new Map();
  for (const id of ids) n.set(id, (n.get(id) || 0) + 1);
  return [...n].map(([id, k]) => `${k} ${set.cards[id].name}`);
}

async function copyBuilder(b) {
  set = b.set;
  const basics = Object.entries(b.basics).filter(([, v]) => v).map(([k, v]) => `${v} ${BASICS[k]}`);
  const text = ['Deck', ...arenaList(b.deck), ...basics, '', 'Sideboard', ...arenaList(b.side)].join('\n');
  const btn = b.root.querySelector('.b-copy');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied!';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = 'Copy deck'; }, 1500);
}

// ---------------------------------------------------------------- sealed

async function openSealed() {
  $('#seal-open').disabled = true;
  try {
    const s = await loadSet($('#set').value);
    const seed = takeSeed(), rng = seededRng(seed);
    const pool = Array.from({ length: SEALED_PACKS }, () => openPack(s, rng)).flat();
    sealed = await makeBuilder(s, pool, 'sealed', $('#sealed-view'), seed);
    if (mode === 'sealed') setMode('sealed');
    if (sealed.model) buildBest(sealed);
    else sealed.ratingsReady.then(() => buildPickNet(sealed));
    showSeed();
    status(`Sealed pool: ${pool.length} cards from ${SEALED_PACKS} ${s.code} packs, seed ${seed}`);
  } finally {
    $('#seal-open').disabled = false;
  }
}

// ---------------------------------------------------------------- draft decks

// The deck builder under a finished draft follows the seat tabs; each seat's builder is kept once made.
async function showDraftBuild(seatIdx) {
  const root = $('#build-view');
  root.hidden = false;
  $('#build-title').textContent = seatIdx === 0 ? 'Build a deck from your picks' : `Build a deck from Bot ${seatIdx}'s picks`;
  const holder = $('#build-holder');
  let b = draftBuilds[seatIdx];
  if (!b) {
    b = draftBuilds[seatIdx] = await makeBuilder(draftSet, draft.seats[seatIdx].pool, 'draft', document.createElement('div'),
      `${draftSeed}:seat${seatIdx}`);
    if (b.model) buildBest(b);
    else b.ratingsReady.then(() => buildPickNet(b));
  }
  if (poolSeat !== seatIdx || mode !== 'draft') return; // another tab was chosen meanwhile
  holder.replaceChildren(b.root);
  renderBuilder(b);
}

// ---------------------------------------------------------------- live (MTG Arena)

// The Live tab follows your draft in MTG Arena. serve.py reads Arena's log (arena.js turns it into drafts) and its
// card database (grpId -> name; names match the set's cards). For the pack in hand it shows PickNet's pick for you,
// wheel odds (draft.js WheelSim) and each card's first-pick rank in the set; then your picks, PickNet's view of each
// pick, and the deck builder on your pool. Older drafts come from Arena's session logs, and any pick, old or new, can
// be looked at again (live.focus: its slot). The tab only appears when serve.py serves the page.
// live.source picks the logs: 'arena' (Arena's own, followed live), 'archive' (tools/archive_arena_logs.sh's copies)
// or 'examples' (the repo's scrubbed logs); those two are read once, with nothing to follow. 'picked' follows one log
// chosen in serve.py's file dialog (live.picked: {file, name, path}).
const live = { source: 'arena', picked: null, drafts: new ArenaDrafts(), files: {}, names: new Map(), codes: [], ranks: new Map(),
  gen: 0, view: null, history: [], merged: new Map(), which: 'latest', focus: null, nets: {} };
const SOURCES = ['arena', 'archive', 'examples', 'picked'];
const POLL_MS = 1000;
const normName = (n) => n.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]/g, '');
const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
const EVENT_KINDS = [[/Premier/i, 'Premier Draft'], [/Quick/i, 'Quick Draft'], [/Trad/i, 'Traditional Draft'],
  [/Sealed/i, 'Sealed'], [/Draft/i, 'Draft']];
const eventKind = (event) => EVENT_KINDS.find(([re]) => re.test(event || ''))?.[1] ?? event ?? 'Draft';

async function api(path) {
  const r = await fetch(`api/arena/${path}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// One log file's new entries -> events into live.drafts; true if there were any. The last entry comes again each
// time (it may still be growing), so entries already read are skipped by their offset.
async function readLog(file) {
  const drafts = live.drafts, f = live.files[file] ??= { id: '', from: 0, seen: new Set() };
  const r = await api(`log?source=${live.source}&file=${encodeURIComponent(file)}&from=${f.from}&id=${encodeURIComponent(f.id)}`);
  if (r.reset) f.seen.clear();
  Object.assign(f, { id: r.id, from: r.next });
  let changed = false;
  for (const e of r.entries) {
    if (f.seen.has(e.at)) continue;
    const evs = parseEntry(e.text);
    if (evs === null) continue; // not all written yet
    f.seen.add(e.at);
    for (const ev of evs) { drafts.add(ev); changed = true; }
  }
  return changed;
}

// Every draft in the source's old logs (Arena's session logs, or the archived or example ones), read once. Each log
// gets its own tracker: drafts don't carry over between sessions, except one Arena restarted in the middle of, which
// mergeDrafts puts back together.
async function loadHistory() {
  const { source, history } = live, q = `source=${source}`;
  for (const f of await api(`history?${q}`)) {
    const r = await api(`log?${q}&file=${encodeURIComponent(f.name)}&from=0&id=`), a = new ArenaDrafts();
    for (const e of r.entries) for (const ev of parseEntry(e.text) || []) a.add(ev);
    for (const d of a.drafts) history.push(Object.assign(d, { source: f.name }));
  }
}

// Start over on another source's logs.
function setSource(source) {
  clearTimeout(live.timer);
  pref.set('live-source', source);
  $('#live-source').value = source;
  Object.assign(live, { source, drafts: new ArenaDrafts(), files: {}, history: [], merged: new Map(), which: 'latest',
    focus: null, view: null, started: false, loaded: false, error: null });
  $('#live-build-holder').replaceChildren();
  showLive();
  pollLive();
}

// A log chosen in serve.py's file dialog, to follow (it may open behind the browser).
async function pickLog() {
  $('#live-source').value = live.source; // until one is chosen
  liveNote('Choose a log in the file dialog (it may have opened behind this window)…');
  let r;
  try {
    r = await api('pick');
  } catch (e) {
    return liveNote(`Couldn't open a file dialog: ${e.message}`, true);
  }
  if (r.cancelled) return showLive();
  setPicked(r);
  setSource('picked');
}

function setPicked(r) {
  live.picked = r;
  const sel = $('#live-source');
  let o = sel.querySelector('option[value="picked"]');
  if (!o) sel.querySelector('option[value="pick"]').before(o = new Option('', 'picked'));
  o.text = `File: ${r.name}`;
  o.title = r.path;
}

// Old logs and this session's as one list of drafts, oldest first.
const allDrafts = () => mergeDrafts([...live.history, ...live.drafts.drafts], live.merged);

// Read the previous session's log once (a draft may have started there), then follow the current one, or the chosen
// log. Archived and example logs, and a chosen .gz, are only read once.
async function pollLive() {
  clearTimeout(live.timer);
  if (live.polling || mode !== 'live') return;
  const picked = live.source === 'picked' ? live.picked : null;
  if (live.source === 'archive' || live.source === 'examples') {
    if (live.started) return;
    live.started = true;
    const source = live.source;
    liveNote('Reading the logs…');
    return loadHistory().then(() => { if (live.source === source) { live.loaded = true; showLive(); } }, (e) => {
      if (live.source === source) liveNote(`Couldn't read the ${source === 'archive' ? 'archived' : 'example'} logs: ${e.message}`, true);
    });
  }
  live.polling = true;
  try {
    if (!live.started && !picked) {
      live.started = true;
      await readLog('prev').catch(() => { /* no previous log */ });
      const source = live.source;
      loadHistory().then(() => live.source === source && showLive(), (e) => status(`Couldn't read Arena's older logs: ${e.message}`));
    }
    const changed = await readLog(picked ? picked.file : 'cur');
    if (picked) for (const d of live.drafts.drafts) d.source ??= picked.name; // its name may date it
    if (changed || !live.started || !live.view) { live.started = live.loaded = true; showLive(); }
    live.error = null;
  } catch (e) {
    if (!live.error) {
      liveNote(!picked ? `Can't read Arena's log: ${e.message}` : / 404/.test(e.message)
        ? `serve.py has lost ${picked.path} (was it restarted?). Choose it again from the Logs menu.` : `Can't read ${picked.path}: ${e.message}`, true);
    }
    live.error = e;
  } finally {
    live.polling = false;
  }
  if (mode === 'live' && !(picked && (picked.name.endsWith('.gz') || / 404/.test(live.error?.message)))) live.timer = setTimeout(pollLive, POLL_MS);
}

function liveNote(text, warn = false) {
  $('#live-note').textContent = text;
  $('#live-note').classList.toggle('warn', warn);
}

// Names for the draft's grpIds not looked up yet (null: not in Arena's card database).
async function nameCards(d) {
  const all = [...d.steps.flatMap((s) => [...s.cards, s.taken]), ...(d.picked || []), ...(d.pool || [])];
  const need = [...new Set(all)].filter((g) => g !== null && !live.names.has(g));
  for (let i = 0; i < need.length; i += 200) {
    const chunk = need.slice(i, i + 200), got = await api(`cards?ids=${chunk.join(',')}`);
    for (const g of chunk) live.names.set(g, got[g] ?? null);
  }
}

// The draft's set: from its event name, or else the set most of its cards come from.
function draftSetCode(d) {
  const code = eventSet(d.event, live.codes);
  if (code) return code;
  const n = {};
  for (const s of d.steps) for (const g of s.cards) { const c = live.names.get(g); if (c && live.codes.includes(c.set)) n[c.set] = (n[c.set] || 0) + 1; }
  return Object.entries(n).sort((a, b) => b[1] - a[1])[0]?.[0];
}

// The draft as PickNet sees it: your seat (every pack shown and pick made, in slots), the pack in hand, your pool.
function liveState(d, s) {
  const byName = s.byName ??= (() => {
    const m = new Map();
    for (const [id, c] of Object.entries(s.cards)) {
      for (const n of [c.name, c.name.split(' // ')[0]]) if (!m.has(normName(n))) m.set(normName(n), +id);
    }
    return m;
  })();
  const toId = (g) => { const c = live.names.get(g); return c ? byName.get(normName(c.name)) : undefined; };
  const known = (gs) => gs.map(toId).filter((x) => x !== undefined);
  d.ui ??= { choices: new Map(), wheel: new Map(), wnet: new Map() };
  const placed = placeSteps(d, s.n_picks), seat = new Seat(s.pad);
  for (const p of placed) {
    seat.show(p.t, known(p.step.cards));
    const c = p.step.taken === null ? undefined : toId(p.step.taken);
    if (c !== undefined) seat.take(p.t, c);
  }
  const last = placed[placed.length - 1];
  const finished = d.done || (!!last && last.round === 2 && last.k === s.n_picks - 1 && last.step.taken !== null);
  const cur = !finished && last && last.step.taken === null && seat.tokens(last.t).length ? last : null;
  // the event's own list of your cards once it has one (a draft whose start the log missed, or sealed), else your picks
  const listed = d.pool ?? (d.picked && d.picked.length > seat.pool.length ? d.picked : null);
  // the pack on screen: the one in hand, or else a pick you're looking at again
  const shown = cur ?? (live.focus?.key === d.key ? placed.find((p) => p.t === live.focus.t) : null) ?? null;
  return { d, s, placed, seat, cur, shown, finished, toId, pool: listed ? known(listed) : seat.pool.slice() };
}

// When a draft happened: the start of its session, from the session log's name ("UTC_Log - 10-13-2026 18.35.07.log",
// in UTC; an archived one "2026-10-14_102834_77dbf69c.log.gz", in local time), or null for one only in this
// session's Player.log (it's happening now).
function draftWhen(d) {
  const m = /(\d+)-(\d+)-(\d+) (\d+)\.(\d+)/.exec(d.source || '');
  if (m) return new Date(Date.UTC(+m[3], m[1] - 1, +m[2], +m[4], +m[5]));
  const a = /^(\d{4})-(\d\d)-(\d\d)_(\d\d)(\d\d)/.exec(d.source || '');
  return a ? new Date(+a[1], a[2] - 1, +a[3], +a[4], +a[5]) : null;
}

// "Tue, Oct 13, 08:35 PM" (your time), or "today".
function draftDate(d) {
  const when = draftWhen(d);
  if (!when) return 'today';
  return when.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function draftLabel(d) {
  const picks = d.steps.filter((s) => s.taken !== null).length, code = eventSet(d.event, live.codes);
  return `${code ? `${code} ` : ''}${eventKind(d.event)} · ${draftDate(d)} · ${d.steps.length ? `${picks} pick${picks === 1 ? '' : 's'}` : 'pool'}`;
}

function renderPicker(list) {
  const sel = $('#live-which'), opts = [['latest', list.length ? `Latest: ${draftLabel(list[list.length - 1])}` : 'Latest'],
    ...[...list].reverse().map((d) => [d.key, draftLabel(d)])];
  const sig = opts.map((o) => o.join('|')).join('\n');
  if (sel.dataset.sig !== sig) {
    sel.dataset.sig = sig;
    sel.replaceChildren(...opts.map(([v, t]) => new Option(t, v)));
  }
  if (!list.some((d) => d.key === live.which)) live.which = 'latest';
  sel.value = live.which;
}

async function showLive() {
  const gen = ++live.gen, stale = () => gen !== live.gen || mode !== 'live';
  const list = allDrafts();
  renderPicker(list);
  const d = list.find((x) => x.key === live.which) ?? list[list.length - 1];
  if (live.focus && live.focus.key !== d?.key) live.focus = null;
  if (!d) {
    live.view = null;
    $('#live-title').textContent = 'Live draft';
    for (const id of ['#live-pick', '#live-pool-view', '#live-build', '#live-log-view']) $(id).hidden = true;
    const st = live.status || {};
    if (live.source === 'picked') {
      if (live.loaded && live.picked) liveNote(`Following ${live.picked.path}. Drafts show up here as Arena logs them.`);
      return;
    }
    if (live.source !== 'arena') {
      if (!live.started) return;
      const dir = live.source === 'archive' ? `${st.archive?.dir ?? '~/mtga-log-archive'} (tools/archive_arena_logs.sh fills it; set MTGA_ARCHIVE_DIR for serve.py if yours is elsewhere)`
        : 'tools/fixtures/arena';
      return live.loaded ? liveNote(`No drafts in ${dir}.`, true) : liveNote('Reading the logs…');
    }
    liveNote(!st.log ? 'No MTG Arena log found. Is Arena installed? (Set MTGA_LOG_DIR for serve.py if it lives somewhere unusual.)'
      : st.detailed === false ? 'Arena\'s detailed logs are off: turn on Options → Account → Detailed Logs (Plugin Support), then restart Arena.'
        : 'Watching Arena\'s log. Start a draft in Arena and it shows up here.', !st.log || st.detailed === false);
    return;
  }
  try {
    await nameCards(d);
  } catch (e) {
    return liveNote(`Can't read Arena's card database: ${e.message}`, true);
  }
  if (stale()) return;
  const code = draftSetCode(d);
  if (!code) {
    $('#live-title').textContent = `${eventKind(d.event)}${d.event ? ` (${d.event})` : ''}`;
    for (const id of ['#live-pick', '#live-pool-view', '#live-build', '#live-log-view']) $(id).hidden = true;
    return liveNote(live.status?.cardDb ? 'PickNet has no data for this set.' :
      'Arena\'s card database wasn\'t found, so the cards can\'t be named. (Set MTGA_CARD_DB for serve.py.)', true);
  }
  const s = await loadSet(code);
  if (stale()) return;
  if (!(code in live.nets)) loadWheelNet(code).then((meta) => { live.nets[code] = meta; if (mode === 'live') showLive(); });
  const v = live.view = liveState(d, s);
  renderLive(v);
  await ready;
  if (stale()) return;
  if (v.shown) {
    await liveChoices(v, v.shown.t);
    if (stale()) return;
    renderLivePack(v);
    await liveWheelNet(v, v.shown);
    if (stale()) return;
    renderLivePack(v);
    if (wheelMode(s) !== 'net') startWheel(v);
  }
  await setRanks(s);
  if (stale()) return;
  renderLivePack(v);
  for (const p of v.placed) {
    if (p.step.taken === null) continue;
    await liveChoices(v, p.t);
    await liveWheelNet(v, p);
    if (stale()) return;
  }
  renderLiveLog(v);
  if (v.finished && v.pool.length && !d.ui.build) liveBuild(v, false);
}

const choiceKey = (t) => `${t}:${$('#skill').value}`;

// PickNet's read of your pack at slot t, cached per draft.
async function liveChoices(v, t) {
  const { d, s, seat } = v, key = choiceKey(t);
  if (!d.ui.choices.has(key) && seat.tokens(t).length) {
    const r = await run(feeds([seat], t, [s.ctx[$('#skill').value]], s.sid));
    d.ui.choices.set(key, choices(seat, t, r.logits, 0));
  }
  return d.ui.choices.get(key);
}

// Every card's first-pick rank in its set (PickNet, no context), once per set and drafter profile.
async function setRanks(s) {
  const key = `${s.code}:${$('#skill').value}`;
  if (!live.ranks.has(key)) {
    const ids = Object.keys(s.cards).map(Number).filter(notBasic(s));
    const r = await rateCards(s, ids, s.ctx[$('#skill').value], (f) => run(f).then((x) => x.logits), 4, seededRng(`${s.code}:ranks`));
    live.ranks.set(key, new Map([...r].sort((a, b) => b[1] - a[1]).map(([c], i) => [c, i + 1])));
  }
  return live.ranks.get(key);
}

// The pick the wheel odds assume you make at a step: the one you made, or else PickNet's top choice; and the next
// best, which the odds for that card itself assume instead.
function yourTakes(v, p, ch) {
  const first = (p.step.taken === null ? undefined : v.toId(p.step.taken)) ?? ch[0].card;
  return [first, ch.find((x) => x.card !== first)?.card ?? first];
}

// Which wheel odds to show: the viewer's choice, or the simulation where the set has no WheelNet.
const wheelMode = (s) => (live.nets[s.code] ? $('#live-wheel').value : 'sim');

// WheelNet's draft context: the drafter profile's average (rank and skill, for a model that reads them), with days
// into the format from the draft's date, counted from the set's first draft as in training.
function wheelCtx(meta, d) {
  const c = (meta.ctx[$('#skill').value] ?? meta.ctx.typical).slice();
  if (meta.time) {
    const days = Math.max(0, ((draftWhen(d) ?? new Date()) - meta.time.start) / 864e5);
    c[meta.time.at] = days / 30;
    c[meta.time.at + 1] = Math.log1p(days) / Math.log1p(30);
  }
  return c;
}

// WheelNet's odds for the pack at step p (Map card -> P(wheel)), or null: no WheelNet for the set, or a pack that
// won't come back. Cached per draft. One run of two drafts: you take yourTakes' first card, or its second.
async function liveWheelNet(v, p) {
  const { d, s, seat } = v, key = choiceKey(p.t), ch = d.ui.choices.get(key);
  if (!ch || p.k + SEATS >= s.n_picks) return null;
  const meta = await loadWheelNet(s.code);
  if (!meta) return null;
  if (!d.ui.wnet.has(key)) {
    const takes = yourTakes(v, p, ch), f = wheelFeeds(meta, seat, p.t, takes, wheelCtx(meta, d));
    const out = (await runModel(`wheel:${s.code}`, wheelTensors(meta, f))).out;
    const odds = wheelOdds(meta, f, out, p.t, 0);
    odds.set(takes[0], wheelOdds(meta, f, out, p.t, 1).get(takes[0]) ?? null);
    d.ui.wnet.set(key, odds);
  }
  return d.ui.wnet.get(key);
}

// Wheel odds for the pack in hand, in the background; one simulation at a time per draft (they share its tables).
function startWheel(v) {
  const { d, s, shown: cur, seat } = v, step = cur.step;
  const ch = d.ui.choices.get(choiceKey(cur.t));
  if (cur.k + SEATS >= s.n_picks || d.ui.wheel.has(cur.t) || d.ui.wheelFor === step || !ch) return;
  d.ui.wheelFor = step;
  d.ui.sim ??= new WheelSim(s, (f) => run(f).then((r) => r.logits));
  const sim = d.ui.sim, cancelled = () => live.view?.shown?.step !== step;
  const mine = yourTakes(v, cur, ch);
  let done = 0, total = 0;
  const t0 = performance.now();
  d.ui.simQueue = (d.ui.simQueue ?? Promise.resolve()).then(() => {
    if (cancelled()) return null;
    total = Math.max(0, cur.t + SEATS - 1 - sim.t) + SEATS - 1;
    return sim.odds(seat.seen[cur.t], cur.t, mine, cancelled, () => {
      if (!cancelled() && mode === 'live') status(`Wheel odds: sending the pack round the table… ${++done} / ${total}`);
    });
  }).then((odds) => {
    if (!odds) return;
    d.ui.wheel.set(cur.t, odds);
    if (cancelled()) return;
    status(`Wheel odds: ${sim.rollouts} passes round the table in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    renderLivePack(live.view);
  }).catch((e) => status(`Wheel odds failed: ${e.message}`)).finally(() => {
    if (d.ui.wheelFor === step) d.ui.wheelFor = null;
  });
}

function renderLive(v) {
  const { d, s, cur, shown, finished } = v;
  set = s;
  $('#live-title').textContent = `${s.name} ${eventKind(d.event)}` + (cur ? ` · Pack ${cur.round + 1} · Pick ${cur.k + 1} of ${s.n_picks}`
    : finished || !d.steps.length || d !== allDrafts().at(-1) ? '' : ' · waiting for the next pack');
  const review = shown && !cur;
  $('#live-review').hidden = !review;
  if (review) {
    const took = shown.step.taken === null ? undefined : v.toId(shown.step.taken);
    $('#live-review-title').textContent = `Pack ${shown.round + 1} · Pick ${shown.k + 1}` +
      (took === undefined ? '' : `: you took ${s.cards[took].name}`);
    const at = v.placed.indexOf(shown);
    $('#live-prev').disabled = at <= 0;
    $('#live-next').disabled = at >= v.placed.length - 1;
  }
  const unknown = cur ? cur.step.cards.filter((g) => v.toId(g) === undefined).length : 0;
  liveNote([
    finished ? (d.steps.length ? (d.steps.length < 3 * s.n_picks ? 'Draft complete (the log missed some of it).' : 'Draft complete.') : `Your ${isSealed(d.event) ? 'sealed pool' : 'draft pool'} from Arena.`) : '',
    isBotDraft(d.event) ? 'The other seats here are Arena\'s bots, not people, so PickNet\'s wheel odds are rougher.' : '',
    unknown ? `${unknown} card${unknown > 1 ? 's' : ''} in this pack ${unknown > 1 ? 'aren\'t' : 'isn\'t'} in PickNet's card list.` : '',
  ].filter(Boolean).join(' '));
  $('#live-pick').hidden = !shown;
  renderLivePack(v);
  renderLivePool(v);
  renderLiveLog(v);
  const b = d.ui.build;
  $('#live-build').hidden = !b;
  if (b && $('#live-build-holder').firstChild !== b.b.root) $('#live-build-holder').replaceChildren(b.b.root);
  if (b) renderBuilder(b.b);
}

// The pack in hand: built once per pack (so images don't reload), then PickNet's numbers filled in as they arrive.
function renderLivePack(v) {
  const { d, s, shown: cur, seat } = v;
  if (!cur || mode !== 'live' || live.view !== v) return;
  set = s;
  const pack = $('#live-pack');
  if (live.packStep !== cur.step) {
    live.packStep = cur.step;
    live.packEls = new Map();
    const els = seat.tokens(cur.t).map((tok) => {
      const el = cardEl(tok.card);
      if (tok.copies > 1) {
        const b = document.createElement('span');
        b.className = 'copies';
        b.textContent = `×${tok.copies}`;
        el.append(b);
      }
      const o = document.createElement('div');
      o.className = 'model';
      el.append(o);
      live.packEls.set(tok.card, el);
      return el;
    });
    const unknown = cur.step.cards.filter((g) => v.toId(g) === undefined).map((g) => {
      const el = document.createElement('div');
      el.className = 'card noimg unknown';
      const name = live.names.get(g)?.name ?? `Arena card ${g}`;
      el.innerHTML = '<div class="name"></div><div class="model">Not in PickNet\'s card list</div>';
      el.querySelector('.name').textContent = name;
      el.title = name;
      return el;
    });
    pack.replaceChildren(...els, ...unknown);
    renderTaken(v);
  }
  const ch = d.ui.choices.get(choiceKey(cur.t)) ?? [], sim = d.ui.wheel.get(cur.t), net = d.ui.wnet.get(choiceKey(cur.t));
  const ranks = live.ranks.get(`${s.code}:${$('#skill').value}`), nRanked = ranks?.size;
  const wheels = cur.k + SEATS < s.n_picks, took = cur.step.taken === null ? undefined : v.toId(cur.step.taken);
  const wm = wheelMode(s), odds = wm === 'sim' ? sim : net;
  const fmt = (m, c) => (!m ? '…' : typeof m.get(c) === 'number' ? pct(m.get(c)) : '?');
  // a pick you've made whose pack came back: what did
  const back = cur !== v.cur && wheels ? seat.seen[cur.t + SEATS] : undefined;
  $('#live-wheel').value = wm;
  $('#live-wheel').disabled = !live.nets[s.code];
  $('#live-wheel').title = live.nets[s.code] ? $('#live-wheel').title : `No WheelNet for ${s.code} yet (tools/export_wheelnet.py makes one), so the wheel odds come from the simulation.`;
  const rankOf = new Map(ch.map((x, i) => [x.card, i]));
  for (const [card, el] of live.packEls) {
    const r = rankOf.get(card), o = el.querySelector('.model'), name = s.cards[card].name;
    el.classList.toggle('top1', r === 0);
    el.classList.toggle('selected', card === took);
    const w = odds?.get(card), sr = ranks?.get(card);
    const wheel = !wheels ? '' : wm === 'both' ? `wheel ${fmt(net, card)} / ${fmt(sim, card)}` : `wheels ${fmt(odds, card)}`;
    const came = back && card !== took ? back.includes(card) : undefined;
    o.innerHTML = (r === undefined ? 'PickNet …' : `#${r + 1} · ${pct(ch[r].p)}<div class="meter"><i style="width:${(ch[r].p * 100).toFixed(1)}%"></i></div>`) +
      `<div class="sub"><span class="wheel${w >= 0.5 ? ' likely' : ''}">${wheel}</span><span>${sr ? `set #${sr}` : ''}</span></div>` +
      (came === undefined ? '' : `<div class="sub"><span class="${came ? 'back' : 'gone'}">${came ? '✓ came back' : '✗ taken'}</span></div>`);
    const said = (m, who) => (wheels && typeof m?.get(card) === 'number' ? `${who}: comes back ${(m.get(card) * 100).toFixed(0)}% of the time if you pass it` : '');
    el.title = [card === took ? `${name} (you took it)` : name, r === undefined ? '' : `PickNet's #${r + 1} for you (${(ch[r].p * 100).toFixed(1)}%)`,
      said(net, 'WheelNet'), said(sim, 'simulation'), came === undefined ? '' : came ? 'it came back' : 'it was taken',
      sr ? `#${sr} of ${nRanked} in ${s.code} as a first pick` : ''].filter(Boolean).join(' · ');
  }
  for (const x of ch) pack.append(live.packEls.get(x.card)); // best first
  for (const el of pack.querySelectorAll('.unknown')) pack.append(el);
}

// On a pack's second lap: what the other seven took from it since you saw it.
function renderTaken(v) {
  const { shown: cur, seat } = v, before = cur.k >= SEATS ? seat.seen[cur.t - SEATS] : null;
  $('#live-taken').hidden = !before;
  if (!before) return;
  const gone = before.slice();
  for (const c of [seat.pick[cur.t - SEATS], ...seat.seen[cur.t]]) { const i = gone.indexOf(c); if (i >= 0) gone.splice(i, 1); }
  $('#live-taken .pool').replaceChildren(...pileCols(gone, true));
}

function renderLivePool(v) {
  $('#live-pool-view').hidden = !v.pool.length;
  if (!v.pool.length) return;
  set = v.s;
  $('#live-pool-title').textContent = `Your ${v.d.steps.length ? 'picks' : 'pool'} (${v.pool.length})`;
  $('#live-pool-colors').replaceChildren(...colorChips(v.pool));
  $('#live-pool').replaceChildren(...pileCols(v.pool));
}

function renderLiveLog(v) {
  const { d, s, placed, toId } = v;
  const rows = placed.filter((p) => p.step.taken !== null && toId(p.step.taken) !== undefined);
  $('#live-log-view').hidden = !rows.length;
  if (!rows.length) return;
  const esc = (x) => x.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  let agree = 0, scored = 0;
  const calls = { back: [], gone: [] }; // WheelNet's odds for the cards you passed, by what happened to them
  const html = rows.map((p) => {
    const card = toId(p.step.taken), ch = d.ui.choices.get(choiceKey(p.t));
    const r = ch ? ch.findIndex((x) => x.card === card) : -1;
    if (ch && r >= 0) { scored++; if (r === 0) agree++; }
    // the pack 8 picks later: how many came back, and the best of them by PickNet's read of this pick
    const back = p.k + SEATS < s.n_picks ? v.seat.seen[p.t + SEATS] : undefined, net = d.ui.wnet.get(choiceKey(p.t));
    let came = '';
    if (back && ch) {
      const best = ch.find((x) => x.card !== card && back.includes(x.card));
      came = `${back.length} back${best ? ` · best ${esc(s.cards[best.card].name)}${net?.get(best.card) == null ? '' : ` (WheelNet ${pct(net.get(best.card))})`}` : ''}`;
      if (net) for (const [c, q] of net) if (c !== card && q !== null) calls[back.includes(c) ? 'back' : 'gone'].push(q);
    }
    const cells = [`P${p.round + 1}p${p.k + 1}`, esc(s.cards[card].name),
      !ch ? '…' : r < 0 ? '—' : `#${r + 1} (${(ch[r].p * 100).toFixed(0)}%)`,
      !ch ? '…' : r === 0 ? '—' : `${esc(s.cards[ch[0].card].name)} (${(ch[0].p * 100).toFixed(0)}%)`, came];
    const cls = [r === 0 ? 'agree' : '', v.shown === p ? 'focus' : ''].join(' ').trim();
    return `<tr data-t="${p.t}" class="${cls}">${cells.map((c) => `<td>${c}</td>`).join('')}</tr>`;
  }).join('');
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const graded = calls.back.length && calls.gone.length ? ` WheelNet gave the cards you passed that came back ${pct(mean(calls.back))} on average, ` +
    `and the ones that were taken ${pct(mean(calls.gone))} (${calls.back.length + calls.gone.length} cards).` : '';
  $('#live-log').innerHTML = `<caption>You took PickNet's top choice on ${agree} of ${scored} picks.${graded}</caption>` +
    '<tr><th>Pick</th><th>You took</th><th>PickNet\'s rank</th><th>PickNet\'s top choice</th><th>Came back</th></tr>' + html;
  for (const tr of $('#live-log').querySelectorAll('tr[data-t]')) tr.onclick = () => liveFocus(+tr.dataset.t, true);
}

// Look again at the pick in slot t of the draft on screen (null: back to the picks). The pack in hand, if there is
// one, stays on screen until you've picked from it.
function liveFocus(t, scroll = false) {
  const v = live.view;
  if (!v) return;
  live.focus = t === null ? null : { key: v.d.key, t };
  live.view = Object.assign({}, v, { shown: v.cur ?? (t === null ? null : v.placed.find((p) => p.t === t)) });
  renderLive(live.view);
  showLive();
  if (scroll && live.view.shown) $('#live-view').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function liveStep(dir) {
  const v = live.view;
  if (!v?.shown || v.cur) return;
  const p = v.placed[v.placed.indexOf(v.shown) + dir];
  if (p) liveFocus(p.t);
}

// The deck builder on your pool, made again when the pool has changed since. scroll: bring it into view.
async function liveBuild(v, scroll = true) {
  const { d, s } = v, key = v.pool.join(',');
  if (!v.pool.length) return;
  $('#live-build').hidden = false;
  if (d.ui.build?.key === key) {
    $('#live-build-holder').replaceChildren(d.ui.build.b.root);
    return renderBuilder(d.ui.build.b);
  }
  const root = document.createElement('div');
  $('#live-build-holder').replaceChildren(root);
  const b = await makeBuilder(s, v.pool, isSealed(d.event) ? 'sealed' : 'draft', root, `live:${d.draftId ?? d.event}:${v.pool.length}`);
  d.ui.build = { key, b };
  if (b.model) buildBest(b);
  else b.ratingsReady.then(() => buildPickNet(b));
  if (scroll) $('#live-build').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------- modes

const DRAFT_VIEWS = ['#pick-view', '#god-view', '#end-view', '#build-view', '#pool-view', '#log-view', '#intro'];

function setMode(m) {
  mode = m;
  pref.set('mode', m);
  document.body.dataset.mode = m;
  for (const b of $('#mode').children) b.classList.toggle('on', b.dataset.mode === m);
  showSeed();
  $('#live-view').hidden = m !== 'live';
  if (m === 'live') {
    for (const v of DRAFT_VIEWS) $(v).hidden = true;
    $('#sealed-view').hidden = $('#seal-intro').hidden = true;
    if (live.view) renderLive(live.view);
    pollLive();
    return;
  }
  if (m === 'sealed') {
    for (const v of DRAFT_VIEWS) $(v).hidden = true;
    $('#sealed-view').hidden = !sealed;
    $('#seal-intro').hidden = !!sealed;
    if (sealed) renderBuilder(sealed);
    return;
  }
  $('#sealed-view').hidden = $('#seal-intro').hidden = true;
  if (!draft) { $('#intro').hidden = false; return; }
  set = draftSet;
  if (draft.done) return finish();
  $('#pick-view').hidden = $('#pool-view').hidden = false;
  renderPack();
  renderGod();
  renderPool(0);
  step.done.then((s) => { if (s === step && mode === 'draft') { renderPack(); renderGod(); } });
}

// ---------------------------------------------------------------- controls

async function init() {
  const sets = await (await fetch('data/sets.json')).json();
  // the newest set first, then the rest by name
  const FIRST = 'FRA';
  sets.sort((a, b) => (b.code === FIRST) - (a.code === FIRST) || a.name.localeCompare(b.name));
  $('#set').replaceChildren(...sets.map((s) => new Option(`${s.name} (${s.code})`, s.code)));
  live.codes = sets.map((s) => s.code);
  for (const k of ['set', 'style', 'skill']) {
    const v = pref.get(k, null);
    if (v !== null && [...$(`#${k}`).options].some((o) => o.value === v)) $(`#${k}`).value = v;
    $(`#${k}`).addEventListener('change', () => pref.set(k, $(`#${k}`).value));
  }
  $('#show').checked = pref.get('show', '0') === '1';
  $('#show').addEventListener('change', () => { pref.set('show', $('#show').checked ? '1' : '0'); if (draft && !draft.done) renderPack(); });
  $('#new').onclick = newDraft;
  $('#god').checked = pref.get('god', '0') === '1';
  $('#god').addEventListener('change', () => { pref.set('god', $('#god').checked ? '1' : '0'); if (draft && !draft.done) renderGod(); });
  $('#confirm').onclick = () => selected !== null && commit(selected);
  $('#auto').onclick = () => commit(null);
  $('#rest').onclick = autodraftRest;
  $('#undo').onclick = undo;
  $('#copy-deck').onclick = copyDeck;
  $('#seal-open').onclick = openSealed;
  for (const b of $('#mode').children) b.onclick = () => setMode(b.dataset.mode);
  const wanted = pref.get('mode', 'draft');
  setMode(wanted === 'sealed' ? 'sealed' : 'draft');
  // the Live tab, when serve.py is serving the page and can read Arena's log
  api('status').then((st) => {
    live.status = st;
    $('#live-tab').hidden = false;
    for (const o of $('#live-source').options) {
      const n = st[o.value]?.logs;
      if (n !== undefined) o.text += ` (${n} log${n === 1 ? '' : 's'})`;
    }
    if (st.picked) setPicked(st.picked);
    const src = pref.get('live-source', 'arena');
    live.source = SOURCES.includes(src) && (src !== 'picked' || live.picked) ? src : 'arena';
    $('#live-source').value = live.source;
    if (wanted === 'live') setMode('live');
  }).catch(() => { /* a static host: no Live tab */ });
  $('#skill').addEventListener('change', () => { if (mode === 'live') showLive(); });
  $('#live-build-btn').onclick = () => live.view && liveBuild(live.view);
  $('#live-source').onchange = () => ($('#live-source').value === 'pick' ? pickLog() : setSource($('#live-source').value));
  $('#live-which').onchange = () => { live.which = $('#live-which').value; live.focus = null; showLive(); };
  $('#live-prev').onclick = () => liveStep(-1);
  $('#live-next').onclick = () => liveStep(1);
  $('#live-close').onclick = () => liveFocus(null);
  $('#live-replay').onclick = () => live.view?.placed.length && liveFocus(live.view.placed[0].t, true);
  $('#live-wheel').value = pref.get('wheel', 'net');
  $('#live-wheel').onchange = () => {
    pref.set('wheel', $('#live-wheel').value);
    if (live.view?.shown) { renderLivePack(live.view); if (wheelMode(live.view.s) !== 'net') startWheel(live.view); }
  };
  $('#export-csv').onclick = exportCsv;
  $('#export-jsonl').onclick = exportJsonl;
  document.addEventListener('keydown', (e) => {
    if (e.target.closest('select, input, textarea')) return;
    if (mode === 'live') {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); liveStep(e.key === 'ArrowLeft' ? -1 : 1); }
      else if (e.key === 'Escape') liveFocus(null);
      return;
    }
    if (mode !== 'draft') return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
      e.preventDefault();
      undo();
    } else if (e.key === 'Enter' && selected !== null && draft && !draft.done) {
      e.preventDefault();
      commit(selected);
    } else if (e.key === 'Escape' && selected !== null) {
      selected = null;
      renderPack();
    }
  });
  $('#seal-open').disabled = false;
  await ready;
  $('#new').disabled = false;
}

init();
