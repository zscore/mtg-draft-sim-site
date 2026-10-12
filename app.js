import { Draft, feeds, choices, sample, rateCards, SEATS, SLOTS } from './draft.js';
import { openPack } from './packs.js';

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
  } else if (m.type === 'error') {
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

// ---------------------------------------------------------------- state

// history: every step's picks (seat 0 = you), so undo can replay the draft from its opening packs.
let model, set, draft, openers, history, step, log, selected = null, busy = false, autodrafting = false;
let draftId, draftTime, poolSeat = 0;
// set is whichever set is on screen: the draft's (draftSet) or the sealed pool's (sealed.set).
let mode = 'draft', draftSet, sealed;
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
    s.botPicks = s.choices.map((ch, i) => (i === 0 ? null : sample(ch, temp).card));
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
  draft = new Draft(set);
  openers = draft.openers.map((p) => p.slice());
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
  $('#end-view').hidden = $('#log-view').hidden = true;
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
// a card id comparator). decorate(el, id) can add to each card.
function pileCols(ids, small = false, decorate = null, order = null) {
  const cols = new Map([['Lands', []], ['0–1', []], ['2', []], ['3', []], ['4', []], ['5', []], ['6+', []]]);
  for (const id of ids) {
    const c = set.cards[id];
    const key = /Land/.test(c.type) && !/Creature/.test(c.type) ? 'Lands' : c.cmc <= 1 ? '0–1' : c.cmc >= 6 ? '6+' : String(c.cmc);
    cols.get(key).push(id);
  }
  return [...cols].filter(([, v]) => v.length).map(([k, v]) => {
    const col = document.createElement('div');
    col.className = 'col';
    const h = document.createElement('h3');
    h.textContent = `${k} (${v.length})`;
    v.sort(order || ((a, b) => colorKey(set.cards[a]) - colorKey(set.cards[b])));
    col.append(h, ...v.map((id) => {
      const el = cardEl(id, small);
      decorate?.(el, id);
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
    b.onclick = () => { renderPool(i); [...tabs.children].forEach((x, j) => x.classList.toggle('on', j === i)); };
    if (i === 0) b.classList.add('on');
    return b;
  }));
  renderPool(0);
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

// ---------------------------------------------------------------- sealed

const SEALED_PACKS = 6, BASICS = { W: 'Plains', U: 'Island', B: 'Swamp', R: 'Mountain', G: 'Forest' };
const isLand = (c) => /Land/.test(c.type) && !/Creature/.test(c.type);

async function openSealed() {
  $('#seal-open').disabled = true;
  try {
    const s = await loadSet($('#set').value);
    const pool = Array.from({ length: SEALED_PACKS }, () => openPack(s)).flat();
    status(`Opened ${SEALED_PACKS} ${s.code} packs · waiting for PickNet to rate the pool…`);
    await ready;
    const ratings = await rateCards(s, [...new Set(pool)], s.ctx[$('#skill').value], (f) => run(f).then((r) => r.logits));
    const rank = new Map([...ratings].sort((a, b) => b[1] - a[1]).map(([c], i) => [c, i + 1]));
    sealed = { set: s, deck: [], side: pool, basics: { W: 0, U: 0, B: 0, R: 0, G: 0 }, ratings, rank };
    autoBuild();
    if (mode === 'sealed') setMode('sealed');
    status(`Sealed pool: ${pool.length} cards from ${SEALED_PACKS} ${s.code} packs, rated by PickNet`);
  } finally {
    $('#seal-open').disabled = false;
  }
}

// Best two colours by the sum of PickNet's top 23 nonland ratings there (colourless cards count for
// every pair), then those 23 and 17 basics split by the deck's coloured cards.
function autoBuild() {
  const all = [...sealed.deck, ...sealed.side], r = (c) => sealed.ratings.get(c);
  let best = null;
  for (let i = 0; i < 5; i++) {
    for (let j = i + 1; j < 5; j++) {
      const pair = 'WUBRG'[i] + 'WUBRG'[j];
      const fits = all.filter((c) => !isLand(sealed.set.cards[c]) && [...sealed.set.cards[c].colors].every((x) => pair.includes(x)))
        .sort((a, b) => r(b) - r(a)).slice(0, 23);
      const score = fits.reduce((a, c) => a + r(c), 0) - 20 * (23 - fits.length);
      if (!best || score > best.score) best = { pair, fits, score };
    }
  }
  const deck = best.fits, side = all.slice();
  for (const c of deck) side.splice(side.indexOf(c), 1);
  const pips = { W: 0, U: 0, B: 0, R: 0, G: 0 };
  for (const c of deck) for (const x of sealed.set.cards[c].colors) pips[x] += 1;
  const total = Object.values(pips).reduce((a, b) => a + b, 0) || 1, n = 40 - deck.length;
  const basics = Object.fromEntries(Object.entries(pips).map(([k, v]) => [k, Math.floor((n * v) / total)]));
  const left = n - Object.values(basics).reduce((a, b) => a + b, 0);
  Object.keys(pips).sort((a, b) => ((n * pips[b]) / total) % 1 - ((n * pips[a]) / total) % 1).slice(0, left).forEach((k) => { basics[k] += 1; });
  Object.assign(sealed, { deck, side, basics });
  renderSealed();
}

function moveCard(id, from, to) {
  sealed[from].splice(sealed[from].indexOf(id), 1);
  sealed[to].push(id);
  renderSealed();
}

function renderSealed() {
  if (!sealed || mode !== 'sealed') return;
  set = sealed.set;
  const n = Object.values(sealed.basics).reduce((a, b) => a + b, 0);
  const lands = n + sealed.deck.filter((c) => isLand(set.cards[c])).length;
  $('#deck-title').textContent = `Deck: ${sealed.deck.length + n} cards (${sealed.deck.length + n - lands} spells, ${lands} lands)`;
  $('#deck-colors').replaceChildren(...colorChips(sealed.deck));
  $('#side-title').textContent = `Sideboard (${sealed.side.length})`;
  const byRating = (a, b) => sealed.ratings.get(b) - sealed.ratings.get(a);
  const deco = (from, to) => (el, id) => {
    const b = document.createElement('span');
    b.className = 'rate';
    b.textContent = `#${sealed.rank.get(id)}`;
    el.title = `${set.cards[id].name} · PickNet's #${sealed.rank.get(id)} of ${sealed.rank.size} in this pool`;
    el.append(b);
    el.onclick = () => moveCard(id, from, to);
  };
  $('#deck').replaceChildren(...pileCols(sealed.deck, false, deco('deck', 'side'), byRating));
  $('#side').replaceChildren(...pileCols(sealed.side, false, deco('side', 'deck'), byRating));
  $('#basics').replaceChildren(...Object.entries(BASICS).map(([k, name]) => {
    const el = document.createElement('span');
    el.className = 'land';
    const dot = document.createElement('i');
    dot.style.background = `var(--${k})`;
    const minus = document.createElement('button');
    minus.className = 'ghost';
    minus.textContent = '−';
    minus.title = `One fewer ${name}`;
    minus.disabled = !sealed.basics[k];
    minus.onclick = () => { sealed.basics[k] -= 1; renderSealed(); };
    const v = document.createElement('span');
    v.className = 'n';
    v.textContent = sealed.basics[k];
    const plus = document.createElement('button');
    plus.className = 'ghost';
    plus.textContent = '+';
    plus.title = `One more ${name}`;
    plus.onclick = () => { sealed.basics[k] += 1; renderSealed(); };
    el.append(dot, name, minus, v, plus);
    return el;
  }));
  for (const id of ['#deck', '#side']) $(id).classList.toggle('show-rate', $('#seal-show').checked);
}

function arenaList(ids) {
  const n = new Map();
  for (const id of ids) n.set(id, (n.get(id) || 0) + 1);
  return [...n].map(([id, k]) => `${k} ${set.cards[id].name}`);
}

async function copySealed() {
  const basics = Object.entries(sealed.basics).filter(([, v]) => v).map(([k, v]) => `${v} ${BASICS[k]}`);
  const text = ['Deck', ...arenaList(sealed.deck), ...basics, '', 'Sideboard', ...arenaList(sealed.side)].join('\n');
  const btn = $('#seal-copy');
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copied!';
  } catch {
    btn.textContent = 'Copy failed';
  }
  setTimeout(() => { btn.textContent = 'Copy deck'; }, 1500);
}

// ---------------------------------------------------------------- modes

const DRAFT_VIEWS = ['#pick-view', '#god-view', '#end-view', '#pool-view', '#log-view', '#intro'];

function setMode(m) {
  mode = m;
  pref.set('mode', m);
  document.body.dataset.mode = m;
  for (const b of $('#mode').children) b.classList.toggle('on', b.dataset.mode === m);
  if (m === 'sealed') {
    for (const v of DRAFT_VIEWS) $(v).hidden = true;
    $('#sealed-view').hidden = !sealed;
    $('#seal-intro').hidden = !!sealed;
    if (sealed) { set = sealed.set; renderSealed(); }
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
  $('#set').replaceChildren(...sets.map((s) => new Option(`${s.name} (${s.code})`, s.code)));
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
  $('#seal-auto').onclick = () => sealed && autoBuild();
  $('#seal-clear').onclick = () => {
    if (!sealed) return;
    sealed.side.push(...sealed.deck.splice(0));
    for (const k in sealed.basics) sealed.basics[k] = 0;
    renderSealed();
  };
  $('#seal-copy').onclick = () => sealed && copySealed();
  $('#seal-show').checked = pref.get('sealShow', '1') === '1';
  $('#seal-show').addEventListener('change', () => { pref.set('sealShow', $('#seal-show').checked ? '1' : '0'); renderSealed(); });
  for (const b of $('#mode').children) b.onclick = () => setMode(b.dataset.mode);
  setMode(pref.get('mode', 'draft') === 'sealed' ? 'sealed' : 'draft');
  $('#export-csv').onclick = exportCsv;
  $('#export-jsonl').onclick = exportJsonl;
  document.addEventListener('keydown', (e) => {
    if (e.target.closest('select, input, textarea') || mode !== 'draft') return;
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
