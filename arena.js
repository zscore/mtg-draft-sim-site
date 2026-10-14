// MTG Arena's log, read for the Live tab: entries (from serve.py's /api/arena/log) -> draft events -> drafts.
// Arena card ids (grpIds) stay as they are here; the page maps them to PickNet's cards by name.
//
// The messages, as the 17Lands client reads them (Arena has changed them over time, so several are handled):
//   Draft.Notify {draftId, SelfPack, SelfPick, PackCards: "1,2,3"}         the pack in hand, Premier/Traditional
//   ==> EventPlayerDraftMakePick {request: {DraftId, GrpIds, Pack, Pick}}  your pick there
//   ==> LogBusinessEvents {request: {DraftId, EventId, PackNumber, PickNumber, CardsInPack, PickGrpId}}  pack and pick
//   <== BotDraftDraftStatus / DraftPick {EventName, DraftStatus, PackNumber, PickNumber, DraftPack, PickedCards}
//                                                                          the pack in hand and your picks, Quick Draft
//   ==> BotDraftDraftPick {request: {PickInfo: {EventName, CardIds, PackNumber, PickNumber}}}  your pick there
//   ==> EventJoin {request: {EventName}}                                   a new event
//   Courses / Course {InternalEventName, CurrentModule, CardPool}          an event's pool (the draft's picks, once done)

import { SLOTS } from './draft.js';

const ids = (x) => (typeof x === 'string' ? x.split(',') : Array.isArray(x) ? x : []).map(Number).filter(Number.isFinite);

// The JSON in an entry: from its first { or [ to the end of that line (a stack trace may follow), or else to the end
// of the entry or one of its last few closing brackets.
function json(text) {
  const i = text.search(/[{[]/);
  if (i < 0) return undefined;
  const eol = text.indexOf('\n', i);
  try { return JSON.parse(text.slice(i, eol < 0 ? undefined : eol)); } catch { /* spans lines */ }
  let s = text.slice(i).trimEnd();
  for (let tries = 0; tries < 4 && s; tries++) {
    try { return JSON.parse(s); } catch { /* cut at the previous closing bracket */ }
    s = s.slice(0, Math.max(s.lastIndexOf('}', s.length - 2), s.lastIndexOf(']', s.length - 2)) + 1);
  }
  return undefined;
}

// Requests and some responses wrap the message in a JSON string, sometimes twice.
function unwrap(o) {
  for (let depth = 0; depth < 4 && o && typeof o === 'object'; depth++) {
    const k = ['payload', 'Payload', 'request'].find((key) => key in o);
    if (!k) break;
    let v = o[k];
    if (typeof v === 'string') { try { v = JSON.parse(v); } catch { break; } }
    o = v;
  }
  return o;
}

// One log entry -> [event]: {type: 'pack' | 'pick' | 'join' | 'pool' | 'done', ...}, or null if its JSON doesn't parse
// (yet: the log's last entry may still be being written). pack and pick numbers are as logged (some messages count
// from 0, some from 1); cards are grpIds.
export function parseEntry(text) {
  const raw = json(text);
  if (raw === undefined) return null;
  const o = unwrap(raw);
  if (!o || typeof o !== 'object') return [];
  const out = [];
  if (o.PackCards !== undefined) {
    out.push({ type: 'pack', src: 'notify', draftId: o.draftId ?? o.DraftId, pack: o.SelfPack, pick: o.SelfPick, cards: ids(o.PackCards) });
  }
  if (o.DraftStatus !== undefined) {
    const picked = o.PickedCards ? ids(o.PickedCards) : undefined;
    if (o.DraftStatus === 'PickNext' && o.DraftPack) {
      out.push({ type: 'pack', src: 'bot', event: o.EventName, pack: o.PackNumber, pick: o.PickNumber, cards: ids(o.DraftPack), picked });
    } else if (o.DraftStatus === 'Completed') {
      out.push({ type: 'done', event: o.EventName, picked });
    }
  }
  if (o.PickGrpId !== undefined && o.CardsInPack) {
    const base = { draftId: o.DraftId, event: o.EventId };
    out.push({ type: 'pack', src: 'business', ...base, pack: o.PackNumber, pick: o.PickNumber, cards: ids(o.CardsInPack) });
    out.push({ type: 'pick', ...base, cards: ids([o.PickGrpId]) });
  } else if (o.PickInfo) {
    out.push({ type: 'pick', event: o.PickInfo.EventName, cards: ids(o.PickInfo.CardIds ?? [o.PickInfo.CardId]) });
  } else if (/MakePick/.test(text) && (o.GrpIds || o.GrpId !== undefined)) {
    out.push({ type: 'pick', draftId: o.DraftId, cards: ids(o.GrpIds ?? [o.GrpId]) });
  }
  if (/Event_?Join/.test(text) && typeof o.EventName === 'string' && !o.Course) out.push({ type: 'join', event: o.EventName });
  const courses = Array.isArray(o.Courses) ? o.Courses : o.Course ? [o.Course] : o.CardPool ? [o] : [];
  for (const c of courses) {
    if (c && c.InternalEventName && Array.isArray(c.CardPool)) {
      out.push({ type: 'pool', event: c.InternalEventName, draftId: c.DraftId, module: c.CurrentModule, cards: ids(c.CardPool) });
    }
  }
  return out;
}

const isLimited = (event) => /Draft|Sealed/i.test(event || '');
export const isSealed = (event) => /Sealed/i.test(event || '');
export const isBotDraft = (event) => /Quick|Bot/i.test(event || '');

// The set code in an event name ("PremierDraft_FRA_20260929" -> "FRA"), if it is one of `codes`.
export function eventSet(event, codes) {
  return (event || '').split('_').find((x) => codes.includes(x.toUpperCase()))?.toUpperCase();
}

// Every draft (and sealed pool) in the log, oldest first. A draft: {event, draftId, steps, picked, pool, done};
// a step is one pack you were shown: {cards, pack, pick, src, taken} (taken: the grpId you took, once you have).
export class ArenaDrafts {
  constructor() {
    this.drafts = [];
    this.cur = null; // the draft packs and picks go to
    this.joined = null; // the last event joined, for messages that don't name theirs
  }

  get latest() { return this.drafts[this.drafts.length - 1]; }

  start(ev) {
    const d = { event: ev.event ?? this.joined, draftId: ev.draftId, steps: [], picked: null, pool: null, done: false };
    this.drafts.push(d);
    return (this.cur = d);
  }

  add(ev) {
    let d = this.cur;
    if (ev.type === 'join') {
      if (isLimited(ev.event)) { this.joined = ev.event; this.cur = null; }
    } else if (ev.type === 'pack') {
      const differs = d && ((ev.draftId && d.draftId && ev.draftId !== d.draftId) || (ev.event && d.event && ev.event !== d.event));
      const restart = d && ev.src === 'bot' && ev.pack === 0 && ev.pick === 0 && d.steps.some((s) => s.taken);
      if (!d || d.done || differs || restart) d = this.start(ev);
      d.event ??= ev.event ?? this.joined;
      d.draftId ??= ev.draftId;
      // the same pack can be logged twice (Draft.Notify, then the business event with the pick)
      const key = [...ev.cards].sort((a, b) => a - b).join(',');
      let s = d.steps.find((x) => x.key === key);
      if (!s) d.steps.push(s = { key, cards: ev.cards, taken: null });
      if (ev.pack !== undefined && s.pack === undefined) Object.assign(s, { pack: ev.pack, pick: ev.pick, src: ev.src });
      if (ev.picked) d.picked = ev.picked;
    } else if (ev.type === 'pick') {
      if (!d || (ev.draftId && d.draftId && ev.draftId !== d.draftId)) return;
      for (let i = d.steps.length - 1; i >= 0; i--) {
        const s = d.steps[i];
        if (s.taken === null && s.cards.includes(ev.cards[0])) { s.taken = ev.cards[0]; break; }
        if (s.taken === ev.cards[0]) break; // logged twice
      }
    } else if (ev.type === 'done') {
      if (d && (!ev.event || ev.event === d.event)) { d.done = true; if (ev.picked) d.picked = ev.picked; }
    } else if (ev.type === 'pool') {
      // a finished draft's pool, or an event you are still playing (Arena lists them at login)
      if (!isLimited(ev.event) || ev.module === 'Draft' || ev.module === 'Complete' || !ev.cards.length) return;
      const same = (x) => (ev.draftId && x.draftId === ev.draftId) || (x.event === ev.event && (!x.pool || x.pool.length === ev.cards.length));
      const known = [...this.drafts].reverse().find(same);
      if (known) {
        known.pool = ev.cards;
        known.done = true;
      } else if (!isLimited(this.cur?.event) || this.cur?.done) {
        this.drafts.push({ event: ev.event, draftId: ev.draftId, steps: [], picked: null, pool: ev.cards, done: true });
      }
    }
  }
}

// Each step's place in the draft for packs of n cards: [{step, round, k, t}] in draft order (t = round * SLOTS + k,
// the slot draft.js uses). A pack of m cards is pick n - m of its round; the round comes from the logged pack
// number, which counts from 0 or 1 depending on the message (told apart by its pick number), or failing that
// from the order the packs came in.
export function placeSteps(d, n) {
  const out = [];
  let round = 0, lastK = -1;
  for (const s of d.steps) {
    const k = n - s.cards.length;
    if (k < 0 || k >= n) continue;
    let r;
    if (Number.isInteger(s.pack)) {
      const base = s.pick - k === 0 || s.pick - k === 1 ? s.pick - k : s.src === 'notify' ? 1 : 0;
      r = s.pack - base;
    } else {
      r = k <= lastK ? round + 1 : round;
    }
    if (r < 0 || r > 2) continue;
    round = r;
    lastK = k;
    const t = r * SLOTS + k, at = out.findIndex((x) => x.t === t);
    if (at >= 0) out.splice(at, 1); // a later pack for the same slot replaces it
    out.push({ step: s, round: r, k, t });
  }
  return out.sort((a, b) => a.t - b.t);
}

const sig = (cards) => [...cards].sort((a, b) => a - b).join(',');
const takenCards = (d) => d.steps.map((s) => s.taken).filter((g) => g !== null);

// Whether `pool` is the pool of draft `m`: its cards are all m's picks, give or take a few the log missed (Arena
// restarting mid-draft drops a pick or two).
function poolOf(m, pool) {
  const left = new Map();
  for (const g of pool) left.set(g, (left.get(g) || 0) + 1);
  const picks = takenCards(m);
  for (const g of picks) { if (!left.get(g)) return false; left.set(g, left.get(g) - 1); }
  return picks.length > 0 && pool.length - picks.length <= 3;
}

// Drafts from several logs (oldest first) as one list: a draft that went on after Arena restarted is one draft (by
// its draftId), and an event pool Arena lists at every login is folded into the draft it came from (same cards) or
// kept once. `into` (Map key -> draft) keeps each merged draft the same object from one call to the next, so the page
// can hang its caches on it; its steps are the logs' own step objects.
export function mergeDrafts(drafts, into = new Map()) {
  const out = [], seen = new Set();
  const take = (key, d) => {
    let m = into.get(key);
    if (!m) into.set(key, m = { key });
    if (!seen.has(key)) {
      seen.add(key);
      out.push(Object.assign(m, { event: d.event, draftId: d.draftId, source: d.source, steps: [], picked: d.picked, pool: d.pool, done: d.done }));
    }
    return m;
  };
  for (const d of drafts) {
    if (d.steps.length) {
      const m = take(d.draftId ?? `${d.event}|${d.steps[0].key}`, d);
      for (const s of d.steps) {
        const at = m.steps.findIndex((x) => x.key === s.key);
        if (at < 0) m.steps.push(s);
        else if (m.steps[at].taken === null && s.taken !== null) m.steps[at] = s;
      }
      m.event ??= d.event;
      m.pool = d.pool ?? m.pool;
      m.done ||= d.done;
      if (d.picked && (!m.picked || d.picked.length >= m.picked.length)) m.picked = d.picked;
    } else if (d.pool) {
      const s = sig(d.pool);
      const from = out.find((m) => m.steps.length && (m.pool ? sig(m.pool) === s : poolOf(m, d.pool)));
      if (from) {
        from.event ??= d.event;
        from.pool ??= d.pool;
        from.done = true;
      } else {
        take(`${d.event}|pool|${s}`, d);
      }
    }
  }
  return out;
}
