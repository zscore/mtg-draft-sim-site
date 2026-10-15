"""Per-set data for the draft page: cards (+ Scryfall image/colour/rarity), pack collation, and
per-draft context vectors for the model.

    python tools/build_data.py [SET ...]

Writes data/sets.json and data/{SET}.json, the cards with their mana facts (tools/mana_facts.py). The
collation (tools/collation.py) is resolved to card ids here: each sheet's Scryfall query, matched by
name to the set's 17Lands card list. Scryfall lookups are cached in tools/scryfall_cache.json and
tools/scryfall_sheets.json.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from collation import SETS, per_card
from mana_facts import add_mana_facts

SITE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTG_DATA_DIR") or SITE.parent / "mtg-data" / "data")  # shared 17Lands data (mtg-data)
SHARDS = DATA / "features" / "picks_all"
CACHE = SITE / "tools" / "scryfall_cache.json"
SHEET_CACHE = SITE / "tools" / "scryfall_sheets.json"
HEADERS = {"User-Agent": "mtg-draft-sim-site/0.1", "Accept": "application/json",
           "Content-Type": "application/json"}


def scryfall(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=HEADERS)
    for wait in (0.5, 2, 10, 30, 60):  # Scryfall asks for <= 2 searches/s; back off on 429
        time.sleep(wait)
        try:
            with urllib.request.urlopen(req) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    raise RuntimeError(f"Scryfall kept rate-limiting {url}")


def summarize(card):
    face = card.get("card_faces", [{}])[0]
    img = card.get("image_uris") or face.get("image_uris") or {}
    colors = card.get("colors", face.get("colors", []))
    return {"img": img.get("normal", ""), "rarity": card["rarity"][0], "colors": "".join(colors),
            "cmc": card.get("cmc", 0), "type": card.get("type_line", "")}


def lookup(keys, cache):
    """keys [(name, set)] -> fills cache['name|set'] from Scryfall's collection endpoint, falling back
    to the name alone (any printing) and then to the front face."""
    todo = [k for k in keys if "|".join(k) not in cache]
    for attempt in ("set", "name", "front"):
        if not todo:
            break
        missed = []
        for i in range(0, len(todo), 75):
            chunk = todo[i:i + 75]
            ids = [({"name": n, "set": s.lower()} if attempt == "set" else
                    {"name": n if attempt == "name" else n.split(" // ")[0]}) for n, s in chunk]
            res = scryfall("https://api.scryfall.com/cards/collection", {"identifiers": ids})
            found = {}
            for c in res["data"]:
                found[c["name"].lower()] = found[c["name"].split(" // ")[0].lower()] = c
            for n, s in chunk:
                c = found.get(n.lower()) or found.get(n.split(" // ")[0].lower())
                if c:
                    cache["|".join((n, s))] = summarize(c)
                else:
                    missed.append((n, s))
        print(f"  scryfall by {attempt}: {len(todo) - len(missed)} found, {len(missed)} left")
        todo = missed
    for n, s in todo:
        cache["|".join((n, s))] = {"img": "", "rarity": "c", "colors": "", "cmc": 0, "type": ""}


def search(query, cache):
    """The cards matching a Scryfall query, one printing each (the one the query matched, so a bonus
    sheet's query gives its bonus printing): [{name, img, ...}]."""
    if not all(isinstance(c, dict) for c in cache.get(query, [None])):  # missing, or an old names-only entry
        url = "https://api.scryfall.com/cards/search?" + urllib.parse.urlencode({"q": query, "unique": "cards"})
        found = []
        while url:
            try:
                res = scryfall(url)
            except urllib.error.HTTPError as e:
                if e.code != 404:  # Scryfall answers 404 for an empty result
                    raise
                break
            found += [{"name": c["name"], **summarize(c)} for c in res["data"]]
            url = res.get("next_page")
        cache[query] = sorted(found, key=lambda c: c["name"])
    return cache[query]


def resolve_collation(code, cards, cache):
    """tools/collation.py's entry for a set -> ({sheets: {name: [card ids]}, layouts, balance}, images),
    with the sheets' Scryfall names matched to the set's cards (full name or front face) and per-card
    slot weights turned into per-slot ones. Fails loudly on an empty sheet. images: card id -> the image
    of the printing its sheet holds, the main set's sheets first, so a card that only comes on a bonus
    sheet (Special Guests, Breaking News, ...) shows that printing and a main-set card its plain one."""
    spec = SETS[code]
    by_name = {}
    norm = lambda n: re.sub(r"[^\x00-\x7f]", "?", n.lower())  # 17Lands writes "Bespoke Bō" as "Bespoke B?"
    for i, c in cards.items():
        by_name.setdefault(norm(c["name"]), i)
        by_name.setdefault(norm(c["name"].split(" // ")[0]), i)
    sheets, images = {}, {}
    main_first = sorted(spec["sheets"].items(), key=lambda kv: not kv[1]["q"].startswith(f"e:{code.lower()} "))
    for name, sh in main_first:
        ids = set()
        for c in search(sh["q"], cache):
            n = c["name"]
            i = by_name.get(norm(n), by_name.get(norm(n.split(" // ")[0])))
            if i is not None:
                ids.add(i)
                if c["img"]:
                    images.setdefault(i, c["img"])
        sheets[name] = ids
    for name, sh in spec["sheets"].items():
        for m in sh["minus"]:
            sheets[name] -= sheets[m]
        if not sheets[name]:
            raise ValueError(f"{code}: sheet {name} ({sh['q']}) matched no cards")
    sheets = {k: sorted(v) for k, v in sheets.items()}

    def slot(sl):
        w = {k: v * len(sheets[k]) if isinstance(sl["w"], per_card) else v for k, v in sl["w"].items() if v > 0}
        tot = sum(w.values())
        out = {"n": sl["n"], "w": [[k, round(v / tot, 6)] for k, v in w.items()]}
        for k in ("balance", "dedupe", "pair"):
            if sl.get(k):
                out[k] = True
        return out

    layouts = [{"w": lay["w"], "slots": [slot(sl) for sl in lay["slots"] + spec.get("common", [])]}
               for lay in spec["layouts"]]
    out = {"sheets": sheets, "layouts": layouts, "balance": spec["balance"], "unique": spec.get("unique", False)}
    if spec.get("pairs"):  # card id -> its partner, both ways, for "pair" slots
        ids = [(by_name[norm(a)], by_name[norm(b)]) for a, b in spec["pairs"]]
        out["pairs"] = {**{a: b for a, b in ids}, **{b: a for a, b in ids}}
    return out, images


def main():
    cards = np.load(SHARDS / "cards.npz")
    sets, printing, set_of = list(cards["sets"]), cards["printing"], cards["set_of"]
    pad = len(printing)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    sheet_cache = json.loads(SHEET_CACHE.read_text()) if SHEET_CACHE.exists() else {}
    lookup([(str(printing[i]), sets[set_of[i]]) for i in range(pad)], cache)
    CACHE.write_text(json.dumps(cache))

    index = []
    (SITE / "data").mkdir(exist_ok=True)
    only = sys.argv[1:]
    for sid, code in enumerate(sets):
        if only and code not in only:
            continue
        d = SHARDS / f"{code}.PremierDraft"
        info = json.loads((d / "info.json").read_text())
        split = np.load(d / "split.npy")
        test = np.flatnonzero(split == 2)
        ctx, wr = np.load(d / "ctx.npy")[test], np.load(d / "wr.npy")[test]
        strong = ctx[wr >= 0.62] if (wr >= 0.62).sum() > 100 else ctx
        ids = np.flatnonzero(set_of == sid)
        try:
            set_name = scryfall(f"https://api.scryfall.com/sets/{code.lower()}")["name"]
        except Exception:
            set_name = code
        out = {"code": code, "name": set_name, "sid": sid, "n_picks": info["n_picks"], "pad": pad,
               "ctx": {"typical": ctx.mean(0).round(4).tolist(), "strong": strong.mean(0).round(4).tolist()},
               "cards": {int(i): {"name": str(printing[i]), **cache[f"{printing[i]}|{code}"]} for i in ids}}
        out["collation"], images = resolve_collation(code, out["cards"], sheet_cache)
        for i, img in images.items():
            out["cards"][i]["img"] = img
        add_mana_facts(out["cards"])
        SHEET_CACHE.write_text(json.dumps(sheet_cache, indent=0))
        (SITE / "data" / f"{code}.json").write_text(json.dumps(out, separators=(",", ":")))
        subprocess.run(["node", str(SITE / "tools" / "fit_balance.mjs"), code], check=True)
        index.append({"code": code, "name": set_name, "n_picks": info["n_picks"]})
        sizes = {k: len(v) for k, v in out["collation"]["sheets"].items()}
        print(f"{code} ({set_name}): {len(ids)} cards, {info['n_picks']}-card packs, sheets {sizes}")
    if not only:
        (SITE / "data" / "sets.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
