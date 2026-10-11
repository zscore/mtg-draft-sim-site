"""Per-set data for the draft page: cards (+ Scryfall image/colour/rarity), real opening packs, and
per-draft context vectors for the model.

    python tools/build_data.py

Writes data/sets.json and data/{SET}.json. Opening packs are the full packs seen at the first pick of
a round in held-out (test split) Premier drafts, so collation (rarity, bonus sheets) comes for free;
round 1 is skipped for sets whose P1P1 was not logged. Scryfall lookups are cached in
tools/scryfall_cache.json.
"""
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

OL = Path("/Users/zblanton/Documents/repos/mtg-open-lane")
SHARDS = OL / "data" / "features" / "picks_all"
SITE = Path(__file__).resolve().parents[1]
CACHE = SITE / "tools" / "scryfall_cache.json"
N_DRAFTS = 400  # test drafts per set to take opening packs from
HEADERS = {"User-Agent": "mtg-draft-sim-site/0.1", "Accept": "application/json",
           "Content-Type": "application/json"}


def scryfall(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=HEADERS)
    time.sleep(0.12)  # Scryfall asks for <= 10 requests/s
    with urllib.request.urlopen(req) as r:
        return json.load(r)


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


def main():
    cards = np.load(SHARDS / "cards.npz")
    sets, printing, set_of = list(cards["sets"]), cards["printing"], cards["set_of"]
    pad = len(printing)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    lookup([(str(printing[i]), sets[set_of[i]]) for i in range(pad)], cache)
    CACHE.write_text(json.dumps(cache))

    index = []
    (SITE / "data").mkdir(exist_ok=True)
    for sid, code in enumerate(sets):
        d = SHARDS / f"{code}.PremierDraft"
        info = json.loads((d / "info.json").read_text())
        split = np.load(d / "split.npy")
        test = np.flatnonzero(split == 2)
        rows = test[:N_DRAFTS]
        idx = np.load(d / "idx.npy", mmap_mode="r")
        cnt = np.load(d / "cnt.npy", mmap_mode="r")
        packs = []
        for slot in ((15, 30) if info["p1p1_missing"] else (0, 15, 30)):
            ii, cc = np.asarray(idx[rows, slot]), np.asarray(cnt[rows, slot])
            for a, b in zip(ii, cc):
                packs.append([int(x) for x, k in zip(a, b) for _ in range(k)])
        ctx, wr = np.load(d / "ctx.npy")[test], np.load(d / "wr.npy")[test]
        strong = ctx[wr >= 0.62] if (wr >= 0.62).sum() > 100 else ctx
        ids = np.flatnonzero(set_of == sid)
        try:
            set_name = scryfall(f"https://api.scryfall.com/sets/{code.lower()}")["name"]
        except Exception:
            set_name = code
        out = {"code": code, "name": set_name, "sid": sid, "n_picks": info["n_picks"], "pad": pad,
               "ctx": {"typical": ctx.mean(0).round(4).tolist(), "strong": strong.mean(0).round(4).tolist()},
               "cards": {int(i): {"name": str(printing[i]), **cache[f"{printing[i]}|{code}"]} for i in ids},
               "packs": packs}
        (SITE / "data" / f"{code}.json").write_text(json.dumps(out, separators=(",", ":")))
        index.append({"code": code, "name": set_name, "n_picks": info["n_picks"]})
        print(f"{code} ({set_name}): {len(ids)} cards, {len(packs)} opening packs, {info['n_picks']}-card packs")
    (SITE / "data" / "sets.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
