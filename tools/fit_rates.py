"""Measure a Play Booster set's Arena slot rates from 17Lands packs, next to the documented ones:
the rare slot's mythic share and the mix of the two wildcard slots (the paper wildcard, and the
paper foil slot that Arena makes a second non-foil wildcard), bonus sheets included.

    python tools/fit_rates.py SET ...

Maximum likelihood over P1P1 packs: a pack's cards beyond the fixed common and uncommon slots
come from the rare slot, the two wildcards and any other slot (a bonus-sheet slot, TMT's Turtle
slot), which keeps its documented rates. Cards are grouped by rarity (main-set sheets) or by bonus
sheet; land-slot cards are left out. Fits on drafts whose id ends in an even hex digit;
tools/verify_packs.py tests on the odd ones, so its comparison stays out-of-sample.
Prints values for tools/collation.py's arena=.
"""
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import chi2

from collation import SETS, per_card

SITE = Path(__file__).resolve().parents[1]
PARQUET = Path("/Users/zblanton/Documents/repos/mtg-data/data/parquet")
# Sheets that are just part of a main-set rarity for this purpose (rare dual lands, Turtles).
RARITY = {"DUAL": "R", "TC": "C", "TU": "U", "TR": "R", "TM": "M"}
group = lambda s: RARITY.get(s, s)


def half(lf, even):
    """Half of the drafts, split on the draft id's last hex digit."""
    last = pl.col("draft_id").str.slice(-1)
    return lf.filter(last.is_in(list("02468ace")) if even else ~last.is_in(list("02468ace")))


def dist(w, cats):
    z = np.array([sum(p for s, p in w.items() if group(s) == c) for c in cats])
    return z / z.sum()


def fit(code):
    data = json.loads((SITE / "data" / f"{code}.json").read_text())
    col = data["collation"]
    ids = {v["name"]: int(k) for k, v in data["cards"].items()}
    sheet_of = {}
    for s, cs in col["sheets"].items():
        for c in cs:
            sheet_of.setdefault(c, s)
    slots = [(sl["n"], dict(sl["w"]), sl.get("balance", False)) for sl in col["layouts"][0]["slots"]]
    n_c = sum(n for n, _, b in slots if b)
    replaces_common = {s for n, w, b in slots if b for s in w if s != "C"}  # SPG, BIG, source material, ...
    n_u = next(n for n, w, _ in slots if w == {"U": 1})
    rest = [w for n, w, b in slots if not b and w != {"U": 1} and not set(w) <= {"L", "B"}]
    rare = next(w for w in rest if {group(s) for s in w} == {"R", "M"})
    wild, foil = [w for w in rest if {"C", "U", "R", "M"} <= set(w)]
    fixed = [w for w in rest if w is not rare and w is not wild and w is not foil]
    cats = sorted({group(s) for w in rest for s in w}, key=lambda c: ("CURM" + c).index(c))

    lf = pl.scan_parquet(PARQUET / f"{code}.PremierDraft.parquet")
    cols = [c for c in lf.collect_schema().names() if c.startswith("pack_card_")]
    X = half(lf, True).filter(pl.col("pick_number") == 0).select(cols).collect().to_numpy()
    cid = np.array([ids.get(c[10:], -1) for c in cols])
    seen = Counter()
    for row in X:
        cnt = Counter()
        for j in np.flatnonzero(row):
            cnt[group(sheet_of.get(int(cid[j]), "?"))] += int(row[j])
        bonus = sum(cnt[s] for s in replaces_common)
        if bonus > 1 or cnt["?"]:
            continue
        cnt["C"] -= n_c - bonus
        cnt["U"] -= n_u
        seen[tuple(cnt[c] for c in cats)] += 1
    vs, w = list(seen), np.array(list(seen.values()), float)

    fixed_d = [dist(f, cats) for f in fixed]
    ways = defaultdict(list)  # category counts -> (rare, wildcard, foil, fixed slots...) outcomes
    for r in ("R", "M"):
        for xs in itertools.product(range(len(cats)), repeat=2 + len(fixed)):
            k = Counter(cats[i] for i in xs)
            k[r] += 1
            ways[tuple(k[c] for c in cats)].append((r, xs))
    K = len(cats)

    def unpack(t):
        a, b = np.exp(t[1:1 + K]), np.exp(t[1 + K:])
        return 1 / (1 + np.exp(-t[0])), a / a.sum(), b / b.sum()

    def nll(t):
        m, a, b = unpack(t)
        ds = [a, b] + fixed_d
        return -sum(wt * np.log(max(sum((m if r == "M" else 1 - m) * np.prod([d[i] for d, i in zip(ds, xs)])
                                         for r, xs in ways.get(v, [])), 1e-300)) for v, wt in zip(vs, w))

    # Compare with the documented rates, not tools/collation.py's arena= overrides.
    doc = {k: ({s: x * len(col["sheets"][s]) for s, x in v.items()} if isinstance(v, per_card) else v)
           for k, v in SETS[code].get("documented", {}).items()}
    rare, wild, foil = doc.get("rare", rare), doc.get("wild", wild), doc.get("foil", foil)
    m0 = dist(rare, ["R", "M"])[1]
    t0 = np.concatenate([[np.log(m0 / (1 - m0))], np.log(dist(wild, cats) + 1e-3), np.log(dist(foil, cats) + 1e-3)])
    res = minimize(nll, t0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-5, "fatol": 1e-3})
    m, a, b = unpack(res.x)
    lr = 2 * (nll(t0) - res.fun)
    pct = lambda z: "{" + ", ".join(f'"{c}": {100 * x:.1f}' for c, x in zip(cats, z) if x > 5e-4) + "}"
    print(f"{code}: {int(w.sum())} packs; documented vs measured: likelihood ratio {lr:.0f}, "
          f"p = {chi2.sf(lr, 2 * K - 1):.2g}")
    print(f"    rare mythic  documented {100 * m0:.1f}  measured {100 * m:.1f}")
    print(f"    wildcard     documented {pct(dist(wild, cats))}\n                 measured   {pct(a)}")
    print(f"    foil slot    documented {pct(dist(foil, cats))}\n                 measured   {pct(b)}")
    print(f'    arena=dict(rare={{"R": {100 * (1 - m):.1f}, "M": {100 * m:.1f}}}, wild={pct(a)}, foil={pct(b)}),')


if __name__ == "__main__":
    for code in sys.argv[1:]:
        fit(code)
