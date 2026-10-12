"""Check packs.js against real Arena packs: 17Lands Premier Draft P1P1 packs (the whole pack, unopened)
vs packs from tools/gen_packs.mjs.

    python tools/verify_packs.py [SET ...] [--real 20000] [--sim 200000]

Main test (likelihood): every pack gets a log-probability under the generator,
    log P(pack) ~ log P(how many cards of each sheet it has) + sum over cards log P(card | its sheet),
both terms estimated from --sim generated packs. Held-out generated packs and real packs are scored
the same way; if the generator is right the two score distributions match (KS test). The two terms
are also tested on their own, to tell "wrong slot/rarity mix" from "wrong cards within a sheet".
A real pack whose sheet mix the generator never produced scores -inf and is counted separately.

Real packs come from the half of the drafts that tools/fit_rates.py does not fit on.
The score cannot see colour balance or which cards come together, so those get their own checks.
Needs numpy, scipy, polars and node (mtg-open-lane's or picknet's .venv has the Python side).
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from fit_rates import half

SITE = Path(__file__).resolve().parents[1]
PARQUET = Path("/Users/zblanton/Documents/repos/mtg-data/data/parquet")


def real_packs(code, ids_by_name, n, rng):
    lf = pl.scan_parquet(PARQUET / f"{code}.PremierDraft.parquet")
    cols = [c for c in lf.collect_schema().names() if c.startswith("pack_card_")]
    X = (half(lf, even=False).filter(pl.col("pick_number") == 0).select([pl.col(c).cast(pl.UInt8) for c in cols])
         .collect().to_numpy())
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    unknown = [c[10:] for c in cols if c[10:] not in ids_by_name]
    col_ids = np.array([ids_by_name.get(c[10:], -1) for c in cols])
    return [[int(col_ids[j]) for j in np.flatnonzero(row) for _ in range(row[j])] for row in X], unknown


def sim_packs(code, n, seed):
    out = subprocess.run(["node", str(SITE / "tools" / "gen_packs.mjs"), code, str(n), str(seed)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


class Model:
    """Sheet-mix frequencies and per-card frequencies within each sheet, from generated packs."""

    def __init__(self, packs, label):
        self.label = label
        self.sig = Counter(self.signature(p) for p in packs)
        self.n = len(packs)
        cnt = Counter(c for p in packs for c in p)
        self.tot = Counter()
        for c, k in cnt.items():
            self.tot[label(c)] += k
        self.q = {c: k / self.tot[label(c)] for c, k in cnt.items()}

    def signature(self, p):
        return tuple(sorted(Counter(self.label(c) for c in p).items()))

    def score(self, p):
        k = self.sig.get(self.signature(p), 0)
        s_sig = np.log(k / self.n) if k else -np.inf
        s_card = sum(np.log(self.q[c]) if c in self.q else -np.inf for c in p)
        return s_sig, s_card


def ks(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = stats.ks_2samp(a, b)
    return r.statistic, r.pvalue


def colours_present(p, keep, colors):
    return set("".join(colors[c] for c in p if keep(c))) >= set("WUBRG")


def same_colour_ratio(packs, sheet, colors):
    """Mean over same-colour mono pairs of P(both in pack) / (P(a) P(b)); 1 = independent."""
    mono = [c for c in sheet if len(colors[c]) == 1]
    ix = {c: i for i, c in enumerate(mono)}
    B = np.zeros((len(packs), len(mono)), np.float32)
    for r, p in enumerate(packs):
        for c in p:
            if c in ix:
                B[r, ix[c]] = 1
    f = B.mean(0)
    co = (B.T @ B) / len(packs)
    same = np.array([[colors[a] == colors[b] and a != b for b in mono] for a in mono])
    ok = same & (np.outer(f, f) > 0)
    return float((co[ok] / np.outer(f, f)[ok]).mean())


def check(code, n_real, n_sim, rng):
    data = json.loads((SITE / "data" / f"{code}.json").read_text())
    col, cards = data["collation"], {int(k): v for k, v in data["cards"].items()}
    ids_by_name = {v["name"]: k for k, v in cards.items()}
    colors = {k: v["colors"] for k, v in cards.items()}
    sheet_of = {}
    for name, ids in col["sheets"].items():
        for c in ids:
            sheet_of.setdefault(c, name)
    label = lambda c: sheet_of.get(c, "?")

    real, unknown = real_packs(code, ids_by_name, n_real, rng)
    sim = sim_packs(code, n_sim + len(real), int(rng.integers(1 << 30)))
    fit, held = sim[:n_sim], sim[n_sim:]
    m = Model(fit, label)
    rs, hs = np.array([m.score(p) for p in real]), np.array([m.score(p) for p in held])
    tot_r, tot_h = rs.sum(1), hs.sum(1)

    print(f"\n=== {code}: {len(real)} real P1P1 packs vs {len(held)} generated (model fit on {n_sim})")
    if unknown:
        print(f"  17Lands cards not in the site's card list: {unknown[:5]}")
    print(f"  pack size  real {dict(sorted(Counter(map(len, real)).items()))}  sim {dict(sorted(Counter(map(len, held)).items()))}")
    impossible = ~np.isfinite(tot_r)
    print(f"  real packs the generator can't produce: {impossible.mean():.2%}"
          f" (sheet mix never generated {np.mean(~np.isfinite(rs[:, 0])):.2%}, unseen card {np.mean(~np.isfinite(rs[:, 1])):.2%})")
    if impossible.any():
        ex = Counter(m.signature(p) for p, bad in zip(real, impossible) if bad).most_common(3)
        print("    e.g.", "; ".join(" ".join(f"{k}{v}" for k, v in s) + f" x{n}" for s, n in ex))
    res = {"set": code}
    for name, a, b in (("log P(pack)", tot_r, tot_h), ("  sheet-mix term", rs[:, 0], hs[:, 0]),
                       ("  card term", rs[:, 1], hs[:, 1])):
        d, pv = ks(a, b)
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        print(f"  {name:16s} KS D={d:.4f} p={pv:.3g}   mean real {fa.mean():8.3f} sim {fb.mean():8.3f}"
              f"   5%/50%/95% real {np.percentile(fa, [5, 50, 95]).round(2)} sim {np.percentile(fb, [5, 50, 95]).round(2)}")
        res[name.strip()] = (d, pv)
    res["impossible"] = impossible.mean()

    # Sheet mix: mean count per sheet, and the sheet mixes whose frequency differs most.
    names = list(col["sheets"]) + ["?"]
    cr = np.array([[sum(label(c) == s for c in p) for s in names] for p in real], float)
    ch = np.array([[sum(label(c) == s for c in p) for s in names] for p in held], float)
    z = (cr.mean(0) - ch.mean(0)) / np.sqrt(cr.var(0) / len(cr) + ch.var(0) / len(ch) + 1e-12)
    print("  per pack   " + "  ".join(f"{s}:{a:.3f}/{b:.3f}{'*' if abs(zz) > 4 else ''}"
                                       for s, a, b, zz in zip(names, cr.mean(0), ch.mean(0), z) if a or b)
          + "   (real/sim, * = |z|>4)")
    sr, sh = Counter(m.signature(p) for p in real), Counter(m.signature(p) for p in held)
    diff = sorted(set(sr) | set(sh), key=lambda s: -abs(sr[s] / len(real) - sh[s] / len(held)))[:4]
    print("  sheet mixes off most: " + "; ".join(
        " ".join(f"{k}{v}" for k, v in s) + f" {sr[s] / len(real):.3f}/{sh[s] / len(held):.3f}" for s in diff))
    res["max_z"] = float(np.abs(z).max())

    # Cards within each sheet: real vs generated counts (chi-square on the 2 x cards table).
    worst, real_cnt = [], Counter(c for p in real for c in p)
    for s in col["sheets"]:
        ids = [c for c in col["sheets"][s] if sheet_of[c] == s]
        a = np.array([real_cnt[c] for c in ids], float)
        b = np.array([m.q.get(c, 0) for c in ids]) * m.tot[s]
        keep = (a + b) > 0
        if keep.sum() < 2 or a.sum() < 30:
            continue
        p = stats.chi2_contingency(np.stack([a[keep], b[keep]]))[1]
        worst.append((p, s, len(ids)))
    worst.sort()
    print("  cards within sheet, chi-square p (lowest first): "
          + "  ".join(f"{s}[{n}] {p:.2g}" for p, s, n in worst[:6]))
    res["min_chi_p"] = worst[0][0] if worst else 1.0

    # Things the score can't see.
    dup = lambda ps: np.mean([len(set(p)) < len(p) for p in ps])
    bal = col["sheets"][col["balance"]["sheet"]]
    in_bal = set(bal)
    cb = lambda ps: np.mean([colours_present(p, lambda c: c in in_bal, colors) for p in ps])
    print(f"  packs with a duplicate: real {dup(real):.4f} sim {dup(held):.4f}   "
          f"all 5 colours among {col['balance']['sheet']}: real {cb(real):.3f} sim {cb(held):.3f}   "
          f"same-colour {col['balance']['sheet']} pairs vs independent: real {same_colour_ratio(real, bal, colors):.2f}"
          f" sim {same_colour_ratio(held, bal, colors):.2f}")
    res.update(dup=(dup(real), dup(held)), bal=(cb(real), cb(held)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*")
    ap.add_argument("--real", type=int, default=20000)
    ap.add_argument("--sim", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    codes = a.sets or [s["code"] for s in json.loads((SITE / "data" / "sets.json").read_text())]
    rng = np.random.default_rng(a.seed)
    rows = [check(c, a.real, a.sim, rng) for c in codes]
    print("\nset   KS D (logP / mix / cards)   impossible  max|z| mix  min chi p   dup real/sim    5 colours real/sim")
    for r in rows:
        print(f"{r['set']:5s} {r['log P(pack)'][0]:.4f} / {r['sheet-mix term'][0]:.4f} / {r['card term'][0]:.4f}"
              f"     {r['impossible']:7.2%}   {r['max_z']:7.1f}   {r['min_chi_p']:9.2g}   "
              f"{r['dup'][0]:.4f}/{r['dup'][1]:.4f}   {r['bal'][0]:.3f}/{r['bal'][1]:.3f}")


if __name__ == "__main__":
    main()
