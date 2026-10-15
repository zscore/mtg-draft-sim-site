"""Per-set deck models for the deck builder: wheelnet's game-level deck model, retrained and exported.

    python tools/export_deck_model.py [SET ...]      # run with wheelnet's .venv (torch, polars)

The model is deck synergy rounds 3-4's best game-level model (mtg-open-lane-synergy, scripts/deck_synergy_v4.py,
"fm4 nonland" with the deck features and the affinity column):

    logit P(win game) = b + ctx . w_ctx + w . x + sum_{i<j} x_i x_j <v_i, v_j> + F(x) . w_F + aff(x) * w_aff

  x       the registered deck's card counts, basic lands included
  v       rank-4 pair vectors, nonland cards only (so synergy can't just be the mana base)
  F(x)    openlane.cardinfo.deck_features: land-count buckets, curve shape, colours / splash, castability,
          standardised on training games
  aff(x)  the deck's mean pairwise affinity from the set's picknet `pool` model (how much card j in your pool
          raises the odds a human takes card i), standardised; sets without a pool model go without it
  ctx     play/draw, mulligans, ranks, skill, experience, record; constant for deck building, so it is folded into
          `base`, the mean of b + ctx . w_ctx over training games

One model per event: Premier Draft games for draft pools, Sealed games for sealed pools (the Sealed games have no
draft table, so they are split by event time). Trained on the first 70% of events, the next 15% for early stopping.
Writes data/deck/{SET}.json (the browser side is deck.js) and tools/fixtures/deck_{SET}.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

SYN = Path("/Users/zblanton/Documents/repos/mtg-open-lane-synergy")
OL = Path("/Users/zblanton/Documents/repos/wheelnet")
sys.path.insert(0, str(SYN / "src"))
from openlane.cardinfo import DECK_FEATURES, card_info, deck_features  # noqa: E402
from openlane.data import COLORS, PARQUET, RAW  # noqa: E402
from openlane.deckstrength import (aligned, covariates, event_split, experience, flag, record_before,  # noqa: E402
                                   record_onehot)
from openlane.games import game_path  # noqa: E402
from openlane.synergy import DeckFM, _train  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
SETS = ["BLB", "DFT", "DSK", "ECL", "EOE", "FDN", "FIN", "HOB", "LCI", "MKM", "MOM", "MSH", "OTJ", "SOS", "TDM",
        "TLA", "TMT", "WOE"]  # FRA: no 17Lands game data yet
K, LR_DROPS = 4, 2
# Penalties on card power and pairs: round 4's validation chose (1e-5, 1e-3) for fm4 nonland on all 12 sets (Premier
# Draft, from a smaller grid). Sealed has ~7x fewer games, so each event picks its own on validation games.
GRID = [(l2_w, l2_v) for l2_w in (1e-5, 1e-4) for l2_v in (1e-3, 3e-3, 1e-2, 3e-2)]
# Cards that rarely make a deck (bonus sheets, off-archetype chaff) get noisy weights: a card in 59 decks can rate
# above 95% of real playables, and a deck optimiser goes straight for those. Power is shrunk toward 0 (about the
# median playable) by n / (n + SHRINK) training games, the pair vectors by its square root.
SHRINK = 2000


def pool_vectors(set_code: str, names: list[str]):
    """The picknet `pool` model's card vectors, in `names` order: a_i (what a candidate responds to) and e_j (what
    a pool card contributes); affinity S_ij = (<a_i, e_j> + <a_j, e_i>) / 2. None without a pool model. Cards the
    pick data lacks get zero vectors."""
    path = OL / "results" / f"{set_code}.PremierDraft" / "picknet" / "models" / "pool_d32.pt"
    if not path.exists():
        return None
    from openlane.picknet import load_model
    cols = pl.scan_parquet(PARQUET / f"{set_code}.PremierDraft.parquet").collect_schema().names()
    pos = {c[len("pack_card_"):]: i for i, c in enumerate(c for c in cols if c.startswith("pack_card_"))}
    m, _ = load_model(path)
    with torch.no_grad():
        e, a = m.cards().numpy()[:-1], m.a.weight.numpy()[:-1]
    A, E = np.zeros((len(names), a.shape[1]), np.float32), np.zeros((len(names), e.shape[1]), np.float32)
    for i, n in enumerate(names):
        if n in pos:
            A[i], E[i] = a[pos[n]], e[pos[n]]
    return A, E


def scryfall_spelling(names: list[str]) -> list[str]:
    """17Lands writes non-ASCII letters as '?' ("Bespoke Bō" -> "Bespoke B?"); card_info looks names up in Scryfall's
    oracle file, so those get Scryfall's spelling back. Other names pass through."""
    odd = [n for n in names if "?" in n]
    if not odd:
        return names
    import gzip
    import re
    norm = lambda n: re.sub(r"[^\x00-\x7f]", "?", n)  # noqa: E731
    found = {}
    with gzip.open(RAW / "scryfall_oracle_cards.jsonl.gz", "rt") as f:
        for line in f:
            name = json.loads(line)["name"]
            for key in (name, name.split(" // ")[0]):
                if norm(key) in odd:
                    found.setdefault(norm(key), key)
    return [found.get(n, n) for n in names]


def mean_affinity(X, A, E, nonland):
    """Mean pairwise S over the nonland copies of each deck (deck_synergy_v2.mean_pair, S's diagonal zeroed)."""
    Xn = X * nonland
    n = Xn.sum(1)
    xa, xe = Xn @ A, Xn @ E
    tot = (xa * xe).sum(1) - (Xn ** 2 * (A * E).sum(1)).sum(1)
    return tot / np.maximum(n * (n - 1), 1)


def centres(X, info):
    """The batch means deck_features centres on (curve counts, mean mana value, creatures, removal). It takes them
    from whatever batch it is given; for the model that is every game of the event, so the browser needs them."""
    land, mv = info["is_land"].to_numpy(), info["mv"].to_numpy()
    buckets = np.stack([~land & (np.clip(mv, 1, 6) == b) for b in range(1, 7)], 1).astype(np.float32)
    n_spell = X @ (~land).astype(np.float32)
    mv_mean = (X @ (mv * ~land)) / np.maximum(n_spell, 1)
    return {"curve": (X @ buckets).mean(0).tolist(), "mv_mean": float(mv_mean.mean()),
            "creatures": float((X @ info["is_creature"].to_numpy().astype(np.float32)).mean()),
            "removal": float((X @ info["text_removal"].to_numpy().astype(np.float32)).mean())}


def fit_event(set_code, event, names, info, nonland, vec, log):
    g = pl.read_parquet(game_path(set_code, event))
    sp = event_split(g)
    X = aligned(g, "deck_", names).astype(np.float32)
    y = flag(g["won"]).astype(np.int8)
    wb, lb, _ = record_before(g)
    ctx = np.hstack([covariates(g), experience(g), record_onehot(wb, lb)]).astype(np.float32)
    n_decks = (X[sp == 0] > 0).astype(np.float32)
    games_with = n_decks.sum(0)
    del g
    F, _ = deck_features(X, info)
    tr = sp == 0
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    cols = [ctx, (F - mu) / sd]
    aff = None
    if vec is not None:
        a = mean_affinity(X, *vec, nonland)
        aff = (float(a[tr].mean()), float(a[tr].std() + 1e-6))
        cols.append(((a - aff[0]) / aff[1])[:, None])
    c = np.hstack(cols).astype(np.float32)
    t0 = time.time()
    fits = []
    for l2_w, l2_v in GRID:
        torch.manual_seed(0)
        m = DeckFM(len(names), c.shape[1], K, vmask=nonland)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        m, best, _ = _train(m, opt, X, c, y, sp, None, lambda m=m, a=l2_w, b=l2_v: m.penalty(a, b, 0), 40, 4096, 3,
                            "cpu", lr_drops=LR_DROPS)
        fits.append((best, (l2_w, l2_v), m))
    best, l2, m = min(fits, key=lambda f: f[0])
    keep = games_with / (games_with + SHRINK)
    basic = info["is_basic"].to_numpy()
    with torch.no_grad():
        m.w.mul_(torch.from_numpy(keep))
        m.v_.mul_(torch.from_numpy(np.sqrt(keep))[:, None])
        # A basic's own weight says how well decks of its colour do (Mountain +0.013, Island -0.009 on MSH), not what
        # the land does; the builder would add Mountains for it. One shared weight (the games-weighted mean) leaves
        # basic counts to the mana-base features.
        wb = m.w[basic]
        m.w[basic] = (wb * torch.from_numpy(games_with[basic])).sum() / games_with[basic].sum()
    wc = m.cov.weight.detach().numpy()[0]
    nctx = ctx.shape[1]
    base = float(m.cov.bias.detach().numpy()[0] + (ctx[tr] @ wc[:nctx]).mean())
    with torch.no_grad():
        deck_logit = m.deck_score(torch.from_numpy(X[tr])).numpy() + c[tr, nctx:] @ wc[nctx:]
    # Where the model has data: 0.5% / 99.5% quantiles over training decks of the pair term, each deck feature and
    # the affinity. The deck builder clamps to these, so it can't score a deck by extrapolating (stacking cards whose
    # pair vectors align far beyond any real deck, or an odd feature value no human deck reached).
    q = lambda a: np.quantile(a, [0.005, 0.995]).round(5).tolist()  # noqa: E731
    with torch.no_grad():
        xv = torch.from_numpy(X[tr]) @ m.v
        pair = (0.5 * (xv.pow(2) - torch.from_numpy(X[tr] ** 2) @ m.v.pow(2)).sum(-1)).numpy()
    ranges = {"pair": q(pair), "features": [q(F[tr, j]) for j in range(F.shape[1])],
              "aff": None if aff is None else q(a[tr])}
    # a few training decks with their deck terms, for tools/verify_deck.mjs (the browser scorer must match)
    fx = np.random.default_rng(0).choice(np.flatnonzero(tr), 25, replace=False)
    fixture = [{"deck": {int(j): int(X[i, j]) for j in np.flatnonzero(X[i])},
                "logit": float(deck_logit[np.searchsorted(np.flatnonzero(tr), i)])} for i in fx]
    log(f"  {event}: {len(y)} games (train {tr.sum()}), l2 (w, v) {l2}, val logloss {best:.5f} [{time.time() - t0:.0f}s]")
    r = lambda v, d=5: np.round(np.asarray(v, dtype=np.float64), d).tolist()  # noqa: E731
    return {
        "games": int(len(y)), "train_games": int(tr.sum()), "val_logloss": round(float(best), 5), "l2": list(l2),
        "base": round(base, 5),
        # training decks' deck terms: the spread the builder reports scores against
        "deck_logit_mean": round(float(deck_logit.mean()), 5), "deck_logit_sd": round(float(deck_logit.std()), 5),
        "w": r(m.w.detach().numpy()), "v": r(m.v.detach().numpy()),
        "games_with": games_with.astype(int).tolist(), "ranges": ranges,
        "features": {"names": list(DECK_FEATURES), "mu": r(mu), "sd": r(sd), "w": r(wc[nctx:nctx + len(mu)]),
                     "centres": centres(X, info)},
        "aff": None if aff is None else {"mu": aff[0], "sd": aff[1], "w": round(float(wc[-1]), 5)},
    }, fixture


def main(sets):
    out_dir = SITE / "data" / "deck"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for code in sets:
        t0 = time.time()
        print(f"{code}", flush=True)
        names = sorted({c[len("deck_"):] for ev in ("PremierDraft", "Sealed")
                        for c in pl.scan_parquet(game_path(code, ev)).collect_schema().names() if c.startswith("deck_")})
        info = card_info(scryfall_spelling(names), code)
        nonland = (~info["is_land"].to_numpy()).astype(np.float32)
        vec = pool_vectors(code, names)
        models, fixtures = {}, {}
        for ev in ("PremierDraft", "Sealed"):
            models[ev], fixtures[ev] = fit_event(code, ev, names, info, nonland, vec, print)
        r = lambda v, d=4: np.round(np.asarray(v, dtype=np.float64), d).tolist()  # noqa: E731
        card = {  # the per-card facts deck_features reads
            "is_land": info["is_land"].to_list(), "is_creature": info["is_creature"].to_list(),
            "removal": info["text_removal"].to_list(), "mv": r(info["mv"]),
            "pips": r(info.select([f"pip_{k}" for k in COLORS]).to_numpy(), 3),
            "makes": info.select([f"makes_{k}" for k in COLORS]).to_numpy().astype(int).tolist(),
            "nonland_source": info["nonland_source"].to_list(),
        }
        out = {"code": code, "names": names, "card": card,
               "affinity": None if vec is None else {"a": r(vec[0]), "e": r(vec[1])},
               "draft": models["PremierDraft"], "sealed": models["Sealed"]}
        path = out_dir / f"{code}.json"
        path.write_text(json.dumps(out, separators=(",", ":")))
        fix = SITE / "tools" / "fixtures" / f"deck_{code}.json"
        fix.write_text(json.dumps({"draft": fixtures["PremierDraft"], "sealed": fixtures["Sealed"]}))
        print(f"{code}: {len(names)} cards, affinity {'yes' if vec is not None else 'no'}, "
              f"{path.stat().st_size / 1e6:.2f} MB [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SETS)
