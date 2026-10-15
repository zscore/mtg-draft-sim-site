"""Export a WheelNet (../wheelnet) to ONNX for one set, and check it.

    ../wheelnet/.venv/bin/python tools/export_wheelnet.py SET [path/to/checkpoint.pt]
    (default: the newest of ../wheelnet/results/SET.PremierDraft/models/wheelnet*.pt)

Either kind of checkpoint, as long as it was trained for SET: a per-set WheelNet (scripts/train_nn.py: learned card
vectors plus card features, e.g. wheelnet_time.pt), or a features-only one trained on several sets with SET as the
target (scripts/train_wheelnet_multiset.py), whose card table ends with SET's cards. Writes:
  model/wheelnet_SET.onnx      one batch of drafts -> P(wheel) for every passed card at every wheel-eligible pick.
                               The set's card table (projected card features) is baked in, cards are the set's local
                               indices (17Lands' card order, padding = n cards), inputs are int32 for the page.
  data/wheel/SET.json          the page's side: PickNet card id -> local index, the shapes, and the draft context: the
                               mean of the set's test drafts (all, and 62%+ win rate) for the drafter's rank and skill,
                               if the model reads them, and the set's first draft, for "days into the format" ("time":
                               the page works it out from each draft's date).
  tools/fixtures/wheel_SET.json  a few test drafts (card names per pick) and the model's probabilities for them, for
                               tools/check_wheelnet.mjs to check the page's encoding gives the same.
Inputs, per draft (T = 3 x n_picks history positions, W = n_picks - 8 wheel-eligible picks per pack, M = n_picks - 1):
  pack, pack_n [B, T, n_picks]   the pack shown at each pick: distinct cards (ascending index) and copies
  oth, oth_n   [B, T, M]         from the 9th pick of a pack: what the other seven took from it since you saw it
  pick         [B, T]            the card taken (padding where there is no pick yet)
  idx, copies, mask [B, 3, W, M] the cards passed on at each wheel-eligible pick (the pack minus the pick)
  ctx          [B, n_ctx]        the draft context
-> p [B, 3, W, M]: the chance each passed card comes back 8 picks later. The history is causal and each pick's cards
only see the history up to that pick, so a draft in progress just leaves later positions as padding.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OPEN_LANE = Path("/Users/zblanton/Documents/repos/wheelnet")
sys.path[:0] = [str(OPEN_LANE / "src")]
from mtgdata.data import N_PACKS, RANK_ORDER, load_tensors  # noqa: E402
from mtgdata.encode import draft_context, splits_for  # noqa: E402
from openlane.features import wheel_tensors  # noqa: E402
from openlane.nn import DraftDataset, WheelNet, collate  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
INPUTS = ("pack", "pack_n", "oth", "oth_n", "pick", "idx", "copies", "mask", "ctx")


class WheelStep(nn.Module):
    """WheelNet with one set's card table baked in, int32 inputs and probabilities out."""

    def __init__(self, m: WheelNet, table: torch.Tensor):
        super().__init__()
        self.m = m
        self.register_buffer("tab", table)
        m.card_table = lambda: self.tab  # local indices; the last row (padding) is zero
        for k in ("feats", "feat_proj", "card_"):
            if hasattr(m, k):
                delattr(m, k)
        # additive float masks: the exporter drops the in-place fill that turns bool masks into these
        f = lambda b: torch.zeros(b.shape).masked_fill(b, float("-inf"))
        for k in ("causal", "block", "future_tok"):
            self.register_buffer(k, f(getattr(m, k)))

    def forward(self, pack, pack_n, oth, oth_n, pick, idx, copies, mask, ctx):
        """WheelNet.forward (cross-attention model), with its in-place padding fix written out for ONNX."""
        m = self.m
        pack, oth, pick, idx, copies = (a.long() for a in (pack, oth, pick, idx, copies))
        B = pack.shape[0]
        h = m._bag(pack, pack_n, "bag") + m.pos.weight[None]
        if m.n_ctx:
            h = h + m.draft_ctx(ctx)[:, None]
        h = h + m.taken(m.card(pick))
        if m.use_others:
            h = h + m._bag(oth, oth_n, "others")
        h = m.hist_norm(m.hist(h, mask=self.causal))
        state = h[:, m.wheel_pos]
        n_pos, M = N_PACKS * m.W, m.max_pass
        idx = idx.reshape(B * n_pos, M)
        real = mask.reshape(B * n_pos, M) > 0
        x = (m.card(idx) + m.copies(copies.reshape(B * n_pos, M).clamp(max=3))
             + (m.ctx(state) + m.pick_pos.weight[None]).reshape(B * n_pos, 1, -1))
        # an empty pack attends to its first padding slot (nn.py: pad[:, 0] &= m.any(-1))
        pad = torch.cat([~real[:, :1] & real.any(-1, keepdim=True), ~real[:, 1:]], 1)
        x = m.packenc(x.reshape(B, n_pos * M, -1), h, tgt_mask=self.block, memory_mask=self.future_tok,
                      tgt_key_padding_mask=torch.zeros(pad.shape).masked_fill(pad, float("-inf")).reshape(B, -1))
        return torch.sigmoid(m.head(x).squeeze(-1).reshape(B, N_PACKS, m.W, M))


def load(path: Path, C: int):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["state"]
    assert ck["cfg"].get("cross", True) and ck["cfg"].get("tied_bag", True), "only the default WheelNet layout exports"
    if "card_.weight" in st:  # one set's model: its own cards
        N = st["card_.weight"].shape[0] - 1
        assert N == C, f"the checkpoint has {N} cards, the set {C}: is it this set's?"
        feats = st["feats"][:-1].numpy() if "feats" in st else None
        m = WheelNet(N, card_feats=feats, **ck["cfg"])
    else:  # features-only, several sets: the target is the last block before the padding row
        N = st["feats"].shape[0] - 1
        m = WheelNet(N, card_feats=np.zeros((N, st["feats"].shape[1]), np.float32), ids=False, **ck["cfg"])
    m.load_state_dict(st)
    m.eval()
    with torch.no_grad():
        table = torch.cat([m.card_table()[N - C:N], torch.zeros(1, m.pos.weight.shape[1])]).clone()
    return m, table, ck


def main(code: str, path: str = ""):
    path = Path(path) if path else max((OPEN_LANE / "results" / f"{code}.PremierDraft" / "models").glob("wheelnet*.pt"),
                                       key=lambda p: p.stat().st_mtime)
    print(f"checkpoint {path}")
    t0 = time.time()
    t = load_tensors(code)
    C, n = len(t.names), t.n_picks
    full, table, ck = load(path, C)
    step = WheelStep(load(path, C)[0], table).eval()
    print(f"{code}: {C} cards, {n}-card packs, trained on {ck.get('sets') or [code]} [{time.time() - t0:.0f}s]")

    # a few test drafts, encoded exactly as in training
    inputs = ck.get("inputs") or {"rank": True, "skill": True, "time": True}  # the multi-set script reads all three
    ctx_all = draft_context(t, **inputs)
    test = splits_for(t, code, "PremierDraft")["test"]
    passed, back = wheel_tensors(t)
    rows = test[:8]
    ds = DraftDataset(t, passed, back, rows, ctx_all)
    b = collate([ds[i] for i in range(len(rows))])
    x = {k: v for k, v in zip(INPUTS[:8], b[:7] + [b[8]])}
    x["ctx"] = b[9]
    ex = {k: (v > 0 if k == "mask" else v).to(torch.int32) if k != "ctx" else v.float() for k, v in x.items()}

    out = SITE / "model" / f"wheelnet_{code}.onnx"
    # the TorchScript exporter: torch.export stops at the attention layers' padding-mask checks
    torch.onnx.export(step, tuple(ex[k] for k in INPUTS), str(out), dynamo=False, opset_version=17,
                      input_names=list(INPUTS), output_names=["p"], dynamic_axes={k: {0: "B"} for k in [*INPUTS, "p"]})
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # check: ONNX == the PyTorch model on the real test drafts
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out))
    feed = {k: v.numpy() for k, v in ex.items()}
    p = sess.run(None, feed)[0]
    # the full model indexes every training set's cards: shift local indices to the target's block, as training does
    N = full.C
    glob = lambda a: torch.where(a == C, N, a + (N - C))
    with torch.no_grad():
        ref = torch.sigmoid(full(glob(b[0].long()), b[1], glob(b[2].long()), b[3], glob(b[4]), glob(b[5]), b[6],
                                 b[8], b[9])).numpy()
    m = b[8].numpy()
    with torch.no_grad():
        mine = step(*(ex[k] for k in INPUTS)).numpy()
    print(f"max |rewritten forward - torch| P(wheel): {np.abs(mine - ref)[m].max():.2e}")
    t1 = time.time()
    sess.run(None, {k: v[:2] for k, v in feed.items()})
    print(f"max |onnx - torch| P(wheel): {np.abs(p - ref)[m].max():.2e} ({(time.time() - t1) * 1000:.0f} ms for 2 drafts)")

    # the page's side: PickNet card ids (by name) -> local indices, and the draft context
    site = json.loads((SITE / "data" / f"{code}.json").read_text())
    by_name = {c["name"]: int(i) for i, c in site["cards"].items()}
    local = {str(by_name[nm]): j for j, nm in enumerate(t.names) if nm in by_name}
    missing = [nm for nm in t.names if nm not in by_name]
    if missing:
        print(f"warning: {len(missing)} cards not in data/{code}.json: {missing[:5]}")
    # days into the format: the last two context columns, from the set's first draft
    first = int(np.array(t.draft_time, dtype="datetime64[s]").min().astype(np.int64))
    at = (len(RANK_ORDER) + 1 if inputs["rank"] else 0) + (4 if inputs["skill"] else 0)
    strong = t.user_wr[test] >= 0.62
    ctx = {"typical": ctx_all[test].mean(0), "strong": ctx_all[test][strong].mean(0) if strong.sum() > 100 else ctx_all[test].mean(0)}
    W = n - 8
    meta = {"code": code, "model": f"model/wheelnet_{code}.onnx", "n_picks": n, "W": W, "M": n - 1, "pad": C,
            "n_ctx": int(ctx_all.shape[1]), "ctx": {k: [round(float(v), 4) for v in a] for k, a in ctx.items()},
            "time": {"start": first * 1000, "at": at} if inputs["time"] else None, "inputs": inputs,
            "trained_on": ck.get("sets") or [code], "checkpoint": path.name, "cards": local}
    (SITE / "data" / "wheel").mkdir(exist_ok=True)
    (SITE / "data" / "wheel" / f"{code}.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote data/wheel/{code}.json ({len(local)} cards)")

    # fixture: each test draft's packs and picks by name, its context, and the model's P(wheel) per passed card
    drafts = []
    for i, d in enumerate(rows):
        picks = []
        for k in range(N_PACKS):
            for j in range(n):
                cnt = t.pack[d, k, j]
                pack = [t.names[c] for c in np.flatnonzero(cnt) for _ in range(int(cnt[c]))]
                picks.append({"pack": pack, "pick": t.names[t.pick[d, k, j]]})
        probs = {}
        for k in range(N_PACKS):
            for w in range(W):
                for s in range(n - 1):
                    if m[i, k, w, s]:
                        probs.setdefault(f"{k}.{w}", {})[t.names[int(b[5][i, k, w, s])]] = round(float(ref[i, k, w, s]), 5)
        drafts.append({"picks": picks, "ctx": [float(v) for v in ctx_all[d]], "p": probs})
    fx = SITE / "tools" / "fixtures" / f"wheel_{code}.json"
    fx.write_text(json.dumps({"code": code, "drafts": drafts}))
    print(f"wrote {fx.relative_to(SITE)} ({len(drafts)} drafts) [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(*sys.argv[1:])
