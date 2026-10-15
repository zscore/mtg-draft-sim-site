"""Export a picknet_all checkpoint to ONNX for the browser, and check it against PyTorch.

    python tools/export_model.py [path/to/all_d256_xxx.pt]   (default: the newest all_d256 .pt)

The browser model is one pick step: given a batch of seats' drafts so far (T history slots in the
3 x 15 layout, the last slot being the pack in hand), it returns logits over that pack's 15 tokens.
The card table (learned name vectors + projected card features) is precomputed into one
[n_cards + 1, d] constant, and the candidate blocks only run for the last slot, so a step costs
one pass of the history transformer over T <= 45 tokens plus the pack blocks over <= 15 cards.
Inputs are int32 so the page can feed Int32Arrays.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PICKNET = Path(__file__).resolve().parents[2] / "picknet"
DATA = Path(os.environ.get("MTG_DATA_DIR") or PICKNET.parent / "mtg-data" / "data")  # shared 17Lands data (mtg-data)
sys.path[:0] = [str(PICKNET / "src"), str(PICKNET.parent / "mtg-data" / "src")]
from picknet.model import PickNet, bag  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
SHARDS = DATA / "features" / "picks_all"
OUT = SITE / "model" / "picknet.onnx"


def load(path: Path) -> PickNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, cards = ck["cfg"], np.load(SHARDS / "cards.npz")
    feats = (cards["feats"] - ck["feat_mu"]) / ck["feat_sd"]
    m = PickNet(len(cards["name_id"]), 15, d=cfg["d"], heads=cfg["heads"], hist_layers=cfg["hist_layers"],
                pack_layers=cfg["pack_layers"], dropout=cfg["dropout"], feats=feats, id_map=cards["name_id"],
                n_ctx=ck["n_ctx"], n_sets=len(ck["sets"]), deck_head=cfg["deck_w"] > 0)
    m.load_state_dict(ck["state"])
    return m.eval()


class PickStep(nn.Module):
    """PickNet.forward restricted to the last history slot, with the card table baked in."""

    def __init__(self, m: PickNet):
        super().__init__()
        self.m = m
        with torch.no_grad():
            self.register_buffer("tab", m.cards().clone())
        del m.cards  # the raw features and projection are no longer needed

    def forward(self, idx, cnt, pick, oidx, ocnt, ctx, sid):
        m, tab = self.m, self.tab
        idx, cnt, pick, oidx, ocnt, sid = (a.long() for a in (idx, cnt, pick, oidx, ocnt, sid))
        T = idx.shape[1]
        e_pick = F.embedding(pick, tab)
        prev = torch.cat([torch.zeros_like(e_pick[:, :1]), e_pick[:, :-1]], 1)
        h = (m.pos.weight[None, :T] + m.pack_bag(bag(tab, idx, cnt)) + m.ctx_mlp(ctx)[:, None]
             + m.set_emb(sid)[:, None] + m.pick_in(prev) + m.oth_bag(bag(tab, oidx, ocnt)))
        h = m.hist_norm(m.hist(h, mask=m.causal[:T, :T], is_causal=True))
        ci, cc = idx[:, -1:], cnt[:, -1:]
        x = F.embedding(ci, tab) + m.copies(cc.clamp(max=3)) + m.state_proj(h[:, -1:])[:, :, None]
        see = torch.ones(1, T, dtype=torch.bool, device=h.device)
        for blk in m.blocks:
            x = blk(x, cc > 0, h, see)
        return m.head(x).squeeze(-1)[:, 0]


def test_batch(set_code="BLB", n=8):
    d = SHARDS / f"{set_code}.PremierDraft"
    rows = np.flatnonzero(np.load(d / "split.npy") == 2)[:n]
    a = {k: np.asarray(np.load(d / f"{k}.npy", mmap_mode="r")[rows]) for k in
         ("idx", "cnt", "pick", "oidx", "ocnt", "ctx", "sid")}
    return {k: v.astype(np.float32 if k == "ctx" else np.int32) for k, v in a.items()}


def main(path: str = ""):
    path = Path(path) if path else max((PICKNET / "results/picknet_all/models").glob("all_d256*.pt"),
                                       key=lambda p: p.stat().st_mtime)
    print(f"checkpoint {path}")
    full = load(path)
    step = PickStep(load(path)).eval()
    b = test_batch()
    names = ("idx", "cnt", "pick", "oidx", "ocnt", "ctx", "sid")
    ex = tuple(torch.from_numpy(b[k][:, :20]) if b[k].ndim > 1 and k != "ctx" else torch.from_numpy(b[k])
               for k in names)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    B, T = torch.export.Dim("B", min=1, max=64), torch.export.Dim("T", min=1, max=45)
    dyn = {"idx": {0: B, 1: T}, "cnt": {0: B, 1: T}, "pick": {0: B, 1: T}, "oidx": {0: B, 1: T},
           "ocnt": {0: B, 1: T}, "ctx": {0: B}, "sid": {0: B}}
    prog = torch.onnx.export(step, ex, dynamo=True, input_names=list(names), output_names=["logits"],
                             dynamic_shapes=dyn, external_data=False)
    prog.optimize()
    prog.save(str(OUT), external_data=False)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

    # check: ONNX step at slot t == full PyTorch forward at slot t, on real test drafts
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT))
    with torch.no_grad():
        ref = full(*(torch.from_numpy(b[k]) for k in names)).numpy()
    worst, t0 = 0.0, time.time()
    for t in (0, 1, 7, 8, 14, 15, 23, 30, 44):
        feed = {k: (b[k][:, :t + 1] if b[k].ndim > 1 and k != "ctx" else b[k]) for k in names}
        out = sess.run(None, feed)[0]
        real = b["cnt"][:, t] > 0
        if real.any():
            worst = max(worst, float(np.abs(out - ref[:, t])[real].max()))
    print(f"max |onnx - torch| logit over real cards: {worst:.2e}  ({(time.time() - t0) * 1000 / 9:.0f} ms/step, "
          f"batch {len(b['sid'])}, onnxruntime CPU)")


if __name__ == "__main__":
    main(*sys.argv[1:])
