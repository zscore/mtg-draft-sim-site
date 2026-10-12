"""Real test drafts + PyTorch logits per (slot, card), for tools/verify_js.mjs.

    python tools/make_fixture.py [checkpoint.pt]
"""
import json
import sys

import numpy as np
import torch

from export_model import OL, SHARDS, SITE, load

NAMES = ("idx", "cnt", "pick", "oidx", "ocnt", "ctx", "sid")


def main(path=str(OL / "results/picknet_all/models/all_d256_ep3.pt")):
    m = load(path)
    out = []
    for code in ("BLB", "LCI", "ECL"):
        d = SHARDS / f"{code}.PremierDraft"
        n = json.loads((d / "info.json").read_text())["n_picks"]
        rows = np.flatnonzero(np.load(d / "split.npy") == 2)[:3]
        a = {k: np.asarray(np.load(d / f"{k}.npy", mmap_mode="r")[rows]) for k in NAMES}
        with torch.no_grad():
            ref = m(*(torch.from_numpy(a[k].astype(np.float32 if k == "ctx" else np.int64)) for k in NAMES)).numpy()
        for b in range(len(rows)):
            slots = [r * 15 + k for r in range(3) for k in range(n)]
            out.append({"set": code, "sid": int(a["sid"][b]), "ctx": a["ctx"][b].tolist(), "slots": [{
                "t": t, "pick": int(a["pick"][b, t]),
                "pack": [int(x) for x, c in zip(a["idx"][b, t], a["cnt"][b, t]) for _ in range(c)],
                "logit": {int(x): float(ref[b, t, j]) for j, (x, c) in enumerate(zip(a["idx"][b, t], a["cnt"][b, t])) if c},
            } for t in slots]})
    (SITE / "tools/fixtures/replay.json").write_text(json.dumps(out))
    print(f"{len(out)} drafts")


if __name__ == "__main__":
    main(*sys.argv[1:])
