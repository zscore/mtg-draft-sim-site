# Draft vs PickNet

Draft against 7 bots driven by PickNet (mtg-open-lane's all-sets pick model), running in the browser
with onnxruntime-web. Packs are opened by packs.js from each set's MTG Arena collation
(tools/collation.py): Wizards' published slot rates, corrected where real Arena packs differ.

    python3 serve.py            # http://localhost:8000 (COOP/COEP headers enable WASM threads)

Sealed mode opens six packs and has PickNet rate the pool: each card's first-pick logit, read against
an anchor card shared by every pack the pool is split into (draft.js rateCards). Auto-build takes the
two colours with the best top 23 by that rating, plus 17 basics.

Card images are hotlinked from Scryfall's image CDN. A card on a bonus sheet (Special Guests, Breaking
News, ...) shows the printing its sheet's Scryfall query matched; main-set cards show the main set's.

Rebuilding:

    python tools/export_model.py [checkpoint.pt]   # model/picknet.onnx, checked against PyTorch (needs torch, onnx, onnxscript, onnxruntime)
    python tools/build_data.py [SET ...]           # data/*.json (cards, Scryfall images, collation, context presets)
    npm install && npm run vendor                  # vendor/ort/ (onnxruntime-web WASM runtime)
    python tools/make_fixture.py && node tools/verify_js.mjs   # replay real drafts through draft.js + ONNX vs PyTorch
    node tools/sim_js.mjs OTJ 4                    # one 8-bot draft in Node: timing and seat colours

Pack collation:

    python tools/fit_rates.py SET ...              # Arena rarity rates measured from 17Lands vs the documented ones
    python tools/verify_packs.py [SET ...]         # generated packs vs held-out real 17Lands packs

verify_packs scores every pack's log-probability under the generator and compares real packs with
generated ones (KS test), split into "which sheets" and "which cards within a sheet", plus per-sheet
counts, per-card chi-square, duplicates and colour balance. fit_rates fits on half of the drafts and
verify_packs tests on the other half. Both need the 17Lands parquet files from mtg-open-lane.
