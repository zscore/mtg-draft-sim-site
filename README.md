# Draft vs PickNet

Draft against 7 bots driven by PickNet (mtg-open-lane's all-sets pick model), running in the browser
with onnxruntime-web. Packs are real opening packs from held-out 17Lands drafts.

    python3 serve.py            # http://localhost:8000 (COOP/COEP headers enable WASM threads)

Rebuilding:

    python tools/export_model.py [checkpoint.pt]   # model/picknet.onnx, checked against PyTorch (needs torch, onnx, onnxscript, onnxruntime)
    python tools/build_data.py                     # data/*.json (cards, Scryfall images, opening packs, context presets)
    npm install && npm run vendor                  # vendor/ort/ (onnxruntime-web WASM runtime)
    python tools/make_fixture.py && node tools/verify_js.mjs   # replay real drafts through draft.js + ONNX vs PyTorch
    node tools/sim_js.mjs OTJ 4                    # one 8-bot draft in Node: timing and seat colours
