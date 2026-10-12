# Draft vs PickNet

Draft against 7 bots driven by PickNet (the all-sets pick model from ../picknet), running in the browser
with onnxruntime-web. Packs are opened by packs.js from each set's MTG Arena collation
(tools/collation.py): Wizards' published slot rates, corrected where real Arena packs differ.

    python3 serve.py            # http://localhost:8000 (COOP/COEP headers enable WASM threads)

The Drafter profile picks the context vector PickNet is conditioned on: "typical" is the mean context
of every 17Lands drafter in the set's test split, "strong" the mean over drafters with a 62%+ win rate
(all drafters if a set has 100 or fewer of those). It is one model either way, asked to pick as that
average drafter would, not a separate model trained on strong players. The profile applies to
everything PickNet does: the bots, your seat's "Show PickNet's view" ranks and percentages (your slice
of the same 8-seat run), the PickNet-pick and autodraft buttons, the pick log's comparison, and sealed
ratings. Changing it mid-draft takes effect from the next pick; the pack on screen keeps its numbers.

The Seed box makes packs repeatable. The same seed opens the same sealed pool, or the same draft packs and
bot picks (as long as your picks are the same). Leave it empty for a random seed. Either way, the seed in use
shows next to the box, and clicking it puts it in the box to replay or share.

Sealed mode opens six packs and has PickNet rate the pool: each card's first-pick logit, read against
an anchor card shared by every pack the pool is split into (draft.js rateCards). "PickNet build" takes
the two colours with the best top 23 by that rating, as many of their best spells as the curve's land
count leaves room for, and the mana base below. A utility land goes in for a basic when PickNet rates it
at least as high as the deck's weakest spell.

## Mana base

Both builds get their lands from the same rules (deck.js manaBase), working from the spells alone:
- Land count from the curve: 17 at a mean mana value of 2.9, one more or fewer per 0.4, and half a
  land fewer per mana rock, mana creature or landcycler, held to 16-18.
- Nonbasic lands from the pool only when they make two of the deck's colours (main or splash). A
  basic-fetcher (Terramorphic Expanse) or a choose-a-colour land (Mirage Mesa) only goes in with a
  splash. At most three lands that enter tapped. A dual that makes one of the deck's colours and one
  it doesn't play is a tapped basic, and stays in the sideboard. Filter lands and lands whose mana only
  casts some spells don't count as fixing, but can still go in as utility lands.
- Utility lands (Mutavault, Hidden Necropolis, the BLB villages) make only colourless mana or only the
  deck's colours. At most two go in, each replacing a basic. "Build best deck" plays one when the model
  rates the deck higher with it. The PickNet build plays one when PickNet rates it at least as high as
  the deck's weakest spell.
- Basics fill the rest. A splash colour (three cards or fewer) gets 3 sources, or 4 with a double-pip
  card. The main colours split the rest by their pips, but get at least 7 sources each.
- Up to four colours, as long as each colour gets the sources it needs (Frank Karsten's numbers for 40
  cards): 7 for a main colour, 3 for a splash. Two colours always do. A third main colour, or a fourth
  colour, only fits with enough dual lands. Fetch lands count as a source for every colour.

These read the cards' mana facts, which tools/mana_facts.py takes from Scryfall's oracle text into
data/{SET}.json: pips (front face only, since an adventure or omen half is optional), the colours a land
taps for, enters tapped, ramp, and landcycling.

## Deck builder

Sealed pools, and every seat's picks once a draft ends, get a deck builder driven by a deck model:
mtg-open-lane's best game-level model from the deck synergy work (mtg-open-lane-synergy, rounds 3-4),
retrained per set by tools/export_deck_model.py and run in deck.js.

    logit P(win game) = base + w.x + sum_{i<j} x_i x_j <v_i, v_j> + F(x).w_F + aff(x) * w_aff

x is the deck's card counts with basics, w card power, v rank-4 pair vectors between nonland cards
(synergy), F the deck features (land count, curve, colours and splash, coloured-source shortfall) and
aff the deck's mean pairwise affinity from the set's PickNet `pool` model (sets without one, MOM, TMT
and WOE, go without). It is fit on 17Lands game results with play/draw, mulligans, ranks, skill and
record as covariates, so card values are skill-adjusted. Draft pools use a model fit on Premier Draft
games, sealed pools one fit on Sealed games. Each picks its penalties on validation games (the first
70% of events train, the next 15% validate).

"Build best deck" searches the pool for the 40-card deck with the highest predicted win rate: a
best-improvement hill climb over spell swaps (a deck spell out, a pool spell in) and utility lands in or
out, from each one- and
two-colour start, then random three-swap kicks and re-climbs from the best deck. Every swap rebuilds the
lands with the mana base rules above, so the search can move into and out of a splash. Once a climb
settles, the land count is set to what the curve asks for: the best spell is added or dropped, and the
climb runs again. A build takes about 0.7 s. The card labels show what each spell and utility land is
worth to the current deck in win-rate points.

A search finds a model's blind spots, so the builder has guardrails:
- The pair term, each deck feature and the affinity are clamped to the 0.5-99.5% range of real
  training decks. Unclamped, the search stacked cards whose pair vectors aligned far past any real
  deck.
- The search picks spells and utility lands only. The other lands come from the mana base rules,
  because a search over lands found the model's small liking for extra sources and filled decks with
  off-colour duals.
- Every colour needs the sources the mana base rules ask for. Without that, the search built
  four-colour piles of bombs that no mana base could cast.
- Card power and pair vectors are shrunk by n / (n + 2000) training games, because cards that rarely
  make a deck get noisy weights.
- The five basics share one weight. A basic's own weight measured how well its colour does, not what
  the land does.

The predicted win rate is the model's estimate for an average 17Lands player and opponent. A built
deck is chosen as the maximum over many candidate decks, so its prediction runs optimistic (winner's
curse). It is best read as a ranking of decks from the same pool, not a forecast.

Card images are hotlinked from Scryfall's image CDN. A card on a bonus sheet (Special Guests, Breaking
News, ...) shows the printing its sheet's Scryfall query matched; main-set cards show the main set's.

Rebuilding:

    python tools/export_model.py [checkpoint.pt]   # model/picknet.onnx, checked against PyTorch (needs torch, onnx, onnxscript, onnxruntime)
    python tools/build_data.py [SET ...]           # data/*.json (cards, Scryfall images, collation, context presets, mana facts)
    python tools/mana_facts.py [SET ...]           # just the mana facts in data/*.json, from ../mtg-data's Scryfall oracle dump
    python tools/export_deck_model.py [SET ...]    # data/deck/*.json, deck models (mtg-open-lane's .venv; ~2 min/set)
    node tools/verify_deck.mjs [SET ...]           # deck.js vs PyTorch on training decks, plus a timed build
    npm install && npm run vendor                  # vendor/ort/ (onnxruntime-web WASM runtime)
    python tools/make_fixture.py && node tools/verify_js.mjs   # replay real drafts through draft.js + ONNX vs PyTorch
    node tools/sim_js.mjs OTJ 4                    # one 8-bot draft in Node: timing and seat colours

Pack collation:

    python tools/fit_rates.py SET ...              # Arena rarity rates measured from 17Lands vs the documented ones
    python tools/verify_packs.py [SET ...]         # generated packs vs held-out real 17Lands packs

verify_packs scores every pack's log-probability under the generator and compares real packs with
generated ones (KS test), split into "which sheets" and "which cards within a sheet", plus per-sheet
counts, per-card chi-square, duplicates and colour balance. fit_rates fits on half of the drafts and
verify_packs tests on the other half. Both need the 17Lands parquet files from ../mtg-data.
