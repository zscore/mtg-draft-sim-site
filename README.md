# Draft vs PickNet

Draft against 7 bots driven by PickNet (the all-sets pick model from ../picknet), running in the browser
with onnxruntime-web. Packs are opened by packs.js from each set's MTG Arena collation
(tools/collation.py): Wizards' published slot rates, corrected where real Arena packs differ.

    python3 serve.py            # http://localhost:8000 (COOP/COEP headers enable WASM threads)

run-locally.html (linked from the page) walks someone new through cloning and running it, for the Live tab.

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

## Live: drafting in MTG Arena

Served by serve.py, the page gets a Live tab that follows your Arena draft as you make it. It reads Arena's log
(turn on Options → Account → Detailed Logs (Plugin Support) in Arena, then restart it) and names Arena's card ids
from Arena's own card database. The log is only served to this machine. On GitHub Pages there is no log
to read, so the tab stays hidden. For the pack in hand it shows:
- PickNet's pick for you: its rank and probability for each card, given your whole draft so far.
- Wheel odds: the chance a card comes back 8 picks later if you pass it, from WheelNet, the simulation, or both
  (the Wheel odds menu). Both assume you take the card you took, or PickNet's top card when you haven't
  picked yet; a card's own odds assume you take the next best instead.
  - WheelNet (../wheelnet, run by wheelnet.js) is trained on 17Lands drafts to predict which passed cards come
    back, from your whole draft so far. It needs a model exported for the set (below); without one the menu is
    off and the odds come from the simulation.
  - The simulation (draft.js WheelSim) sends the pack round the other seven seats 48 times. Each seat is a
    stand-in drafter: a seat from a few simulated tables of PickNet drafting against itself, kept a few picks
    ahead of yours, picking the way a human would (sampled).
- Each card's first-pick rank in the set (rateCards over the whole set).

Below the pack are your picks, what the other seats took from a pack since you last saw it, and PickNet's
choice next to each of your picks. Clicking a pick in that table (or Replay from pick 1) brings its pack back up,
with PickNet's numbers, the wheel odds and the card you took outlined. For a pick whose pack came back, each card
also says whether it did, and the table lists how many came back and the best of them. The table's caption grades
WheelNet on the draft: its average odds for the cards that came back against those for the ones that were taken.
◀ / ▶ (or the arrow keys) step through the draft. The deck builder opens on your pool when the draft ends, or on
request before then.

The Draft menu lists your older drafts too. Arena keeps a log per session for a few weeks (UTC_Log - *.log in
MTGA_Data/Logs/Logs, next to its card database), and serve.py reads those as well, keeping only the draft
messages (about 1 MB from a few weeks of logs). If Arena restarted mid-draft, the draft is put back together
from both sessions, and the pool Arena lists at each login is matched to the draft it came from. An event
you're still playing that Arena lists at login (a draft whose picks aren't in the log, or sealed) shows its pool
for building. arena.js parses the log: Premier and Traditional Draft
(Draft.Notify, EventPlayerDraftMakePick, LogBusinessEvents), Quick Draft (BotDraftDraftStatus/Pick), and
event pools (CardPool). In Quick Draft the other seats are Arena's bots, not the humans PickNet learned
from, so the wheel odds are rougher there.

The Logs menu picks where the drafts come from:
- Arena (live): Arena's own logs, as above, following the draft you're in.
- Archived logs: the copies tools/archive_arena_logs.sh keeps in ~/mtga-log-archive (MTGA_ARCHIVE_DIR for serve.py
  if they're elsewhere), so drafts older than Arena's few weeks of session logs can be gone back over.
- Example logs, anonymized: the scrubbed logs in tools/fixtures/arena (tools/scrub_arena_log.py), to try the tab
  without Arena. They hold two FRA Premier Draft pools but none of the picks, so they open on the pool and deck builder.
  Naming their cards still needs Arena's card database on the machine.
- Choose a log file…: follows any log you pick, as it grows (Arena's Player.log while you draft, a log from another
  install, an archived .log.gz). serve.py opens the system's file dialog (it starts in Arena's log folder, and may
  open behind the browser), since the page's own file picker can't open ~/Library or AppData and can't follow a file
  that's still being written. The page remembers the file until serve.py restarts.

## Mana base

Both builds get their lands from the same rules (deck.js manaBase), working from the spells alone:
- Land count from the curve: 17 at a mean mana value of 2.9, one more or fewer per 0.4, and half a
  land fewer per mana rock, mana creature or landcycler, held to 16-18.
- Nonbasic lands from the pool only when they make two of the deck's colours (main or splash). A
  basic-fetcher (Terramorphic Expanse) only goes in a deck of three colours or more, and a
  choose-a-colour land (Mirage Mesa) only with a splash. At most three lands that enter tapped. A dual that makes one of the deck's colours and one
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
the best game-level model from the deck synergy work in mtg-deck-strength (rounds 3-4),
retrained per set by tools/export_deck_model.py and run in deck.js.

    logit P(win game) = base + w.x + sum_{i<j} x_i x_j <v_i, v_j> + F(x).w_F + aff(x) * w_aff

x is the deck's card counts with basics, w card power, v rank-4 pair vectors between nonland cards
(synergy), F the deck features (land count, curve, colours and splash, coloured-source shortfall) and
aff the deck's mean pairwise affinity from the set's PickNet `pool` model (sets without one, HOB, MOM,
TMT and WOE, go without). Reality Fracture (FRA) has no 17Lands game data yet, so no deck model: its
pools get the PickNet build only. It is fit on 17Lands game results with play/draw, mulligans, ranks, skill and
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

"Best deck by colours" runs the same search aimed at each colour combination: every pair of main colours
with no splash and with its best splash, and every three main colours. It lists the results by predicted
win rate, next to the "Build best deck" result, and clicking one loads it. The pickers above the list build
any main colours with no splash, the best splash, or a splash of a chosen colour. The search reaches a
target by penalising a deck for each card a main colour is short of four, and for a requested splash it
doesn't have. Colours outside the target are off limits, and a splash is one colour of at most three
cards. All 30 combinations take about a second, built one at a time in the background.

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
                                                   # the site runs ../picknet's frahob_freeze.pt: all_d256_ep3 with FRA and HOB fitted in
    python tools/build_data.py [SET ...]           # data/*.json (cards, Scryfall images, collation, context presets, mana facts)
    python tools/mana_facts.py [SET ...]           # just the mana facts in data/*.json, from ../mtg-data's Scryfall oracle dump
    python tools/export_deck_model.py [SET ...]    # data/deck/*.json, deck models (wheelnet's .venv; ~2 min/set)
    node tools/verify_deck.mjs [SET ...]           # deck.js vs PyTorch on training decks, plus a timed build
    ../wheelnet/.venv/bin/python tools/export_wheelnet.py SET [checkpoint.pt]
                                                   # model/wheelnet_SET.onnx + data/wheel/SET.json from a WheelNet trained
                                                   # for SET, per-set or multi-set (default: its newest wheelnet*.pt)
    node tools/check_wheelnet.mjs SET              # wheelnet.js's inputs vs training's, on the set's test drafts
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
