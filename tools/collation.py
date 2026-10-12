"""MTG Arena Premier Draft pack collation, one entry per set: which cards can appear in which slot,
and how often. tools/build_data.py resolves each sheet's Scryfall query to the set's card ids and
writes the result into data/{SET}.json as "collation"; packs.js opens packs from it.

Rates come from Wizards first:
  - the set's "Collecting <Set>" article (paper Play/Draft Booster contents),
  - the MTG Arena drop-rates page (bonus-sheet rates for Arena limited packs),
  - the Play Booster Discord Q&A: on Arena "where paper has a non-foil wildcard slot and a foil
    wildcard slot, in MTG Arena there will be two non-foil wildcard slots". So FOIL below is the
    paper foil slot's rarity mix, drawn as a normal card.
Booster Fun printings (showcase, borderless, ...) are the same card on Arena, so their share of a
slot is folded into the base rarity. Rates given as "less than 1%" are dropped unless they are the
only route to a sheet.

Wizards does not document everything Arena does. Where 17Lands data (tools/verify_packs.py)
showed that Arena differs from the article, the entry says so and cites "17Lands"; those are
measured, not documented.

Two things no Wizards source covers, both measured from 17Lands:
  - colour balance of the commons (packs.js redraws them until the rule holds). Through Aetherdrift
    the five colours always show up ("cover"; a gold card counts for each colour). From Tarkir:
    Dragonstorm on there are also never more than two commons of one colour ("cap": 2), in some sets
    counting the wildcard slots' commons too ("all").
  - repeats: cards within a slot are always distinct (the 3 uncommons, say), and the wildcard never
    repeats a card already in the pack ("dedupe"), but the second wildcard (paper's foil slot) can.
    From Aetherdrift on (except TMT) nothing in a pack repeats ("unique").

Sheet specs: a Scryfall query (names are matched to the set's 17Lands card list), optionally
"minus" other sheets. Slot specs: (count, {sheet: weight}), weights per slot (normalised), or
per_card({...}) for weights per card that are multiplied by sheet size at build time.
"""

COLLECTING = "https://magic.wizards.com/en/news/feature/collecting-"
DROP_RATES = "https://magic.wizards.com/en/mtgarena/drop-rates"
PB_QA = "https://magic.wizards.com/en/news/announcements/play-booster-discord-q-and-a"


class per_card(dict):
    """Slot weights given per card: the sheet's weight is this times its size."""


def q(query, minus=()):
    return {"q": query, "minus": list(minus)}


def main(s, last):
    """The main set's commons..mythics up to collector number `last` (basics excluded)."""
    return {r: q(f"e:{s} r:{r[0].lower()} cn<={last} -t:basic") for r in ("C", "U", "R", "M")}


def cns(s, nums):
    return "e:%s (%s)" % (s, " or ".join(f"cn:{n}" for n in nums))


def basics(s):
    return q(f"e:{s} t:basic")


COVER = {"cover": True}  # colour rules, see the module docstring
CAP = {"cover": True, "cap": 2, "all": True}
RM_7 = per_card({"R": 2, "M": 1})  # each rare twice as likely as each mythic (1 mythic in ~7 packs)


def play_booster(code, last, *, commons, uncommons=3, wild, rare, foil, land=None, extra=(),
                 sheets=None, unique=False, balance=COVER, arena=None, sources=(), notes=()):
    """The Arena version of a paper Play Booster: `commons` is a list of (count, weights) common slots,
    then uncommons, any `extra` slots, a rare, the wildcard, the foil slot (non-foil on Arena) and,
    if Arena keeps it, the land slot. rare/wild/foil are the documented rates; `arena` replaces them
    with ones measured by tools/fit_rates.py where 17Lands packs reject the documented ones."""
    documented = dict(rare=rare, wild=wild, foil=foil)
    if arena:
        rare, wild, foil = arena.get("rare", rare), arena.get("wild", wild), arena.get("foil", foil)
        notes = [*notes, "Rare-slot and wildcard rarity rates are measured on Arena (tools/fit_rates.py, "
                         "likelihood-ratio test against the documented ones); the documented rates are kept "
                         "alongside."]
    sh = main(code.lower(), last)
    sh.update(sheets or {})
    slots = [dict(n=n, w=w, balance=True) for n, w in commons]
    slots += [dict(n=uncommons, w={"U": 1})] + [dict(n=n, w=w) for n, w in extra]
    slots += [dict(n=1, w=rare), dict(n=1, w=wild, dedupe=True), dict(n=1, w=foil)]
    if land:
        slots.append(dict(n=1, w=land))
    return dict(sheets=sh, layouts=[dict(w=1, slots=slots)], balance={"sheet": "C", **balance}, unique=unique,
                documented=documented,
                sources=[COLLECTING + sources[0], DROP_RATES, PB_QA, *sources[1:]], notes=list(notes))


SETS = {}

# --- Draft Booster era (before Play Boosters). Arena never had foils or tokens in these. ---------------

SETS["MOM"] = dict(
    # Paper: "1 Rare or mythic rare / 1 Multiverse Legends card / 1 Battle card / 1 Non-battle
    # double-faced card / 3-5 Uncommons / 8-9 Commons ... 1 Basic land". The battle and the DFC slot
    # take the rare/mythic's place when it is itself a battle or DFC, so the layout depends on the
    # rare: weights below are its per-card rare 2 : mythic 1 odds summed over each kind
    # (44 R + 10 M single-faced, 11 R + 5 M battles, 5 R + 5 M DFCs).
    sheets={
        "C": q("e:mom r:c cn<=266 -is:dfc"),
        "U": q("e:mom r:u -is:dfc -t:battle"),
        "UB": q("e:mom r:u t:battle"),
        "DC": q("e:mom r:c is:dfc"), "DU": q("e:mom r:u is:dfc -t:battle"),
        "R": q("e:mom r:r cn<=276 -is:dfc"), "M": q("e:mom r:m cn<=276 -is:dfc"),
        "RB": q("e:mom r:r t:battle"), "MB": q("e:mom r:m t:battle"),
        "RD": q("e:mom r:r is:dfc -t:battle"), "MD": q("e:mom r:m is:dfc -t:battle"),
        "L": q("e:mom cn>=267 cn<=276"), "B": basics("mom"),
        "MULU": q("e:mul cn<=65 r:u"), "MULR": q("e:mul cn<=65 r:r"), "MULM": q("e:mul cn<=65 r:m"),
    },
    layouts=[
        dict(w=98, slots=[dict(n=8, w={"C": 1}, balance=True), dict(n=2, w={"U": 1}), dict(n=1, w={"UB": 1}),
                          dict(n=1, w=per_card({"DC": 2, "DU": 1})), dict(n=1, w=per_card({"R": 2, "M": 1}))]),
        dict(w=27, slots=[dict(n=8, w={"C": 1}, balance=True), dict(n=3, w={"U": 1}),
                          dict(n=1, w=per_card({"DC": 2, "DU": 1})), dict(n=1, w=per_card({"RB": 2, "MB": 1}))]),
        dict(w=15, slots=[dict(n=8, w={"C": 1}, balance=True), dict(n=3, w={"U": 1}), dict(n=1, w={"UB": 1}),
                          dict(n=1, w=per_card({"RD": 2, "MD": 1}))]),
    ],
    # In every layout: one Multiverse Legends card ("Roughly one third of the time ... it will be a
    # rare or mythic rare") and the land.
    common=[dict(n=1, w={"MULU": 2 / 3, "MULR": 4 / 15, "MULM": 1 / 15}), dict(n=1, w={"L": 1, "B": 1})],
    balance={"sheet": "C", **COVER},
    sources=[COLLECTING + "march-of-the-machine", DROP_RATES,
             "https://magic.wizards.com/en/news/mtg-arena/mtg-arena-release-notes-march-of-the-machine"],
    notes=["Arena has no foil slot: 8 commons, not 8-9 with a foil in a third of packs (17Lands).",
           "Land slot is a common gain land or a basic, 50/50 (17Lands; paper lists only a basic).",
           "Multiverse Legends mythics: rare->mythic about 1:5 within the R/M third (17Lands; Wizards gives "
           "only the one-third R/M rate)."],
)

SETS["WOE"] = dict(
    # Paper: "1x Rare or Mythic Rare, 1x Enchanting Tales U/R/M, 3x Uncommons, 9x Commons*, 1x Basic
    # Land". Arena drop rates: "1 Enchanting Tales card of any rarity appears in each Wilds of Eldraine
    # booster pack ... Uncommons in this slot can upgrade to rares at approximately 1:3. Rares in this
    # slot can upgrade to mythics at a rate of approximately 1:5"; main rare->mythic "1:7".
    sheets={**main("woe", 261),
            "WOTU": q("e:wot cn<=63 r:u"), "WOTR": q("e:wot cn<=63 r:r"), "WOTM": q("e:wot cn<=63 r:m")},
    layouts=[dict(w=1, slots=[dict(n=9, w={"C": 1}, balance=True), dict(n=3, w={"U": 1}), dict(n=1, w=RM_7),
                              dict(n=1, w={"WOTU": 3 / 4, "WOTR": 1 / 5, "WOTM": 1 / 20})])],
    balance={"sheet": "C", **COVER},
    sources=[COLLECTING + "wilds-of-eldraine", DROP_RATES,
             "https://media.wizards.com/2023/images/daily/woe_draft_breakdown_1920x1080_en.jpg"],
    notes=["Arena has no basic land and no foil: 14 cards (17Lands).",
           "Enchanting Tales uncommon->rare is about 1:4 on Arena, not the drop-rates page's 1:3 "
           "(17Lands: U 0.748 / R 0.203 / M 0.049 per pack; 1:3 gives 0.667 / 0.267 / 0.067)."],
)

SETS["LCI"] = dict(
    # Paper: "1 Rare or mythic rare, 1 Double-faced common or uncommon, 3 Uncommons, 9 Commons*,
    # 1 Land (Non-Basic Cave 70% of the time / Core full-art basic land 30% of the time)". Arena drop
    # rates: rare->mythic "approximately 1:6.8" (= 64 rares x2 : 22 mythics x1).
    sheets={
        "C": q("e:lci r:c cn<=286 -is:dfc -t:cave -t:basic"),
        "U": q("e:lci r:u cn<=286 -is:dfc"),
        "R": q("e:lci r:r cn<=286"), "M": q("e:lci r:m cn<=286"),
        "DC": q("e:lci r:c is:dfc"), "DU": q("e:lci r:u is:dfc"),
        "L": q("e:lci r:c t:cave"), "B": basics("lci"),
    },
    layouts=[dict(w=1, slots=[dict(n=9, w={"C": 1}, balance=True), dict(n=1, w=per_card({"DC": 2, "DU": 1})),
                              dict(n=3, w={"U": 1}), dict(n=1, w=RM_7), dict(n=1, w=per_card({"L": 1, "B": 1}))])],
    balance={"sheet": "C", **COVER},
    sources=[COLLECTING + "the-lost-caverns-of-ixalan", DROP_RATES,
             "https://magic.wizards.com/en/news/mtg-arena/mtg-arena-release-notes-the-lost-caverns-of-ixalan"],
    notes=["Arena has no foil: 9 commons in every pack (17Lands).",
           "Land slot: each of the 7 Caves and 5 basics equally likely, 58/42, not paper's 70/30 (17Lands: "
           "0.579 Caves per pack)."],
)

# --- Play Boosters (MKM on). Arena: two non-foil wildcards, no art/token card. ------------------------

MKM_LIST_CU = ["Bishop of the Bloodstained", "Burden of Guilt", "Combine Chrysalis", "Consign // Oblivion",
               "Duskmantle, House of Shadow", "Enlisted Wurm", "Ghost Quarter", "Gnaw to the Bone",
               "Goblin Warchief", "Hard Evidence", "High Alert", "Krosan Tusker", "Kuldotha Rebirth",
               "Laid to Rest", "Leonin Relic-Warder", "Magmaw", "Maverick Thopterist",
               "Mentor of the Meek", "Metalspinner's Puzzleknot", "Millstone", "Mistveil Plains",
               "Mystery Key", "Nyx Weaver", "Putrid Warrior",
               "Quintorius, Field Historian", "Shard of Broken Glass", "Spell Snare", "Stromkirk Captain",
               "Syr Konrad, the Grim", "Treacherous Terrain"]
MKM_LIST_RM = ["Ixidor, Reality Sculptor", "Jace, Wielder of Mysteries", "Mass Hysteria", "Molten Psyche",
               "Monologue Tax", "Ranger-Captain of Eos", "Worldspine Wurm",
               # Arena swaps: "Baleful Mastery, Cavalier of Thorns, and Death Cloud ... Possibility
               # Storm, Evolutionary Leap, Smuggler's Copter instead"
               "Possibility Storm", "Evolutionary Leap", "Smuggler's Copter"]


def names(ns):
    return q(" or ".join(f'!"{n}"' for n in ns))


SETS["MKM"] = play_booster(
    "MKM", 271,
    # "1 in 8 Murders at Karlov Manor Play Boosters (12.5%) features a card from Special Guests or The
    # List instead of the seventh common card ... Special Guests ... 1.56% ... The List has 30 common
    # and uncommon reprints replacing a common ... 9.38% of the time and 10 rare and mythic rare
    # reprints ... 1.56% of the time."
    commons=[(6, {"C": 1}), (1, {"C": 87.5, "LCU": 9.38, "LRM": 1.56, "SPG": 1.56})],
    # "The 10 rare dual lands aren't found in this slot but instead are in the wildcard and traditional
    # foil wildcard slot." "1 in 6 Play Boosters will have 1 of the 10 rare dual lands in this slot."
    # Wizards gives no other wildcard/foil split; the rest is MTGJSON's (secondary).
    rare=RM_7,
    wild={"C": 16.92, "U": 51.33, "R": 12.93, "DUAL": 16.67, "M": 2.15},
    foil={"C": 58, "U": 32, "R": 7.5, "DUAL": 1.25, "M": 1.25},
    sheets={"R": q("e:mkm r:r cn<=271 -t:basic", minus=["DUAL"]),
            "DUAL": q("e:mkm r:r t:land cn>=259 cn<=271"),
            "SPG": q("e:spg cn>=19 cn<=28"), "LCU": names(MKM_LIST_CU), "LRM": names(MKM_LIST_RM)},
    arena=dict(rare={"R": 85.9, "M": 14.1}, wild={"C": 12.6, "U": 54.2, "R": 30.9 - 16.67, "DUAL": 16.67, "M": 2.3},
               foil={"C": 62.5, "U": 29.1, "R": 7.4 - 1.25, "DUAL": 1.25, "M": 1.1}),
    sources=["murders-at-karlov-manor",
             "https://magic.wizards.com/en/news/announcements/whats-new-on-the-list-for-murders-at-karlov-manor",
             "https://magic.wizards.com/en/news/mtg-arena/play-boosters-on-mtg-arena"],
    notes=["Arena drops the basic land slot: 13 cards (17Lands)."],
)

SETS["OTJ"] = play_booster(
    "OTJ", 271,
    # "5-6 Commons. 20% of Play Boosters contain a card from The List instead of a common ... You'll find
    # a non-foil Special Guests card in 1 out of 64 Play Boosters." (The List here = BIG + SPG.)
    commons=[(5, {"C": 1}), (1, {"C": 80, "BIG": 20 - 100 / 64, "SPG": 100 / 64})],
    # Arena drop rates: "1 Breaking News card of any rarity appears in each Outlaws of Thunder Junction
    # pack ... Uncommons in this slot can upgrade to rares at approximately 1:3. Rares in this slot
    # upgrade to mythics at a rate of approximately 1:5."
    extra=[(1, {"OTPU": 2 / 3, "OTPR": 4 / 15, "OTPM": 1 / 15})],
    rare=RM_7,
    # "In 1 out of 12 Play Boosters, the wildcard is a rare or mythic rare." C/U split not given.
    # MTGJSON's (secondary) C/U split.
    wild={"C": 65.6, "U": 26.1, "R": 100 / 12 * 6 / 7, "M": 100 / 12 / 7},
    # "1 Traditional foil card of any rarity ... can include Booster Fun cards and OTP cards". No split.
    foil={"C": 58, "U": 32, "R": 6.4, "M": 1.1, "OTPU": 1.67, "OTPR": 0.67, "OTPM": 0.17},
    # "Half of all Play Boosters contain a common dual land."
    land={"L": 50, "B": 50},
    sheets={"C": q("e:otj r:c cn<=271 -t:basic", minus=["L"]),
            "L": q(cns("otj", [251, 253, 255, 256, 257, 258, 259, 260, 261, 264])), "B": basics("otj"),
            "BIG": q("e:big cn<=30"), "SPG": q("e:spg cn>=29 cn<=38"),
            "OTPU": q("e:otp cn<=65 r:u"), "OTPR": q("e:otp cn<=65 r:r"), "OTPM": q("e:otp cn<=65 r:m")},
    arena=dict(rare={"R": 85.7, "M": 14.3}, wild={"C": 62.1, "U": 29.8, "R": 7.0, "M": 1.1},
               foil={"C": 58.6, "U": 28.8, "R": 7.1, "M": 1.2, "OTPU": 2.8, "OTPR": 1.1, "OTPM": 0.3}),
    balance={"cover": True, "min": 6},  # 17Lands: packs with a Big Score card (5 slot commons) are not balanced
    sources=["outlaws-of-thunder-junction"],
)

SETS["BLB"] = play_booster(
    "BLB", 261,
    # "1.5% of BLB Play Boosters contain a card from The List instead of a common ... The List for BLB
    # consists of cards from SPG"; Arena: "approximately 1:64".
    commons=[(6, {"C": 1}), (1, {"C": 63, "SPG": 1})],
    rare=RM_7,
    # Wizards gives no wildcard or foil split for BLB; MTGJSON's (secondary).
    wild={"C": 20, "U": 64.5, "R": 13.29, "M": 2.21},
    foil={"C": 58, "U": 32, "R": 8.57, "M": 1.43},
    sheets={"SPG": q("e:spg cn>=54 cn<=63")},
    arena=dict(rare={"R": 85.8, "M": 14.2}, wild={"C": 16.6, "U": 66.6, "R": 14.3, "M": 2.5}, foil={"C": 61.5, "U": 30.0, "R": 7.2, "M": 1.2}),
    sources=["bloomburrow"],
    notes=["Arena drops the basics-only land slot: 13 cards (17Lands)."],
)

SETS["DSK"] = play_booster(
    "DSK", 271,
    # "1 in 64 Play Boosters" have a Special Guests card in place of a common; Arena "approximately 1:64".
    commons=[(6, {"C": 1}), (1, {"C": 63, "SPG": 1})],
    # "60 rares (75%) and 20 mythic rares (12.6%)", "46 Booster Fun rares (8.2%) and 16 mythic rares
    # (1.4%)", "7 Lurking evil rares (2.5%) and 2 lurking evil mythic rares (0.3%)"
    rare={"R": 75 + 8.2 + 2.5, "M": 12.6 + 1.4 + 0.3},
    # The article gives no wildcard or foil split; MTGJSON's (secondary).
    wild={"C": 18.05, "U": 66.9, "R": 13.25, "M": 2.15},
    foil={"C": 58, "U": 32, "R": 8.6, "M": 1.4},
    # "5 full-art manor lands (13.3% non-foil, 3.3% traditional foil), 10 regular basic lands (26.7%
    # non-foil, 6.7% traditional foil), or one of 10 common dual lands (40% non-foil, 10% traditional foil)"
    land={"L": 50, "B": 50},
    sheets={"C": q("e:dsk r:c cn<=271 -t:basic", minus=["L"]),
            "L": q(cns("dsk", [255, 257, 258, 262, 263, 264, 265, 266, 267, 268])), "B": basics("dsk"),
            "SPG": q("e:spg cn>=64 cn<=73")},
    arena=dict(rare={"R": 85.7, "M": 14.3}, wild={"C": 16.6, "U": 66.7, "R": 14.4, "M": 2.3}, foil={"C": 59.2, "U": 32.6, "R": 6.9, "M": 1.3}),
    sources=["duskmourn"],
)

SETS["FDN"] = play_booster(
    "FDN", 271,
    # Special Guests in "1.5% of Play Boosters"; Arena "approximately 1:64".
    commons=[(6, {"C": 1}), (1, {"C": 63, "SPG": 1})],
    # "60 rares (78%) or 20 mythic rares (12.8%)... 43 borderless rares (7.7%) or 17 borderless mythic
    # rares (1.5%)"
    rare={"R": 78 + 7.7, "M": 12.8 + 1.5},
    # "common (16.7%), uncommon (58.3%), rare (16.3%), or mythic rare (2.6%)... borderless rare (1.6%)
    # or borderless mythic rares (0.3%)... 2 borderless commons (1.8%) or 8 borderless uncommons (2.4%)"
    wild={"C": 16.7 + 1.8, "U": 58.3 + 2.4, "R": 16.3 + 1.6, "M": 2.6 + 0.3},
    # No foil split in the article; MTGJSON's (secondary).
    foil={"C": 60, "U": 30, "R": 8.6, "M": 1.4},
    # "10 character lands (25%), 1 of 10 common dual lands (50%), or 1 of 10 regular frame basic lands (25%)"
    land={"L": 50, "B": 50},
    sheets={"C": q("e:fdn r:c cn<=271 -t:basic", minus=["L"]),
            "L": q(cns("fdn", [259, 260, 261, 263, 265, 266, 268, 269, 270, 271])), "B": basics("fdn"),
            "SPG": q("e:spg cn>=74 cn<=83")},
    arena=dict(rare={"R": 85.8, "M": 14.2}, wild={"C": 20.9, "U": 58.1, "R": 17.8, "M": 3.2}, foil={"C": 57.5, "U": 34.2, "R": 7.1, "M": 1.2}),
    sources=["foundations"],
)

SETS["DFT"] = play_booster(
    "DFT", 271,
    # Special Guests in 1.5% of Play Boosters; Arena "approximately 1:64".
    commons=[(6, {"C": 1}), (1, {"C": 63, "SPG": 1})],
    # "60 rares (78%) or 20 mythic rares (13%)... 45 borderless rare cards (8%)... 13 borderless mythic
    # rare cards (1%)"
    rare={"R": 78 + 8, "M": 13 + 1},
    # "common (8.3%), uncommon (62.5%), or rare or mythic rare (20.8%)... or a (8.3%) borderless revved
    # up common or uncommon card" (split evenly; R/M in the rare slot's proportion)
    wild={"C": 8.3 + 4.15, "U": 62.5 + 4.15, "R": 20.8 * 86 / 100, "M": 20.8 * 14 / 100},
    # "common (60.5%), uncommon (30%), rare (6.4%), or mythic rare (1.1%)... borderless ... common
    # (0.5%), uncommon (0.5%), rare (0.9%), or mythic rare (0.1%)"
    foil={"C": 60.5 + 0.5, "U": 30 + 0.5, "R": 6.4 + 0.9, "M": 1.1 + 0.1},
    # "common dual land 50%... default frame basic land 37.5%... full-art driver's seat basic land 12.5%"
    land={"L": 50, "B": 50},
    sheets={"C": q("e:dft r:c cn<=271 -t:basic", minus=["L"]),
            "L": q(cns("dft", [251, 252, 254, 256, 262, 263, 265, 266, 267, 271])), "B": basics("dft"),
            "SPG": q("e:spg cn>=84 cn<=93")},
    unique=True,
    arena=dict(rare={"R": 84.3, "M": 15.7}, wild={"C": 9.1, "U": 68.3, "R": 20.9, "M": 1.7}, foil={"C": 61.2, "U": 30.4, "R": 7.2, "M": 1.2}),
    sources=["aetherdrift"],
)

SETS["TDM"] = play_booster(
    "TDM", 271,
    # "In 1.5% of Play Boosters, 1 of these commons will be replaced with 1 of the 10 Special Guests";
    # Arena: "approximately 1:60".
    commons=[(6, {"C": 1}), (1, {"C": 59, "SPG": 1})],
    # Main set + showcase draconic + borderless clan/Saga/dragon, folded by rarity.
    rare={"R": 75 + 0.8 + 6.4 + 2.7 + 0.8, "M": 12.5 + 0.6 + 1.2 + 0.1 + 0.9},
    wild={"C": 12.5 + 4.6, "U": 58.3 + 3.8, "R": 15.6 + 0.2 + 1.3 + 0.6 + 0.2, "M": 2.6 + 0.1 + 0.2},
    foil={"C": 56.5 + 1.6, "U": 32 + 1.4, "R": 6.4 + 0.5 + 0.2 + 0.1, "M": 1.1 + 0.1},
    # "1 of 10 common two-color lands in non-foil (70.0%) or traditional foil (17.4%)"; basics the rest.
    land={"L": 83, "B": 17},
    sheets={"C": q("e:tdm r:c cn<=271 -t:basic", minus=["L"]),
            "L": q(cns("tdm", [250, 251, 254, 258, 265, 267, 268, 269, 270, 271])), "B": basics("tdm"),
            "SPG": q("e:spg cn>=104 cn<=113")},
    unique=True, balance=CAP,
    arena=dict(rare={"R": 85.6, "M": 14.4}, wild={"C": 18.4, "U": 59.1, "R": 19.3, "M": 3.2}, foil={"C": 58.9, "U": 33.3, "R": 6.7, "M": 1.2}),
    sources=["tarkir-dragonstorm"],
    notes=["Land slot is 83% dual / 17% basic on Arena, not paper's 87.4/13.1 (17Lands: 0.829 duals per pack)."],
)

SETS["FIN"] = play_booster(
    "FIN", 293,
    # "In one third of Play Boosters, one of these commons will be replaced by a card from the FINAL
    # FANTASY Through the Ages bonus sheet ... an uncommon (63.25%), rare (29.75%), or mythic rare (7%)"
    # Arena drop rates: "approximately 1:3, replacing a common slot. Uncommons in this slot can upgrade
    # to rares at approximately 1:3.6. Rares in this slot upgrade to mythics at a rate of approximately
    # 1:3" -> U 72.2 / R 18.5 / M 9.3, which is what 17Lands shows (U 71.7 / R 18.2 / M 10.0).
    commons=[(6, {"C": 1}), (1, {"C": 2 / 3, "FCAU": (1 - 1 / 3.6) / 3, "FCAR": 2 / 3 / 3.6 / 3, "FCAM": 1 / 3 / 3.6 / 3})],
    # "Default frame rare (80%) or mythic rare (10%) / Borderless rare (8%) or mythic rare (1%) /
    # FINAL FANTASY artist rare (0.5%) or mythic rare (0.5%)"
    rare={"R": 88.5, "M": 11.5},
    # "a common (16.7%), uncommon (58.3%) ... a borderless woodblock common (2.6%), borderless woodblock
    # or borderless character uncommon (5.7%), or rare or mythic rare (16.7%; same proportion as below)"
    wild={"C": 16.7 + 2.6, "U": 58.3 + 5.7, "R": 16.7 * 0.885, "M": 16.7 * 0.115},
    foil={"C": 55.75 + 0.1, "U": 35.9 + 0.5 + 0.25, "R": 5.5 + 1, "M": 0.75 + 0.25},
    # "Common two-color land (55%) or basic land (45%)"
    land={"L": 2, "B": 1},
    sheets={"C": q("e:fin r:c cn<=293 -t:basic", minus=["L"]),
            "L": q(cns("fin", [273, 278, 280, 281, 282, 287, 288, 290, 291, 292])),
            "B": q("e:fin t:basic -t:snow -!Wastes"),
            "FCAU": q("e:fca cn<=64 r:u"), "FCAR": q("e:fca cn<=64 r:r"), "FCAM": q("e:fca cn<=64 r:m")},
    unique=True,
    balance=CAP,
    arena=dict(rare={"R": 88.2, "M": 11.8}, wild={"C": 18.1, "U": 63.7, "R": 16.1, "M": 2.2}, foil={"C": 56.3, "U": 36.3, "R": 6.4, "M": 0.9}),
    sources=["final-fantasy"],
    notes=["Land slot is 2/3 dual, 1/3 basic on Arena, not paper's 55/45 (17Lands: 0.666 duals per pack); "
           "Wastes never appears."],
)

SETS["EOE"] = play_booster(
    "EOE", 261,
    # "In 1.8% of Play Boosters, 1 of 10 non-foil Special Guests cards will replace a common"; Arena
    # "approximately 1:55".
    commons=[(6, {"C": 1}), (1, {"C": 54, "SPG": 1})],
    rare={"R": 80.4 + 2 + 2, "M": 14.2},
    # "A common (12.5%), uncommon (62.5%), rare (10.6%) ... A rare (10%) or mythic rare (2.5%) Stellar
    # Sights land / A rare (1%) ... borderless viewport land"
    wild={"C": 12.5, "U": 62.5, "R": 10.6 + 1, "M": 0.5, "EOSR": 10, "EOSM": 2.5},
    foil={"C": 58, "U": 32, "R": 6.4, "M": 1.1, "EOSR": 1, "EOSM": 0.25},
    sheets={"SPG": q("e:spg cn>=119 cn<=128"), "EOSR": q("e:eos cn<=45 r:r"), "EOSM": q("e:eos cn<=45 r:m")},
    unique=True,
    balance=CAP,
    arena=dict(rare={"R": 84.8, "M": 15.2}, wild={"C": 12.6, "U": 62.5, "R": 11.7, "M": 0.7, "EOSR": 10.1, "EOSM": 2.4},
               foil={"C": 57.8, "U": 32.2, "R": 6.6, "M": 1.0, "EOSR": 1.8, "EOSM": 0.6}),
    sources=["edge-of-eternities"],
    notes=["Arena drops the basics-only land slot: 13 cards (17Lands)."],
)

SETS["TLA"] = play_booster(
    "TLA", 281,
    # "In 1 of 26 Play Boosters, 1 of 61 non-foil source material cards will replace a common."
    commons=[(6, {"C": 1}), (1, {"C": 25, "TLE": 1})],
    rare={"R": 80 + 1.6 + 1.4 + 2.4, "M": 12.6},
    wild={"C": 4.2, "U": 74.1, "R": 16.7, "M": 2.6},
    foil={"C": 53.9, "U": 36.7, "R": 6.7, "M": 1.2},
    # common dual land 40% + 10% foil; basics (default, Avatar's journey, Appa) the other 50%
    land={"L": 50, "B": 50},
    sheets={"C": q("e:tla r:c cn<=281 -t:basic", minus=["L"]),
            "L": q(cns("tla", [265, 267, 269, 271, 272, 273, 274, 275, 279, 280])), "B": basics("tla"),
            "TLE": q("e:tle cn<=61")},
    unique=True,
    balance=CAP,
    arena=dict(rare={"R": 85.7, "M": 14.3}, wild={"C": 4.2, "U": 74.8, "R": 18.0, "M": 2.9}, foil={"C": 57.1, "U": 28.8, "R": 9.3, "M": 4.9}),
    sources=["avatar-the-last-airbender"],
)

SETS["ECL"] = play_booster(
    "ECL", 268,
    # "In 1 of 55 Play Boosters, 1 of 20 Special Guests cards will replace a common."
    commons=[(6, {"C": 1}), (1, {"C": 54, "SPG": 1})],
    rare={"R": 78.2 + 4.6, "M": 13.6 + 1.2},
    wild={"C": 18, "U": 58 + 2, "R": 19 + 1, "M": 2},
    foil={"C": 60.4, "U": 29.8 + 1, "R": 6.5 + 1, "M": 1.1},
    sheets={"SPG": q("e:spg cn>=129 cn<=148")},
    unique=True,
    balance={"cover": True, "cap": 2},
    arena=dict(rare={"R": 84.6, "M": 15.4}, wild={"C": 17.9, "U": 59.9, "R": 20.0, "M": 2.2}, foil={"C": 62.5, "U": 24.8, "R": 8.2, "M": 4.4}),
    sources=["lorwyn-eclipsed"],
    notes=["Arena drops the basics-only land slot: 13 cards (17Lands)."],
)

SETS["SOS"] = play_booster(
    "SOS", 266,
    # "5-6 Common cards ... In 1 of 55 Play Boosters, 1 of 10 Special Guests cards will replace a common."
    commons=[(5, {"C": 1}), (1, {"C": 54, "SPG": 1})],
    # "1 Mystical Archive card: An uncommon (87.5%), rare (9.6%), or mythic rare (2.9%)"
    extra=[(1, {"SOAU": 87.5, "SOAR": 9.6, "SOAM": 2.9})],
    # main rare 82.5 / mythic 14.1, field notes rare 1.2, borderless Elder Dragon/PW/dual 1.6 (split evenly)
    rare={"R": 82.5 + 1.2 + 0.8, "M": 14.1 + 0.8},
    wild={"C": 39.1, "U": 39.1, "R": 19.5, "M": 1.9},
    foil={"C": 54.4, "U": 33.6, "R": 6.7, "M": 1.1, "SOAU": 2.8},
    # default basic 26.7 + 6.7, spellcraft basic 13.3 + 3.3, common dual 40 + 10
    land={"L": 50, "B": 50},
    sheets={"C": q("e:sos r:c cn<=266 -t:basic", minus=["L"]),
            "L": q(cns("sos", [255, 256, 258, 262, 266])), "B": basics("sos"),
            # Arena swaps Library of Alexandria (SPG 158a) in for Library of Leng
            "SPG": q("e:spg cn>=149 cn<=158"),
            "SOAU": q("e:soa cn<=65 r:u"), "SOAR": q("e:soa cn<=65 r:r"), "SOAM": q("e:soa cn<=65 r:m")},
    unique=True,
    balance={"cap": 1},
    arena=dict(rare={"R": 84.5, "M": 15.5}, wild={"C": 38.9, "U": 39.4, "R": 20.0, "M": 1.7},
               foil={"C": 54.2, "U": 33.6, "R": 7.2, "M": 1.3, "SOAU": 2.8, "SOAR": 0.7, "SOAM": 0.2}),
    sources=["secrets-of-strixhaven"],
)

SETS["MSH"] = play_booster(
    "MSH", 276,
    # "In 1 out of 24 Play Boosters, 1 of 60 non-foil source material cards will replace a common."
    commons=[(6, {"C": 1}), (1, {"C": 23, "MAR": 1})],
    rare={"R": 76.1 + 1 + 2.4 + 1.9, "M": 16.6 + 0.8 + 0.7},
    wild={"C": 12.4, "U": 64 + 2.7, "R": 16.8 + 1, "M": 2.1},
    foil={"C": 59.9, "U": 29.6 + 1.2, "R": 5.9, "M": 1.2},
    # common dual 40 + 10; default, city calm and city chaos basics the other 50
    land={"L": 50, "B": 50},
    sheets={"C": q("e:msh r:c cn<=276 -t:basic", minus=["L"]),
            "L": q(cns("msh", [257, 258, 259, 262, 265, 268, 270, 271, 272, 273])), "B": basics("msh"),
            "MAR": q("e:mar cn>=41 cn<=100")},
    unique=True,
    balance={"cover": True, "cap": 2},
    arena=dict(rare={"R": 81.3, "M": 18.7}, wild={"C": 12.3, "U": 66.6, "R": 18.9, "M": 2.2}, foil={"C": 60.6, "U": 31.1, "R": 6.9, "M": 1.4}),
    sources=["marvel-super-heroes", "https://magic.wizards.com/en/news/mtg-arena/marvel-super-heroes"],
)

TMT_TURTLES = {"TC": [10, 14, 37, 104, 118], "TU": [16, 38, 101, 120], "TR": [15, 35, 103, 121],
               "TM": [17, 36, 102, 119]}
_tmt = {k: q(cns("tmt", v)) for k, v in TMT_TURTLES.items()}


def tmt(w):
    """Rarity weights over the whole main set, Turtles included: split by card count (56 commons +
    5 Turtle commons, 51 + 4 uncommons, 49 + 4 rares, 11 + 4 mythics). On Arena Turtles show up in
    the second wildcard (paper's foil slot) but not the first wildcard or the rare slot (17Lands: 8.1%
    of packs have two Turtles; Turtles in the second wildcard alone predict 8.2%)."""
    n = {"C": (56, 5), "U": (51, 4), "R": (49, 4), "M": (11, 4)}
    out = {}
    for r, x in w.items():
        a, b = n[r]
        out[r], out["T" + r] = x * a / (a + b), x * b / (a + b)
    return out
SETS["TMT"] = play_booster(
    "TMT", 190,
    # "In 1 out of 28 Play Boosters, 1 of 20 non-foil source material cards will replace a common."
    commons=[(6, {"C": 1}), (1, {"C": 27, "PZA": 1})],
    uncommons=2,
    # "1 Legendary Turtle card of any rarity: There are 5 common (64.1%), 4 uncommon (13.1%), 4 rare
    # (6.9%), and 4 mythic rare (3.4%) ... 1 common (4.6%), 4 uncommon (6.5%), and 4 rare (1%) scene
    # cards ... 4 mythic rare silhouette cards (less than 1%)"
    extra=[(1, {"TC": 64.1 + 4.6, "TU": 13.1 + 6.5, "TR": 6.9 + 1, "TM": 3.4 + 0.5})],
    rare={"R": 84.9 + 4.1, "M": 9.2},
    wild={"C": 24.9, "U": 62.1 + 1.3 + 2.6, "R": 7.6, "M": 0.5},
    foil={"C": 61.5, "U": 27.7, "R": 6.7, "M": 0.5},
    # "A non-foil (48%) or traditional foil (12%) common dual land / non-foil (32%) or traditional foil
    # (8%) rooftop basic land" (60/40); Arena: 55/45, see notes
    land={"L": 55.3, "B": 44.7},
    sheets={**_tmt,
            "C": q("e:tmt r:c cn<=190 -t:basic", minus=["L", "TC"]),
            "U": q("e:tmt r:u cn<=190", minus=["TU"]),
            "R": q("e:tmt r:r cn<=190", minus=["TR"]),
            "M": q("e:tmt r:m cn<=190", minus=["TM"]),
            # the five common duals and Escape Tunnel ("6 common dual lands")
            "L": q(cns("tmt", [183, 184, 185, 186, 187, 189])), "B": basics("tmt"),
            "PZA": q("e:pza cn<=20")},
    balance={"cover": True, "cap": 2},
    arena=dict(rare={"R": 86.3, "M": 13.7}, wild={"C": 25.2, "U": 65.6, "R": 8.2, "M": 1.0},
               foil=tmt({"C": 61.9, "U": 29.4, "R": 7.1, "M": 1.7})),
    notes=["Land slot is 55/45 dual/basic on Arena, not paper's 60/40 (17Lands: 0.553 duals per pack)."],
    sources=["teenage-mutant-ninja-turtles"],
)

SETS["HOB"] = play_booster(
    "HOB", 188,
    # "7 Common cards" (60 commons; scene cards are the same cards on Arena), "3 Uncommon cards"
    commons=[(7, {"C": 1})],
    # "A rare (82.9%) or mythic rare (11.1%) card from the main set", plus rare scene (1.8%) and Dragon
    # hoard (2.3%) printings
    rare={"R": 82.9 + 1.8 + 2.3, "M": 11.1},
    # "A common (74.17%), uncommon (3.9%), rare (17.3%), or mythic rare (2.3%)": mostly commons, unlike other
    # Play Boosters' wildcards, and 17Lands agrees.
    wild={"C": 74.17, "U": 3.9, "R": 17.3, "M": 2.3},
    foil={"C": 59.8, "U": 29.3, "R": 7.1, "M": 1},
    # default and Middle-earth journey basics 50% (non-foil and foil), "common dual land" 50%
    land={"L": 50, "B": 50},
    sheets={"C": q("e:hob r:c cn<=188 -t:basic", minus=["L"]),
            "L": q(cns("hob", [182, 183, 185, 186, 188])), "B": basics("hob")},
    unique=True,
    balance={"cover": True, "cap": 2},
    arena=dict(rare={"R": 87.5, "M": 12.5}, wild={"C": 75.1, "U": 4.1, "R": 18.1, "M": 2.6},
               foil={"C": 60.6, "U": 30.6, "R": 7.7, "M": 1.1}),
    sources=["the-hobbit"],
)

# Reality Fracture's echoed pairs: two versions of one character (or the "Way of the" mirror spells), same
# rarity. Paired from 17Lands packs: a card shows up with its partner 6-8 times as often as with any other.
FRA_PAIRS = [
    ('Ajani Resolute', 'Ajani Unrelenting'),
    ('Arni, Renowned Champion', 'Arni, Humble Scribe'),
    ('Chandra, Chill of Compliance', 'Chandra, Torch of Defiance'),
    ('Danitha, Sword of Hope', 'Danitha, Spear of Agony'),
    ('Edgar, Moonlit Sovereign', 'Edgar, Ancient Bloodlord'),
    ('Fblthp, Knows the Way', 'Fblthp, Impossibly Lost'),
    ('Gallia, Tragic Host', 'Gallia, the Merrymaker'),
    ('Garruk, Veiled Butcher', 'Garruk, Curse Breaker'),
    ('Geist of Saint Thalia', 'Thalia, the Survivor'),
    ('Ghalta the Unstoppable', 'Ghalta the Immovable'),
    ("Gideon's Memorial", 'Gideon the Oathless'),
    ('Hapatra, the Desert Frost', 'Hapatra, the Desert Fang'),
    ('Jiang Yanggu, Alone', 'Jiang Yanggu, Never Alone'),
    ('Karn, Argent Defender', 'Karn, Gilded Guardian'),
    ('Kiora of Fire and Ashes', 'Kiora of Salt and Sand'),
    ('Koth, the Geomancer', 'Koth of the Homestead'),
    ('Liliana the Repentant', 'Liliana the Faultless'),
    ('Loot, the Anomaly', 'Loot, the Nexus'),
    ('Lyra, Archangel of Dawn', 'Lyra, Tolarian Archangel'),
    ('Mabel, Valley Hero', 'Mabel, Bitter Recluse'),
    ('Marwyn, the Preserver', 'Marwyn, the Clearcutter'),
    ('Massacre Girl, Most Wanted', 'Rescue Girl, First Responder'),
    ('Pia, Aether Ascetic', 'Pia, Determined Rebuilder'),
    ('Proft, Consulting Detective', 'Proft, Sinister Mastermind'),
    ('Ruric Thar, Biomagus', 'Ruric Thar, Magecrusher'),
    ('Saheeli, Consul of Oversight', 'Saheeli, Jewel of Avishkar'),
    ('Samut, Tyrant of Naktamun', "Samut, Hazoret's Champion"),
    ('Tam, the Possibility', 'Jace, Reality Sculptor'),
    ('Tetsuko Umezawa, Fugitive', 'Tetsuko Umezawa, Pursuer'),
    ('Teyo, Diamondblade Mage', 'Teyo, Lightshield Expert'),
    ('Tinybones, Pocket Nuisance', 'Titanbones, Towering Heart'),
    ('Tomik, Izzet Sparkmage', 'Tomik, Orzhov Lawmage'),
    ('Traxos, Scourge Eternal', 'Traxos, Academy Guardian'),
    ('Vraska, Soul of Stone', 'Vraska, the Cutting Glare'),
    ('Way of the Cryomancer', 'Way of the Pyromancer'),
    ('Way of the Deathbringer', 'Way of the Wildspeaker'),
    ('Way of the Healer', 'Way of the Necromancer'),
    ('Way of the Mind Sculptor', 'Way of the Paradox'),
    ('Way of the Warlord', 'Way of the Mentor'),
    ('Winter, Tormented Loner', 'Winter, Team Player'),
    ('Yargle, Goliath of Otaria', 'Yargle, Glutton of Urborg'),
    ('Yoshimaru, Scrappy Stray', 'Yoshimaru, Beloved Companion'),
    ('Yuriko, Blade of the Mighty', 'Yuriko, Hope from the Shadows'),
]
EP = {"EPU": 91, "EPR": 7.5, "EPM": 1.6}

SETS["FRA"] = dict(
    # Paper: "Commons (5-6) ... In 1 of 55 Play Boosters, 1 of 10 Special Guests cards will replace a common",
    # "Uncommon", "Common (23%) or uncommon (74%)", "Echoed pairs cards (3): 66 uncommon (88.9%), 14 rare
    # (6.7%), 6 mythic rare (1.4%) ... Must include 2 cards from same echoed pair", "Rare (74.4%+) or mythic
    # rare (14.8%+)", a traditional foil of any rarity, and the land: default or tower basic (45.5%) or
    # "common dual land" (54.5%).
    # Arena (17Lands, 31k P1P1 packs): the foil slot is a non-foil card of any rarity, and the rates below are
    # measured. Every pack has 6 commons (a Special Guest in about 1 in 58), 1 uncommon, the common/uncommon
    # slot, an echoed pair plus one more echoed card, 1 rare or mythic and the land; the any-rarity slot is a
    # common (48.7%), uncommon (16.9%), echoed card (27.1%) or rare/mythic (6.7%).
    sheets={**main("fra", 194),
            "C": q("e:fra r:c cn<=194 -t:basic", minus=["L"]),
            "L": q(cns("fra", [175, 177, 178, 182, 183, 184, 190, 192, 193, 194])), "B": basics("fra"),
            "EPU": q("e:fra cn>=195 cn<=280 r:u"), "EPR": q("e:fra cn>=195 cn<=280 r:r"),
            "EPM": q("e:fra cn>=195 cn<=280 r:m"),
            "SPG": q("e:spg cn>=159 cn<=168")},
    layouts=[dict(w=1, slots=[
        dict(n=5, w={"C": 1}, balance=True), dict(n=1, w={"C": 57, "SPG": 1}, balance=True),
        dict(n=1, w={"U": 1}),
        dict(n=1, w={"C": 23.5, "U": 76.5}, dedupe=True),
        dict(n=1, w=EP, pair=True),  # two cards: one and its partner
        dict(n=1, w=EP, dedupe=True),
        dict(n=1, w={"R": 83.7, "M": 16.3}),
        dict(n=1, w={"C": 48.7, "U": 16.9, "R": 5.6, "M": 1.1,
                     **{k: 27.1 * v / sum(EP.values()) for k, v in EP.items()}}, dedupe=True),
        dict(n=1, w={"L": 54.7, "B": 45.3}),
    ])],
    pairs=FRA_PAIRS,
    balance={"sheet": "C", "cover": True, "cap": 2},
    sources=[COLLECTING + "reality-fracture", DROP_RATES],
    notes=["All slot rates are measured on Arena from 17Lands packs (the paper foil slot is a non-foil card of "
           "any rarity, a quarter of the time an echoed card)."],
)
