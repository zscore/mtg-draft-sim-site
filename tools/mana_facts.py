"""Mana facts for the deck builders' mana base (deck.js manaBase), from Scryfall's oracle cards.

    python tools/mana_facts.py [SET ...]      # patches data/{SET}.json in place; build_data.py calls add_mana_facts

Per card, only when set:
  pips   coloured pips of the (front face's) mana cost, {W: 2, U: 0.5, ...} (hybrid counts half to each colour,
         phyrexian full)
  makes  the colours a land taps for ("WU"; not filter or restricted mana), "?" for one colour chosen as it enters
         (Mirage Mesa), or "*" for a land that fetches a basic (Terramorphic Expanse)
  tapped a land that enters tapped, even only sometimes ("unless a player has 13 or less life"; not fast or shock lands)
  ramp   a permanent that taps for mana, or a spell that fetches a basic onto the battlefield (rocks, dorks, Cultivate)
  cycler landcycling or basic-typecycling (Swampcycling ...)
"""
import gzip
import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ORACLE = Path("/Users/zblanton/Documents/repos/mtg-data/data/raw/scryfall_oracle_cards.jsonl.gz")
COLORS = "WUBRG"
SKIP_LAYOUTS = {"token", "double_faced_token", "emblem", "art_series", "vanguard", "scheme", "planar"}


def oracle(names):
    """name -> Scryfall oracle card for each of `names` found (by full name or front face)."""
    want, out = set(names), {}
    with gzip.open(ORACLE, "rt") as f:
        for line in f:
            c = json.loads(line)
            if c.get("layout") in SKIP_LAYOUTS:
                continue
            keys = {c["name"]} | ({c["card_faces"][0]["name"]} if c.get("card_faces") else set())
            for k in keys & want:
                out.setdefault(k, c)
    return out


def pips(cost):
    out = dict.fromkeys(COLORS, 0.0)
    for sym in re.findall(r"\{([^}]+)\}", cost or ""):
        parts = [p for p in sym.split("/") if p in COLORS]
        for p in parts:
            out[p] += 1 / len(parts)
    return {k: round(v, 3) for k, v in out.items() if v}


def land_colors(text):
    """The colours a land taps for with {T} alone, unrestricted ("WU"). A filter land ({1}, {T}: Add one mana of any
    color) or one whose mana only casts some spells (Bucolic Ranch's Mounts) is no source for a deck's spells, though
    Scryfall's produced_mana lists all five colours for it. Basics' and typed duals' mana is in their reminder text."""
    made = set()
    for line in text.replace("(", "").replace(")", "").split("\n"):
        for m in re.finditer(r"(?:^|\. )\{t\}(?:, pay 1 life)?: add ([^.]*)\.(.*)", line):
            if "spend this mana only" in m.group(2) or " among " in m.group(1):
                continue
            made |= {k for k in COLORS if f"{{{k.lower()}}}" in m.group(1)}
            if re.search(r"any color|any one color", m.group(1)):
                made |= set(COLORS)
            elif "chosen color" in m.group(1):
                made.add("?")
    return "".join(k for k in COLORS if k in made) or ("?" if "?" in made else "")


def facts(c, type_line):
    faces = c.get("card_faces") or [c]
    text = " ".join(f.get("oracle_text", "") for f in faces).lower()
    text = re.sub(r"\([^)]*\)", "", text)  # reminder text: Treasure's "add one mana" is not the card's own
    out = {}
    front = type_line.split(" // ")[0]  # a spell // land card is a spell to the builders
    land = "Land" in front and "Creature" not in front
    if land:
        made = land_colors(" ".join(f.get("oracle_text", "") for f in faces).lower())
        if not made and re.search(r"search your library for an? basic land", text):
            made = "*"
        if made:
            out["makes"] = made
        if re.search(r"enters (the battlefield )?tapped(?! unless you control two or fewer)", text) and \
                "you may pay 2 life" not in text:  # fast lands and shock lands come in untapped early
            out["tapped"] = 1
        return out
    # the front face's cost: an adventure, omen or prepare half (and a split card's other half) is optional
    p = pips(faces[0].get("mana_cost") or c.get("mana_cost")) or pips(c.get("mana_cost"))
    if p:
        out["pips"] = p
    permanent = re.search(r"Artifact|Creature|Enchantment|Planeswalker|Battle", type_line)
    if permanent and re.search(r"\{t\}[^.:]*: add (\{|one mana|two mana|three mana|x mana|an amount)", text) or \
            re.search(r"search your library for (an? |up to \w+ )?basic land cards?,? (and )?put (it|them) onto the battlefield", text):
        out["ramp"] = 1
    if any(k == "Landcycling" or k.endswith("landcycling") or k[:-7] in ("Plains", "Island", "Swamp", "Mountain", "Forest")
           for k in c.get("keywords", []) if k.endswith("cycling")):
        out["cycler"] = 1
    return out


def add_mana_facts(cards, found=None):
    """cards: {id: {name, type, ...}} from data/{SET}.json, updated in place."""
    found = found or oracle({c["name"] for c in cards.values()} | {c["name"].split(" // ")[0] for c in cards.values()})
    for c in cards.values():
        for k in ("pips", "makes", "tapped", "ramp", "cycler"):
            c.pop(k, None)
        o = found.get(c["name"]) or found.get(c["name"].split(" // ")[0])
        if o:
            c.update(facts(o, c["type"]))


def main(sets):
    paths = [SITE / "data" / f"{s}.json" for s in sets] if sets else \
        [p for p in sorted((SITE / "data").glob("*.json")) if p.name != "sets.json"]
    data = {p: json.loads(p.read_text()) for p in paths}
    names = {n for d in data.values() for c in d["cards"].values() for n in (c["name"], c["name"].split(" // ")[0])}
    found = oracle(names)
    for p, d in data.items():
        add_mana_facts(d["cards"], found)
        p.write_text(json.dumps(d, separators=(",", ":")))
        cs = d["cards"].values()
        print(f"{d['code']}: {sum('makes' in c for c in cs)} lands with colours, {sum('ramp' in c for c in cs)} ramp, "
              f"{sum('cycler' in c for c in cs)} landcyclers, {sum(c['name'] not in found and c['name'].split(' // ')[0] not in found for c in cs)} not found")


if __name__ == "__main__":
    main(sys.argv[1:])
