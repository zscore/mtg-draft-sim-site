"""Strip who-was-playing from MTG Arena logs so they can be kept in the repo as examples.

    python3 tools/scrub_arena_log.py OUT_DIR LOG [LOG ...]

Each LOG (Player.log, or an archived .log.gz from tools/archive_arena_logs.sh) is written to OUT_DIR as .log.gz with
  - account ids (userId / clientId, and every other mention of them, e.g. "Match to <id>") -> PLAYER...01, ...02
  - screen names (screenName / playerName)                                               -> Player01, Player02
  - login session ids (sessionId)                                                        -> 00000000-...-000000000001
  - the Steam ID and the /Users/<name> home folder                                       -> placeholders
using one mapping across all the logs, so the same player gets the same placeholder in each. Draft and game messages
(what arena.js reads) are left as they are. Prints what it replaced, and fails if anything it scrubbed is left over.
"""
import gzip
import re
import sys
from pathlib import Path

B32_ID = re.compile(r"\b[A-Z2-7]{26}\b")  # Arena account ids
KEYED = r'"(?:userId|clientId)"\s*:\s*"({})"'
NAME = re.compile(r'"(?:screenName|playerName)"\s*:\s*"([^"]+)"')
SESSION = re.compile(r'"sessionId"\s*:\s*"([0-9a-f-]{36})"')
STEAM = re.compile(r"\b7656119\d{10}\b")
HOME = re.compile(r"/Users/[^/\s\"']+")


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def mapping(texts):
    """Every identifier in the logs -> its placeholder, numbered in order of first appearance."""
    m = {}

    def add(values, fmt):
        for v in values:
            if v not in m:
                m[v] = fmt(sum(1 for x in m.values() if x.startswith(fmt(0)[:6])) + 1)

    for t in texts:
        add(B32_ID.findall(t), lambda i: f"PLAYER{i:020d}")
        add(re.findall(KEYED.format("[0-9A-F]{16}"), t), lambda i: f"B0700000{i:08d}")  # bots' hex ids
        add(NAME.findall(t), lambda i: f"Player{i:02d}")
        add(SESSION.findall(t), lambda i: f"00000000-0000-0000-0000-{i:012d}")
        add(STEAM.findall(t), lambda i: f"7656119{i:010d}")
    return m


def scrub(t, m):
    names = {k: v for k, v in m.items() if v.startswith("Player")}
    t = NAME.sub(lambda x: x.group(0).replace(f'"{x.group(1)}"', f'"{names[x.group(1)]}"'), t)
    others = sorted((k for k in m if k not in names), key=len, reverse=True)
    if others:
        t = re.sub("|".join(map(re.escape, others)), lambda x: m[x.group(0)], t)
    return HOME.sub("/Users/user", t)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in sys.argv[2:]]
    texts = [read(p) for p in paths]
    m = mapping(texts)
    kinds = {"PLAYER": "account ids", "B0700": "bot ids", "Player": "screen names", "000000": "session ids",
             "765611": "Steam ids"}
    for prefix, kind in kinds.items():
        print(f"{sum(v.startswith(prefix) for v in m.values()):3d} {kind}")
    for p, t in zip(paths, texts):
        s = scrub(t, m)
        # names are only replaced where they are named as one, so look for them there (a short one may be a word)
        left = [k for k, v in m.items() if (f'"{k}"' if v.startswith("Player") else k) in s]
        left += HOME.findall(s.replace("/Users/user", ""))
        if left:
            sys.exit(f"{p.name}: {len(left)} identifiers survived scrubbing")
        name = p.name if p.name.endswith(".gz") else p.name + ".gz"
        with gzip.open(out / name, "wt", encoding="utf-8", errors="surrogateescape", newline="", compresslevel=9) as f:
            f.write(s)
        print(f"{p.name} -> {out / name}")


if __name__ == "__main__":
    main()
