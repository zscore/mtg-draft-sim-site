"""Local static server for the draft page.   python3 serve.py [port]

Sends COOP/COEP headers so the page is cross-origin isolated, which lets onnxruntime-web use
WebAssembly threads (COEP "credentialless" still allows Scryfall card images). Browsers without
credentialless support (Safari) fall back to one thread.

It also serves MTG Arena's log for the page's Live tab (only to this machine):
  /api/arena/status                     where the log and Arena's card database are, and whether detailed logs are on
  /api/arena/history                    Arena's per-session logs (UTC_Log - *.log, kept in its install folder for a few
                                        weeks): [{name, size, mtime}], oldest first
  /api/arena/log?file=cur|prev|<session log name>&from=N&id=X
                                        the log's draft-related entries from byte N on: {id, size, next, reset, entries:
                                        [{at, text}]}. The last entry may still be growing, so `next` is its start and it is
                                        sent again next time. A different file id (Arena restarted) starts over at 0.
  /api/arena/cards?ids=1,2              Arena card ids (grpIds) -> {name, set}, from Arena's card database
Arena writes the log only with Options > Account > Detailed Logs (Plugin Support) on. Set MTGA_LOG_DIR,
MTGA_CARD_DB or MTGA_HISTORY_DIR if Arena lives somewhere unusual.
"""
import glob
import http.server
import json
import os
import re
import sqlite3
import sys
import urllib.parse
from pathlib import Path

HOME = Path.home()
LOG_DIRS = [os.environ.get("MTGA_LOG_DIR"), HOME / "Library/Logs/Wizards Of The Coast/MTGA",
            HOME / "AppData/LocalLow/Wizards Of The Coast/MTGA"]
RAW = "MTGA_Data/Downloads/Raw/Raw_CardDatabase_*.mtga"
CARD_DBS = [os.environ.get("MTGA_CARD_DB"),
            str(HOME / "Library/Application Support/com.wizards.mtga/Downloads/Raw/Raw_CardDatabase_*.mtga"),
            str(HOME / "Library/Application Support/Steam/steamapps/common/MTGA" / RAW),
            "C:/Program Files/Wizards of the Coast/MTGA/" + RAW,
            "C:/Program Files (x86)/Steam/steamapps/common/MTGA/" + RAW,
            "C:/Program Files/Epic Games/MagicTheGathering/" + RAW]
# an entry starts at a logger tag (session logs put a frame number first)
ENTRY = re.compile(rb"^(?:\[\d+\] )?\[(?:UnityCrossThreadLogger|Client GRE)\]", re.M)
KEYS = (b"PackCards", b"DraftPack", b"DraftStatus", b"PickGrpId", b"PickInfo", b"MakePick", b"EventJoin", b"Event_Join",
        b"CardPool")


def log_path(which):
    name = "Player-prev.log" if which == "prev" else "Player.log"
    for d in LOG_DIRS:
        if d and (Path(d) / name).exists():
            return Path(d) / name
    return None


def history_logs():
    """Arena's session logs: MTGA_Data/Logs/Logs next to each install's card database."""
    dirs = [os.environ.get("MTGA_HISTORY_DIR")]
    for pat in CARD_DBS:
        for db in glob.glob(pat) if pat else []:
            dirs.append(str(Path(db).parents[2] / "Logs" / "Logs"))
    found = {Path(p).name: Path(p) for d in dirs if d for p in glob.glob(str(Path(d) / "UTC_Log*.log"))}
    return sorted(found.values(), key=lambda p: p.stat().st_mtime)


def card_db():
    found = [p for pat in CARD_DBS if pat for p in glob.glob(pat)]
    return max(found, key=os.path.getmtime) if found else None


def slim(chunk):
    """An entry listing events (Courses) runs to a megabyte of decks; keep the fields arena.js reads."""
    if b"CardPool" not in chunk or len(chunk) < 4096:
        return chunk
    i = chunk.find(b"{")
    eol = chunk.find(b"\n", i)
    try:
        o = json.loads(chunk[i:eol if eol >= 0 else None])
    except ValueError:
        return chunk
    keep = lambda c: {k: c.get(k) for k in ("InternalEventName", "CurrentModule", "CardPool", "DraftId")}
    if isinstance(o, dict) and isinstance(o.get("Courses"), list):
        o = {"Courses": [keep(c) for c in o["Courses"] if isinstance(c, dict) and c.get("CardPool")]}
    elif isinstance(o, dict) and "CardPool" in o:
        o = keep(o)
    else:
        return chunk
    return chunk[:i] + json.dumps(o, separators=(",", ":")).encode()


def read_log(path, start, file_id):
    st = path.stat()
    fid = f"{st.st_ino}-{int(st.st_birthtime if hasattr(st, 'st_birthtime') else st.st_ctime)}"
    reset = fid != file_id or start > st.st_size
    if reset:
        start = 0
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read()
    starts = [m.start() for m in ENTRY.finditer(data)]
    entries = []
    for i, s in enumerate(starts):
        chunk = data[s:starts[i + 1] if i + 1 < len(starts) else len(data)]
        if any(k in chunk for k in KEYS):
            entries.append({"at": start + s, "text": ENTRY.sub(b"", slim(chunk), 1).decode("utf-8", "replace")})
    return {"id": fid, "size": st.st_size, "next": start + (starts[-1] if starts else 0), "reset": reset,
            "entries": entries}


def card_names(ids):
    db = card_db()
    if not db or not ids:
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = ("select c.GrpId, l.Loc, c.ExpansionCode from Cards c join Localizations_enUS l on l.LocId = c.TitleId "
             f"where c.GrpId in ({','.join('?' * len(ids))})")
        return {str(g): {"name": re.sub(r"<[^>]+>", "", name), "set": code} for g, name, code in con.execute(q, ids)}
    finally:
        con.close()


def detailed_logs(path):
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    return True if b"DETAILED LOGS: ENABLED" in head else False if b"DETAILED LOGS: DISABLED" in head else None


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.path.startswith("/api/arena/"):
            return super().do_GET()
        # the log is this machine's business: no other machines, and no pages reaching it by another host name
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if self.client_address[0] not in ("127.0.0.1", "::1") or host not in ("localhost", "127.0.0.1", "[::1]"):
            return self.send_error(403)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if url.path == "/api/arena/status":
                cur = log_path("cur")
                body = {"log": str(cur) if cur else None, "cardDb": card_db(),
                        "detailed": detailed_logs(cur) if cur else None}
            elif url.path == "/api/arena/history":
                body = [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in history_logs()]
            elif url.path == "/api/arena/log":
                which = q.get("file", "cur")
                path = log_path(which) if which in ("cur", "prev") else \
                    next((p for p in history_logs() if p.name == which), None)  # only a listed log, never a path
                if not path:
                    return self.send_error(404, "No Arena log found")
                body = read_log(path, int(q.get("from", 0)), q.get("id", ""))
            elif url.path == "/api/arena/cards":
                body = card_names([int(x) for x in q.get("ids", "").split(",") if x.strip().isdigit()])
            else:
                return self.send_error(404)
        except (OSError, sqlite3.Error, ValueError) as e:
            return self.send_error(500, str(e))
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/arena/"):  # the Live tab polls every second; keep the console readable
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}")
    http.server.ThreadingHTTPServer(("", port), Handler).serve_forever()
