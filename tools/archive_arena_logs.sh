#!/bin/bash
# Archive MTG Arena's Player.log / Player-prev.log before Arena overwrites them.
#
# Arena truncates Player.log on every launch and keeps only one previous
# session in Player-prev.log, so anything older is lost. This copies each
# session into ARCHIVE_DIR as a gzipped file named after the session start.
# Safe to run as often as you like: a session already archived is only
# re-copied when the live log has grown since (Arena still running).
#
#   tools/archive_arena_logs.sh              archive now
#   tools/archive_arena_logs.sh --install    also run every 15 min via launchd
#   tools/archive_arena_logs.sh --uninstall  remove the launchd job
#
# launchd jobs may not touch ~/Documents (macOS privacy protection), so the
# archive lives in ~/mtga-log-archive and --install runs a copy of this
# script from ~/Library/Application Support. Re-run --install after editing.
#
# Read an archived log with: zless ~/mtga-log-archive/<file>.log.gz
set -euo pipefail

SRC="${MTGA_LOG_DIR:-$HOME/Library/Logs/Wizards Of The Coast/MTGA}"
ARCHIVE_DIR="${MTGA_ARCHIVE_DIR:-$HOME/mtga-log-archive}"
INDEX="$ARCHIVE_DIR/index.tsv"   # session key, archived bytes, file name
LABEL="com.zblanton.mtga-log-archive"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
INSTALLED="$HOME/Library/Application Support/mtga-log-archive/archive_arena_logs.sh"

case "${1:-}" in
  --install)
    mkdir -p "$ARCHIVE_DIR" "$(dirname "$INSTALLED")"
    cp "$0" "$INSTALLED"
    script="$INSTALLED"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$script</string></array>
  <key>StartInterval</key><integer>900</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>$ARCHIVE_DIR/archive.err</string>
</dict>
</plist>
EOF
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "installed $PLIST (every 15 min)"
    exit 0 ;;
  --uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST" "$INSTALLED"
    echo "removed $LABEL"
    exit 0 ;;
  "") ;;
  *) echo "usage: $0 [--install|--uninstall]" >&2; exit 2 ;;
esac

mkdir -p "$ARCHIVE_DIR"
touch "$INDEX"

for name in Player-prev.log Player.log; do
  log="$SRC/$name"
  [ -s "$log" ] || continue

  # Every log opens with the same Unity boilerplate; the first 64 KB also
  # holds the session's first timestamp, so its hash identifies the session.
  key=$(head -c 65536 "$log" | shasum -a 256 | cut -c1-16)
  size=$(stat -f %z "$log")

  prev=$(awk -F'\t' -v k="$key" '$1 == k { print $2 "\t" $3 }' "$INDEX")
  prev_size=${prev%%$'\t'*}
  prev_file=${prev#*$'\t'}
  if [ -n "$prev" ] && [ "$size" -le "$prev_size" ]; then
    continue                     # already have this session at full length
  fi

  # Birth time survives Arena's Player.log -> Player-prev.log rename.
  start=$(stat -f %SB -t %Y-%m-%d_%H%M%S "$log")
  file="${start}_${key:0:8}.log.gz"

  # Snapshot first so a log Arena is still writing can't change mid-gzip.
  tmp=$(mktemp "$ARCHIVE_DIR/.tmp.XXXXXX")
  head -c "$size" "$log" | gzip -6 > "$tmp"
  mv "$tmp" "$ARCHIVE_DIR/$file"
  if [ -n "$prev" ] && [ "$prev_file" != "$file" ]; then
    rm -f "$ARCHIVE_DIR/$prev_file"
  fi

  awk -F'\t' -v k="$key" '$1 != k' "$INDEX" > "$INDEX.tmp"
  printf '%s\t%s\t%s\n' "$key" "$size" "$file" >> "$INDEX.tmp"
  mv "$INDEX.tmp" "$INDEX"
  echo "archived $name -> $file ($((size / 1024)) KB)"
done
