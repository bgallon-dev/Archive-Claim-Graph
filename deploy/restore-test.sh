#!/usr/bin/env bash
# Non-destructive restore verification. Usage: restore-test.sh /path/to/backup
set -euo pipefail
backup="${1:?usage: restore-test.sh /path/to/backup}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
count=0
for db in "$backup"/data/*.db; do
  [ -e "$db" ] || continue
  sqlite3 "$db" "PRAGMA integrity_check;" | grep -qx ok
  sqlite3 "$db" ".backup '$tmp/$(basename "$db")'"
  sqlite3 "$tmp/$(basename "$db")" "PRAGMA integrity_check;" | grep -qx ok
  count=$((count + 1))
done
[ "$count" -gt 0 ]
echo "restore verification passed: $count SQLite database(s)"
