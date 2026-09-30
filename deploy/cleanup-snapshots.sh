#!/usr/bin/env bash
# Remove expired immutable snapshots, preserving the active target.
set -euo pipefail
root="$(realpath "${SNAPSHOT_ROOT:-/opt/gemynd/snapshots}")"
retention_days="${SNAPSHOT_RETENTION_DAYS:-30}"
dry_run="${DRY_RUN:-1}"
case "$root" in /opt/gemynd/snapshots|/opt/gemynd/snapshots/*) ;; *) echo "refusing unsafe snapshot root: $root" >&2; exit 2 ;; esac
active=""
[ -e "$root/current" ] && active="$(realpath "$root/current")"
find "$root" -mindepth 1 -maxdepth 1 -type d -mtime "+$retention_days" -print0 |
while IFS= read -r -d '' candidate; do
  [ "$candidate" = "$active" ] && { echo "preserve active: $candidate"; continue; }
  if [ "$dry_run" = "0" ]; then rm -rf --one-file-system -- "$candidate"; echo "removed: $candidate"; else echo "would remove: $candidate"; fi
done
