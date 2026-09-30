#!/usr/bin/env bash
# Point the active release at a previously published snapshot.
set -euo pipefail

RELEASE_ROOT="${1:?usage: rollback_snapshot.sh RELEASE_ROOT SNAPSHOT_ID}"
SNAPSHOT_ID="${2:?usage: rollback_snapshot.sh RELEASE_ROOT SNAPSHOT_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SNAPSHOT_VALIDATOR:-${SCRIPT_DIR}/../scripts/validate_evidence_snapshot.py}"
case "${SNAPSHOT_ID}" in (*[!A-Za-z0-9_.-]*|'') echo "invalid snapshot_id" >&2; exit 1;; esac
[ -f "${RELEASE_ROOT}/releases/${SNAPSHOT_ID}/manifest.json" ] || { echo "snapshot not found" >&2; exit 1; }
python3 "${VALIDATOR}" "${RELEASE_ROOT}/releases/${SNAPSHOT_ID}" \
    --snapshot-id "${SNAPSHOT_ID}" \
    --minimum-records "${SNAPSHOT_MINIMUM_RECORDS:-1}"
next_link="${RELEASE_ROOT}/.current.${SNAPSHOT_ID}.$$"
ln -s "releases/${SNAPSHOT_ID}" "${next_link}"
mv -Tf "${next_link}" "${RELEASE_ROOT}/current"
echo "rolled back to ${SNAPSHOT_ID}"
if [ "${RESTART_SERVICE:-0}" = "1" ]; then
    systemctl restart "${MCP_SERVICE_NAME:-gemynd-mcp}"
    echo "restarted ${MCP_SERVICE_NAME:-gemynd-mcp}"
else
    echo "activation pending: restart gemynd-mcp after rollback"
fi
