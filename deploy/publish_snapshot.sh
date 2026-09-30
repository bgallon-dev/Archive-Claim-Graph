#!/usr/bin/env bash
# Validate, atomically publish, and optionally activate a snapshot.
set -euo pipefail

SOURCE_DIR="${1:?usage: publish_snapshot.sh SNAPSHOT_DIR RELEASE_ROOT}"
RELEASE_ROOT="${2:?usage: publish_snapshot.sh SNAPSHOT_DIR RELEASE_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SNAPSHOT_VALIDATOR:-${SCRIPT_DIR}/../scripts/validate_evidence_snapshot.py}"
MANIFEST="${SOURCE_DIR}/manifest.json"
[ -f "${MANIFEST}" ] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
[ -f "${VALIDATOR}" ] || { echo "missing snapshot validator: ${VALIDATOR}" >&2; exit 1; }

snapshot_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_id"])' "${MANIFEST}")"
case "${snapshot_id}" in (*[!A-Za-z0-9_.-]*|'') echo "invalid snapshot_id" >&2; exit 1;; esac
python3 "${VALIDATOR}" "${SOURCE_DIR}" \
    --snapshot-id "${snapshot_id}" \
    --minimum-records "${SNAPSHOT_MINIMUM_RECORDS:-1}"
mkdir -p "${RELEASE_ROOT}/releases"
staging="${RELEASE_ROOT}/releases/.${snapshot_id}.tmp.$$"
target="${RELEASE_ROOT}/releases/${snapshot_id}"
[ ! -e "${target}" ] || { echo "snapshot release already exists: ${target}" >&2; exit 1; }
[ ! -e "${staging}" ] || { echo "staging path already exists: ${staging}" >&2; exit 1; }
if [ "${SNAPSHOT_PUBLISH_MODE:-copy}" = "move" ]; then
    source_device="$(stat -c %d "${SOURCE_DIR}")"
    release_device="$(stat -c %d "${RELEASE_ROOT}/releases")"
    [ "${source_device}" = "${release_device}" ] || {
        echo "move mode requires source and release root on the same filesystem" >&2
        exit 1
    }
    mv "${SOURCE_DIR}" "${target}"
else
    cp -a --reflink=auto "${SOURCE_DIR}" "${staging}"
    python3 "${VALIDATOR}" "${staging}" \
        --snapshot-id "${snapshot_id}" \
        --minimum-records "${SNAPSHOT_MINIMUM_RECORDS:-1}"
    mv "${staging}" "${target}"
fi
python3 "${VALIDATOR}" "${target}" \
    --snapshot-id "${snapshot_id}" \
    --minimum-records "${SNAPSHOT_MINIMUM_RECORDS:-1}"
next_link="${RELEASE_ROOT}/.current.${snapshot_id}.$$"
ln -s "releases/${snapshot_id}" "${next_link}"
mv -Tf "${next_link}" "${RELEASE_ROOT}/current"
echo "published ${snapshot_id} -> ${RELEASE_ROOT}/current"

# The MCP opens and verifies its SQLite snapshot at process start. Switching
# the symlink does not replace the already-open database connection.
if [ "${RESTART_SERVICE:-0}" = "1" ]; then
    systemctl restart "${MCP_SERVICE_NAME:-gemynd-mcp}"
    echo "restarted ${MCP_SERVICE_NAME:-gemynd-mcp}"
else
    echo "activation pending: restart gemynd-mcp after publishing"
fi
