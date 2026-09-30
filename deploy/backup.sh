#!/usr/bin/env bash
#
# Nightly backup for a Gemynd pilot deployment.
#
#   sudo cp deploy/backup.sh /opt/gemynd/deploy/
#   sudo chmod +x /opt/gemynd/deploy/backup.sh
#   # then, as the gemynd user:
#   crontab -e
#   15 2 * * *  /opt/gemynd/deploy/backup.sh >> /var/log/gemynd/backup.log 2>&1
#
# What's at risk without this:
#   * data/*.db holds users, document notes, annotations, saved searches,
#     query history and tester feedback. None of it is in git, and none of it
#     can be regenerated. This is the irreplaceable part.
#   * The Neo4j graph *can* be rebuilt by re-ingesting from out/, but that is
#     hours-to-weeks of LLM work depending on corpus size.
#
# A retrieval-only pilot writes nothing to the graph, so the Neo4j dump is
# effectively a one-time snapshot; the SQLite side still changes daily.
#
set -euo pipefail

GEMYND_HOME="${GEMYND_HOME:-/opt/gemynd}"
BACKUP_ROOT="${BACKUP_ROOT:-/var/backups/gemynd}"
RETAIN_DAYS="${RETAIN_DAYS:-14}"
NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
# Dumping a running Neo4j requires neo4j-admin >= 5 with the database online.
# On older versions the service must be stopped first — set this to 0 and the
# graph dump is skipped rather than producing a torn, silently-useless file.
DUMP_GRAPH="${DUMP_GRAPH:-1}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
dest="${BACKUP_ROOT}/${timestamp}"
mkdir -p "${dest}"

echo "[$(date -u +%FT%TZ)] backup -> ${dest}"

# ---------------------------------------------------------------------------
# SQLite: use the .backup command, NOT cp.
#
# These databases run in WAL mode and are being written while this script
# runs. Copying the file alone can capture a torn page or miss committed
# transactions still in the -wal file; .backup takes a consistent snapshot of
# a live database. This distinction is the whole reason this script exists
# rather than a cron'd `cp -r`.
# ---------------------------------------------------------------------------
sqlite_count=0
if [ -d "${GEMYND_HOME}/data" ]; then
    mkdir -p "${dest}/data"
    for db in "${GEMYND_HOME}"/data/*.db; do
        [ -e "${db}" ] || continue
        name="$(basename "${db}")"
        if sqlite3 "${db}" ".backup '${dest}/data/${name}'"; then
            sqlite_count=$((sqlite_count + 1))
        else
            echo "  !! FAILED to back up ${name}" >&2
        fi
    done
fi
echo "  sqlite databases: ${sqlite_count}"

# ---------------------------------------------------------------------------
# Configuration holds credentials — restrict the copy to the owner. Support
# both the legacy app .env and the MCP secret file used by systemd.
# ---------------------------------------------------------------------------
if [ -f "${GEMYND_HOME}/.env" ]; then
    cp "${GEMYND_HOME}/.env" "${dest}/env.backup"
    chmod 600 "${dest}/env.backup"
    echo "  .env captured (mode 600)"
fi
if [ -f "/etc/gemynd/kettle.env" ]; then
    cp "/etc/gemynd/kettle.env" "${dest}/kettle.env.backup"
    chmod 600 "${dest}/kettle.env.backup"
    echo "  kettle.env captured (mode 600)"
fi

if [ -f "${GEMYND_HOME}/data/corpus_registry.yaml" ]; then
    cp "${GEMYND_HOME}/data/corpus_registry.yaml" "${dest}/corpus_registry.yaml"
fi

# ---------------------------------------------------------------------------
# Neo4j graph dump.
# ---------------------------------------------------------------------------
if [ "${DUMP_GRAPH}" = "1" ] && command -v neo4j-admin >/dev/null 2>&1; then
    if neo4j-admin database dump "${NEO4J_DATABASE}" \
            --to-path="${dest}" --overwrite-destination=true; then
        echo "  neo4j dump: ok"
    else
        echo "  !! neo4j dump FAILED — graph not backed up tonight" >&2
    fi
else
    echo "  neo4j dump: skipped (DUMP_GRAPH=${DUMP_GRAPH}, neo4j-admin present: $(command -v neo4j-admin >/dev/null 2>&1 && echo yes || echo no))"
fi

# ---------------------------------------------------------------------------
# Verify before rotating. An unverified backup is a guess, and the moment you
# find out it was empty is the moment you needed it.
# ---------------------------------------------------------------------------
for db in "${dest}"/data/*.db; do
    [ -e "${db}" ] || continue
    if ! sqlite3 "${db}" "PRAGMA integrity_check;" | grep -q '^ok$'; then
        echo "  !! INTEGRITY CHECK FAILED for $(basename "${db}") — keeping old backups" >&2
        exit 1
    fi
done
echo "  integrity: ok"

# ---------------------------------------------------------------------------
# Rotate only after this run verified clean, so a failing backup never
# deletes the last good one.
# ---------------------------------------------------------------------------
find "${BACKUP_ROOT}" -maxdepth 1 -type d -name '20*' -mtime "+${RETAIN_DAYS}" \
    -exec rm -rf {} + 2>/dev/null || true

echo "[$(date -u +%FT%TZ)] backup complete: ${dest}"
echo "  size: $(du -sh "${dest}" | cut -f1)"

# Restore is documented in docs/DEPLOYMENT.md. Test it before the pilot, not
# during it — a backup nobody has restored is not yet a backup.
