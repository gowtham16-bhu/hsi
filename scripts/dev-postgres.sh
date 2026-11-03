#!/usr/bin/env bash
# Start the development PostgreSQL.
#   1. Docker available  -> docker compose (same image as the documented dev setup)
#   2. otherwise         -> local PostgreSQL binaries, data in .dev/pgdata
# Prints the DSN to export as HSI_PG_DSN.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${HSI_PG_PORT:-5432}"
DSN="postgresql://hsi:hsi_dev_only@127.0.0.1:${PORT}/hsi"

if docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.dev.yml up -d --wait
  echo "export HSI_PG_DSN=${DSN}"
  exit 0
fi

PGBIN="${PGBIN:-$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)}"
if [[ -z "${PGBIN}" || ! -x "${PGBIN}/pg_ctl" ]]; then
  echo "neither Docker nor local PostgreSQL binaries found" >&2
  exit 1
fi

DATA="$(pwd)/.dev/pgdata"
LOG="$(pwd)/.dev/postgres.log"
mkdir -p .dev

as_pg() {
  if [[ "$(id -u)" == "0" ]]; then runuser -u postgres -- "$@"; else "$@"; fi
}

if [[ "$(id -u)" == "0" ]]; then chown postgres:postgres .dev; fi

if [[ ! -f "${DATA}/PG_VERSION" ]]; then
  as_pg "${PGBIN}/initdb" -D "${DATA}" -U postgres --auth=trust --encoding=UTF8 >/dev/null
fi

if ! as_pg "${PGBIN}/pg_ctl" -D "${DATA}" status >/dev/null 2>&1; then
  as_pg "${PGBIN}/pg_ctl" -D "${DATA}" -l "${LOG}" -o "-p ${PORT} -k /tmp -c listen_addresses=127.0.0.1" -w start >/dev/null
fi

PSQL=( "${PGBIN}/psql" -h 127.0.0.1 -p "${PORT}" -U postgres -v ON_ERROR_STOP=1 -qtA )
if [[ "$("${PSQL[@]}" -c "select 1 from pg_roles where rolname='hsi'")" != "1" ]]; then
  "${PSQL[@]}" -c "create role hsi login password 'hsi_dev_only'"
fi
if [[ "$("${PSQL[@]}" -c "select 1 from pg_database where datname='hsi'")" != "1" ]]; then
  "${PSQL[@]}" -c "create database hsi owner hsi"
fi
echo "export HSI_PG_DSN=${DSN}"
