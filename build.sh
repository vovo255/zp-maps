#!/bin/bash
# Builds the map of Russia from its OSM extract, README.md "Сборка".
# usage: build.sh <work dir> [russia.osm.pbf]; without the extract it is downloaded from Geofabrik.
# Needs osmium-tool, python with osmium and numpy. The map is <work dir>/russia.bin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$1"
cd "$1"
PBF="${2:-}"
if [ -z "$PBF" ]; then
  URL=https://download.geofabrik.de/russia-latest.osm.pbf
  curl -fsSL --retry 5 -o russia-latest.osm.pbf "$URL"
  curl -fsSL --retry 5 -o russia-latest.osm.pbf.md5 "$URL.md5"
  md5sum -c russia-latest.osm.pbf.md5
  PBF=russia-latest.osm.pbf
fi
echo "OSM data of $(osmium fileinfo -g header.option.osmosis_replication_timestamp "$PBF")"
# every drivable street of Russia
PYTHONPATH="$ROOT" "$PY" "$ROOT/tools/zp/osm_highways.py" "$PBF" russia.bin --all-streets
