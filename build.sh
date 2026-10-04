#!/bin/bash
# Builds the map of the south from the OSM extract of Russia, README.md "Сборка".
# usage: build.sh <work dir> [russia.osm.pbf]; without the extract it is downloaded from Geofabrik.
# Needs osmium-tool, python with osmium and numpy. The map is <work dir>/south.bin.
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
TS=$(osmium fileinfo -g header.option.osmosis_replication_timestamp "$PBF")
echo "OSM data of $TS"

# the regions by their borders: only the boundary relation itself, see tools/zp/osm_border.py
osmium tags-filter -O "$PBF" r/admin_level=4 -o adm4.osm.pbf
NAMES=$("$PY" -c "import json; print(' '.join(r['name'] for r in json.load(open('$ROOT/regions.json'))))")
BOXES=()
EXTRACTS=()
for name in $NAMES; do
  rel=$("$PY" -c "import json; print(next(r['relation'] for r in json.load(open('$ROOT/regions.json')) if r['name'] == '$name'))")
  box=$(PYTHONPATH="$ROOT" "$PY" "$ROOT/tools/zp/osm_border.py" adm4.osm.pbf "$rel" "$name.geojson" | awk '{print $NF}')
  BOXES+=(--streets "$box")
  EXTRACTS+=("{\"output\": \"$name.osm.pbf\", \"multipolygon\": {\"file_name\": \"$name.geojson\", \"file_type\": \"geojson\"}, \"output_header\": {\"osmosis_replication_timestamp\": null}}")
done
(IFS=,; echo "{\"directory\": \".\", \"extracts\": [${EXTRACTS[*]}]}") > extracts.json
osmium extract -O -c extracts.json -s complete_ways "$PBF"

# the highways of Russia and the three regions in one extract: the builder takes every street in their boxes
osmium tags-filter -O "$PBF" w/highway=motorway,trunk,motorway_link,trunk_link,primary,primary_link -o hw.osm.pbf
osmium merge -O --output-header=osmosis_replication_timestamp="$TS" hw.osm.pbf $(for n in $NAMES; do echo "$n.osm.pbf"; done) -o south.osm.pbf
PYTHONPATH="$ROOT" "$PY" "$ROOT/tools/zp/osm_highways.py" south.osm.pbf south.bin "${BOXES[@]}"
