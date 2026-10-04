#!/usr/bin/env python3
"""The border of a region as a GeoJSON multipolygon for osmium extract, see docs/speedcam/highways.md.

Takes only the boundary relation itself. A border file of `osmium getid -r` also holds the member relations of the
region (districts, towns), osmium extract takes their polygons too, and the towns fall out of the extract.
Runs on the Mac, needs pyosmium (pip install osmium):
  <venv>/bin/python tools/zp/osm_border.py adm4.osm.pbf 85606 rostov.geojson
adm4.osm.pbf: osmium tags-filter russia-latest.osm.pbf r/admin_level=4 -o adm4.osm.pbf
"""
import json
import math
import sys

import osmium


def main():
  pbf, rid, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
  for a in osmium.FileProcessor(pbf).with_areas():
    if a.is_area() and not a.from_way() and a.orig_id() == rid:
      polys = []
      for outer in a.outer_rings():
        rings = [[[n.lon, n.lat] for n in outer]]
        rings += [[[n.lon, n.lat] for n in inner] for inner in a.inner_rings(outer)]
        polys.append(rings)
      with open(out, 'w') as f:
        json.dump({'type': 'Feature', 'properties': {'relation': rid}, 'geometry': {'type': 'MultiPolygon', 'coordinates': polys}}, f)
      lon = [p[0] for poly in polys for p in poly[0]]
      lat = [p[1] for poly in polys for p in poly[0]]
      box = [math.floor(min(lon) * 100) / 100, math.floor(min(lat) * 100) / 100, math.ceil(max(lon) * 100) / 100,
             math.ceil(max(lat) * 100) / 100]  # outward to 0.01 deg, for --streets of osm_highways.py
      print(f'relation {rid}: {len(polys)} polygons, {sum(len(p) - 1 for p in polys)} inner rings, box', ','.join(f'{x:.2f}' for x in box))
      return
  sys.exit(f'relation {rid} not found in {pbf}')


if __name__ == '__main__':
  main()
