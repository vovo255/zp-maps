#!/usr/bin/env python3
"""Checks the map of build.sh and prepares the release files, README.md "Проверки".

usage: PYTHONPATH=. python3 release.py <work dir of build.sh> <out dir> [--force]
Writes to <out dir>:
  russia.bin.xz the map without the AXES sections (the phone computes them), xz
  map.json      what the phone reads: format version, time of the OSM data, tag and file of the release, md5 of the map
                without the AXES, the counts
  stats.json    the counts the next build is compared with
  notes.md      the text of the release
Exits with an error and writes nothing when a count differs from stats.json of the repo by more than MAX_CHANGE, unless
--force.
"""
import filecmp
import hashlib
import json
import lzma
import os
import sys

import numpy as np

from openpilot.selfdrive.zeropilot.highways import CLASSES, HEADER, VERSION, Highways, complete, core_size

MAX_CHANGE = 0.05
FILE = 'russia.bin.xz'
ROOT = os.path.dirname(os.path.abspath(__file__))
# the rows of the check: edges of the map by the classes of CLASSES; a class alone is too small for 5 %, motorway_link of
# Russia has about 300 edges
GROUPS = (('highways', 'motorway, trunk, primary и их съезды', range(0, 6)),
          ('secondary', 'secondary и съезды', range(6, 8)),
          ('tertiary', 'tertiary и съезды', range(8, 10)),
          ('unclassified', 'unclassified', range(10, 11)),
          ('streets', 'residential, living_street', range(11, 13)))


def main():
  work, out = sys.argv[1], sys.argv[2]
  force = '--force' in sys.argv[3:]
  full = os.path.join(work, 'russia.bin')
  h = np.fromfile(full, HEADER, 1)[0]
  osm_time = h['osm_time'].decode()
  if not osm_time:
    sys.exit('the map has no time of the OSM data')

  per_class = np.bincount(Highways(full).edges['cls'], minlength=len(CLASSES))
  counts = {name: int(per_class[list(classes)].sum()) for name, _, classes in GROUPS}
  titles = {name: title for name, title, _ in GROUPS}
  before = {}
  if os.path.exists(os.path.join(ROOT, 'stats.json')):
    with open(os.path.join(ROOT, 'stats.json')) as f:
      before = json.load(f)
  rows, bad = [], []
  for name, n in counts.items():
    old = before.get('counts', {}).get(name)
    change = (n - old) / old if old else None
    if change is not None and abs(change) > MAX_CHANGE:
      bad.append(name)
    rows.append(f'| {titles[name]} | {n} | {old if old else "нет"} | {f"{change * 100:+.2f} %" if change is not None else ""} |')
  table = '\n'.join(['| Часть | Сейчас | В прошлой карте | Изменение |', '|---|---|---|---|'] + rows)
  print(table)
  if bad and not force:
    sys.exit(f'more than {MAX_CHANGE * 100:.0f} % off the map of {before.get("osm_time")}: {", ".join(bad)}; not published')

  # the phone completes the map with the same code: the result must be this map
  with open(full, 'rb') as f:
    core = f.read(core_size(h))
  check_path = os.path.join(work, 'completed.bin')
  with open(check_path, 'wb') as f:
    f.write(core)
  complete(check_path)
  if not filecmp.cmp(full, check_path, shallow=False):
    sys.exit('the map completed from the published part is not the map built')
  os.remove(check_path)
  os.makedirs(out, exist_ok=True)

  xz = lzma.compress(core, preset=9 | lzma.PRESET_EXTREME)
  with open(os.path.join(out, FILE), 'wb') as f:
    f.write(xz)
  tag = osm_time[:10]
  info = {'version': VERSION, 'osm_time': osm_time, 'tag': tag, 'file': FILE, 'md5': hashlib.md5(core).hexdigest(),
          'size': len(core), 'xz_size': len(xz), 'counts': counts}
  with open(os.path.join(out, 'map.json'), 'w') as f:
    json.dump(info, f, indent=1)
  with open(os.path.join(out, 'stats.json'), 'w') as f:
    json.dump({'osm_time': osm_time, 'counts': counts}, f, indent=1)
    f.write('\n')
  with open(os.path.join(out, 'notes.md'), 'w') as f:
    f.write(f'Карта дорог для ZeroPilot. Данные OpenStreetMap от {osm_time}, © участники OpenStreetMap, лицензия ODbL.\n\n'
            f'Рёбра карты по классам дорог OSM, сравнение с картой от '
            f'{before.get("osm_time", "нет")}:\n\n{table}\n' + ('\nОпубликована без проверки изменений (force).\n' if bad else ''))
  print(f'{tag}: {len(core) / 1e6:.1f} MB, xz {len(xz) / 1e6:.1f} MB')


if __name__ == '__main__':
  main()
