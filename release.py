#!/usr/bin/env python3
"""Checks the map of build.sh and prepares the release files, README.md "Проверки".

usage: PYTHONPATH=. python3 release.py <work dir of build.sh> <out dir> [--force]
Writes to <out dir>:
  south.bin.xz  the map without the AXES sections (the phone computes them), xz
  map.json      what the phone reads: format version, time of the OSM data, tag and file of the release, md5 of the map
                without the AXES, the counts
  stats.json    the counts the next build is compared with
  notes.md      the text of the release
Exits with an error and writes nothing when a count differs from stats.json of the repo by more than MAX_CHANGE, unless
--force.
"""
import hashlib
import json
import lzma
import os
import subprocess
import sys

import numpy as np

from openpilot.selfdrive.zeropilot.highways import CLASSES, HEADER, VERSION, Highways, complete, core_size

MAX_CHANGE = 0.05
FILE = 'south.bin.xz'
ROOT = os.path.dirname(os.path.abspath(__file__))


def ways(pbf: str) -> int:
  out = subprocess.run(['osmium', 'fileinfo', '-e', '-g', 'data.count.ways', pbf], check=True, capture_output=True, text=True)
  return int(out.stdout)


def roads(pbf: str, work: str) -> int:
  """Ways of the classes of the map in an extract."""
  out = os.path.join(work, 'count.osm.pbf')
  subprocess.run(['osmium', 'tags-filter', '-O', '-o', out, pbf, 'w/highway=' + ','.join(CLASSES)], check=True)
  return ways(out)


def main():
  work, out = sys.argv[1], sys.argv[2]
  force = '--force' in sys.argv[3:]
  with open(os.path.join(ROOT, 'regions.json')) as f:
    regions = json.load(f)
  full = os.path.join(work, 'south.bin')
  h = np.fromfile(full, HEADER, 1)[0]
  osm_time = h['osm_time'].decode()
  if not osm_time:
    sys.exit('the map has no time of the OSM data')

  # roads of every region, the highways of Russia and the edges of the map
  counts = {r['name']: roads(os.path.join(work, r['name'] + '.osm.pbf'), work) for r in regions}
  counts['highways'] = ways(os.path.join(work, 'hw.osm.pbf'))
  counts['edges'] = int(h['n_edge'])
  titles = dict({r['name']: r['title'] for r in regions}, highways='Трассы России', edges='Рёбер в карте')
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
    data = f.read()
  core = data[:core_size(h)]
  os.makedirs(out, exist_ok=True)
  core_path, check_path = os.path.join(work, 'core.bin'), os.path.join(work, 'completed.bin')
  with open(core_path, 'wb') as f:
    f.write(core)
  complete(core_path, check_path)
  with open(check_path, 'rb') as f:
    if f.read() != data:
      sys.exit('the map completed from the published part is not the map built')
  Highways(full)

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
            f'Дороги OSM (way) классов карты по регионам, трассы России и рёбра карты, сравнение с картой от '
            f'{before.get("osm_time", "нет")}:\n\n{table}\n' + ('\nОпубликована без проверки изменений (force).\n' if bad else ''))
  print(f'{tag}: {len(core) / 1e6:.1f} MB, xz {len(xz) / 1e6:.1f} MB')


if __name__ == '__main__':
  main()
