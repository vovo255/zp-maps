#!/usr/bin/env python3
"""Builds the highway map (selfdrive/zeropilot/highways.py) from an OSM extract, see docs/speedcam/highways.md.

Runs on the Mac and in the weekly build of the zp-maps repo (tools/zp/zp_maps_sync.sh copies it there), needs
osmium-tool (brew install osmium-tool) and pyosmium with numpy (pip install osmium numpy):
  PYTHONPATH=. <venv>/bin/python tools/zp/osm_highways.py russia-latest.osm.pbf highways.bin [--streets ROSTOV]
--streets lon,lat,lon,lat (repeatable): a box in which every drivable street goes into the file too.
--all-streets: every drivable street of the extract, the map of Russia of the zp-maps repo.
"""
import argparse
import os
import re
import subprocess
import tempfile
from typing import List, NamedTuple

import numpy as np

from openpilot.selfdrive.zeropilot.highways import CELL_COLS, CELL_E7, CLASSES, EDGE, HEADER, HIGHWAY_CLASSES, MAGIC, VERSION, axes, sections

FEDERAL = re.compile(r'\s*[МРАMPA]-\s?\d')  # М-4, Р-217, А-160, also with Latin letters
ROSTOV = '39.35,47.05,40.20,47.50'  # lon,lat,lon,lat: Rostov, Aksay, Bataysk, Azov, Novocherkassk


class Way(NamedTuple):
  nodes: np.ndarray  # OSM node ids
  lat: np.ndarray  # 1e-7 degrees
  lon: np.ndarray
  cls: int  # index in CLASSES
  oneway: int  # 1 from the first node to the last, -1 from the last to the first, 0 both ways
  ref: str
  name: str
  id: int = 0  # OSM way id; a way of a street box that is also in the country pass goes in once


def oneway(tags) -> int:
  v = tags.get('oneway')
  if v in ('yes', 'true', '1'):
    return 1
  if v in ('-1', 'reverse'):
    return -1
  if v in ('no', 'false', '0'):
    return 0
  # without the tag OSM takes motorways, their links and roundabouts as oneway
  return int(tags.get('highway') in ('motorway', 'motorway_link') or tags.get('junction') in ('roundabout', 'circular'))


def wanted(tags, streets=False) -> bool:
  """Primary roads go in only with a federal number: a federal road keeps it when OSM marks a part of it primary.
  In a street box (streets=True) every class of CLASSES goes in."""
  if streets:
    return tags.get('highway') in CLASSES
  if tags.get('highway') not in CLASSES[:HIGHWAY_CLASSES]:
    return False
  return not tags['highway'].startswith('primary') or any(FEDERAL.match(r) for r in (tags.get('ref') or '').split(';'))


def read_osm(pbf: str, box: str = '', streets: bool = False):
  """The wanted ways of the extract and the time of its data; with box (lon,lat,lon,lat) every street inside it, with
  streets every street of the extract."""
  import osmium

  osm_time = subprocess.run(['osmium', 'fileinfo', '-g', 'header.option.osmosis_replication_timestamp', pbf],
                            check=True, capture_output=True, text=True).stdout.strip()
  streets = streets or bool(box)
  classes = CLASSES if streets else CLASSES[:HIGHWAY_CLASSES]
  ways = []
  with tempfile.TemporaryDirectory() as tmp:
    small = os.path.join(tmp, 'highways.osm.pbf')
    src = pbf
    if box:
      src = os.path.join(tmp, 'box.osm.pbf')
      subprocess.run(['osmium', 'extract', '-b', box, '-s', 'complete_ways', '-o', src, pbf], check=True)
    subprocess.run(['osmium', 'tags-filter', '-o', small, src, 'w/highway=' + ','.join(classes)], check=True)
    for w in osmium.FileProcessor(small).with_locations():
      if not w.is_way():
        continue
      tags = {k: w.tags.get(k) for k in ('highway', 'oneway', 'junction', 'ref', 'name')}
      if not wanted(tags, streets):
        continue
      # a way that leaves the extract has nodes without a location: every part with locations is taken on its own
      run = []
      for n in list(w.nodes) + [None]:
        if n is not None and n.location.valid():
          run.append((n.ref, n.location.y, n.location.x))
          continue
        if len(run) > 1:
          ids, lat, lon = zip(*run)
          ways.append(Way(np.array(ids, np.int64), np.array(lat, np.int32), np.array(lon, np.int32),
                          CLASSES.index(tags['highway']), oneway(tags), tags['ref'] or '', tags['name'] or '', w.id))
        run = []
  return ways, osm_time


def combine(ways: List[Way], extra: List[Way]) -> List[Way]:
  """ways with the ways of extra whose OSM id is not in ways: a highway inside a street box comes in both passes."""
  ids = {w.id for w in ways}
  return ways + [w for w in extra if w.id not in ids]


def morton(lat_e7, lon_e7):
  """The Z-order key of the points: the bits of latitude and longitude in turn, the last step about 45 m."""
  def spread(x):
    for shift, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
                        (1, 0x5555555555555555)):
      x = (x | (x << np.uint64(shift))) & np.uint64(mask)
    return x
  la = ((np.asarray(lat_e7, np.int64) + 90 * 10**7) >> 12).astype(np.uint64)
  lo = ((np.asarray(lon_e7, np.int64) + 180 * 10**7) >> 12).astype(np.uint64)
  return spread(la) << np.uint64(1) | spread(lo)


def build(ways: List[Way]) -> dict:
  """The parts of the file: ways cut into edges at junctions, the junctions, the cells. The nodes go through numpy
  arrays, a map of all streets of Russia has 30 million of them."""
  # the nodes of all ways in a row, a way with oneway -1 from its last node to its first
  turn = [slice(None, None, -1) if w.oneway < 0 else slice(None) for w in ways]
  nodes = np.concatenate([np.asarray(w.nodes, np.int64)[t] for w, t in zip(ways, turn)])
  lat = np.concatenate([w.lat[t] for w, t in zip(ways, turn)])
  lon = np.concatenate([w.lon[t] for w, t in zip(ways, turn)])
  way_first = np.r_[0, np.cumsum([len(w.nodes) for w in ways])]
  # a junction: a node of two ways or of two places of one way; an end is a junction even when nothing else meets there
  ids, node, use = np.unique(nodes, return_inverse=True, return_counts=True)
  use += np.bincount(node[np.r_[way_first[:-1], way_first[1:] - 1]], minlength=len(ids))
  cut = np.nonzero(use[node] > 1)[0]
  way_of = np.repeat(np.arange(len(ways)), np.diff(way_first))
  inside = way_of[cut[:-1]] == way_of[cut[1:]]
  a, b = cut[:-1][inside], cut[1:][inside]  # an edge runs from point a to point b of the row
  strings, blob = {'': 0}, bytearray(b'\0')

  def string(s):
    if s not in strings:
      strings[s] = len(blob)
      blob.extend(s.encode() + b'\0')
    return strings[s]

  text = np.array([(string(w.ref), string(w.name)) for w in ways], np.uint32).reshape(-1, 2)
  edge_way = way_of[a]
  edges = np.zeros(len(a), EDGE)
  edges['v_from'], edges['v_to'] = node[a], node[b]
  edges['ref'], edges['name'] = text[edge_way, 0], text[edge_way, 1]
  edges['cls'] = np.array([w.cls for w in ways], np.uint8)[edge_way]
  edges['oneway'] = np.array([w.oneway != 0 for w in ways])[edge_way]
  # the edges in the order of the place of their first point, the junctions in the order of their first edge: the roads of
  # one district come on neighbouring pages of the file, and positiond maps fewer pages
  order = np.argsort(morton(lat[a], lon[a]), kind='stable')
  edges, a, b = edges[order], a[order], b[order]
  first = np.r_[0, np.cumsum(b + 1 - a)].astype(np.uint32)
  pick = np.arange(first[-1]) + np.repeat(a - first[:-1], np.diff(first))
  lat, lon = lat[pick], lon[pick]
  ends = np.column_stack([edges['v_from'], edges['v_to']]).ravel()
  verts_seen, at = np.unique(ends, return_index=True)
  n_vert = len(verts_seen)
  renumber = np.zeros(len(ids), np.uint32)
  renumber[verts_seen[np.argsort(at)]] = np.arange(n_vert, dtype=np.uint32)
  edges['v_from'], edges['v_to'] = renumber[edges['v_from']], renumber[edges['v_to']]
  points = np.column_stack([lat, lon]).ravel()

  ends = np.concatenate([edges['v_from'], edges['v_to']])
  order = np.argsort(ends, kind='stable')
  vert_edges = np.tile(np.arange(len(edges), dtype=np.uint32), 2)[order]
  vert_first = np.searchsorted(ends[order], np.arange(n_vert + 1)).astype(np.uint32)

  # every segment goes into all cells of its bounding box
  is_seg = np.ones(len(lat) - 1, bool)
  is_seg[first[1:-1] - 1] = False
  seg = np.nonzero(is_seg)[0]
  row = (lat.astype(np.int64) + 90 * 10**7) // CELL_E7
  col = (lon.astype(np.int64) + 180 * 10**7) // CELL_E7
  r0, r1 = np.minimum(row[seg], row[seg + 1]), np.maximum(row[seg], row[seg + 1])
  c0, c1 = np.minimum(col[seg], col[seg + 1]), np.maximum(col[seg], col[seg + 1])
  wide = np.nonzero(c1 - c0 > CELL_COLS // 2)[0]
  if len(wide):
    raise ValueError(f'segment {seg[wide[0]]} crosses the 180th meridian, the cells do not handle it')
  # cell k of a segment: row r0 + k // cols, column c0 + k % cols of its box
  cols = c1 + 1 - c0
  n = (r1 + 1 - r0) * cols
  k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
  cols = np.repeat(cols, n)
  keys = (np.repeat(r0, n) + k // cols) * CELL_COLS + np.repeat(c0, n) + k % cols
  segs = np.repeat(seg, n)
  del k, cols
  order = np.lexsort((segs, keys))
  keys, segs = keys[order], segs[order]
  cell_keys, start = np.unique(keys, return_index=True)

  return {'edges': edges, 'edge_first': first, 'points': points, 'vert_first': vert_first, 'vert_edges': vert_edges,
          'cell_keys': cell_keys, 'cell_first': np.r_[start, len(keys)], 'cell_segs': segs,
          'strings': np.frombuffer(bytes(blob), np.uint8), **axes(points, first)}


def write(path: str, parts: dict, osm_time: str):
  h = np.zeros((), HEADER)
  h['magic'], h['version'], h['osm_time'] = MAGIC, VERSION, osm_time.encode()
  for field, part in (('n_edge', 'edges'), ('n_vert_edge', 'vert_edges'), ('n_cell', 'cell_keys'),
                      ('n_cell_seg', 'cell_segs'), ('n_string', 'strings')):
    h[field] = len(parts[part])
  h['n_point'], h['n_vert'] = len(parts['points']) // 2, len(parts['vert_first']) - 1
  # zeros after the strings bring the float64 sections to 8 bytes
  at = HEADER.itemsize
  for name, dtype, n in sections(h):
    at += np.dtype(dtype).itemsize * int(n)
    if name == 'strings':
      break
  pad = -at % 8
  parts = dict(parts, strings=np.r_[parts['strings'], np.zeros(pad, np.uint8)])
  h['n_string'] += np.uint32(pad)  # numpy 1.22 of the phone does not cast an int64 sum to uint32
  with open(path + '.tmp', 'wb') as f:
    f.write(h.tobytes())
    for name, dtype, n in sections(h):
      a = np.ascontiguousarray(parts[name], dtype)
      assert len(a) == n, name
      f.write(a.tobytes())
  os.replace(path + '.tmp', path)


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('pbf', help='OSM extract, .osm.pbf')
  parser.add_argument('out', help='the map file')
  parser.add_argument('--streets', action='append', default=[], metavar='LON,LAT,LON,LAT',
                      help=f'a box with every drivable street; Rostov and around it: {ROSTOV}')
  parser.add_argument('--all-streets', action='store_true', help='every drivable street of the extract')
  args = parser.parse_args()

  ways, osm_time = read_osm(args.pbf, streets=args.all_streets)
  for box in args.streets:
    ways = combine(ways, read_osm(args.pbf, box)[0])
  parts = build(ways)
  write(args.out, parts, osm_time)
  print(f'OSM data of {osm_time}: {len(ways)} ways, {len(parts["edges"])} edges, {len(parts["vert_first"]) - 1} junctions, '
        f'{len(parts["points"]) // 2} points, {len(parts["cell_keys"])} cells, {len(parts["cell_segs"])} cell entries, '
        f'{os.path.getsize(args.out) / 1e6:.1f} MB')


if __name__ == '__main__':
  main()
