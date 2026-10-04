"""The highway map: the OSM motorway and trunk roads of Russia with their links and the primary roads with a federal
number (the old M-4 through Aksay, parts of A-160), as a graph in one file. Inside the street areas given to the builder
(Rostov and around it) the file also has every drivable street.

tools/zp/osm_highways.py builds the file on the Mac, docs/speedcam/highways.md tells how. Little endian, in this order:
  header      HEADER
  edges       EDGE x n_edge; an edge runs between two junctions, a oneway edge only from v_from to v_to
  edge_first  uint32 x (n_edge + 1); the points of edge e are edge_first[e] .. edge_first[e + 1] - 1
  points      int32 x (n_point, 2): lat, lon in 1e-7 degrees, the end points are repeated in every edge
  vert_first  uint32 x (n_vert + 1); the edges at junction v are vert_edges[vert_first[v] .. vert_first[v + 1] - 1]
  vert_edges  uint32 x n_vert_edge
  cell_keys   uint32 x n_cell, sorted; a cell is CELL_E7 square, key = row * CELL_COLS + col from -90, -180
  cell_first  uint32 x (n_cell + 1)
  cell_segs   uint32 x n_cell_seg: segments that touch the cell; segment k runs from point k to point k + 1
  strings     n_string bytes of UTF-8, each string ends with 0; offset 0 is the empty string; zeros at the end bring the
              next section to 8 bytes
  along       float64 x n_point: m from the start of the edge to the point
  gal         float64 x n_point: the point on one axis where every edge has its own stretch, along + edge_off of its edge
  bear        float64 x n_point: degrees from north of the segment from the point to the next one
  edge_len    float64 x n_edge: m
  edge_off    float64 x n_edge: the start of the edge on the axis of gal, 1 m after the end of the edge before

positiond maps the file and reads only the pages it needs, so the builder computes along, gal and bear once
(docs/speedcam/position.md). A map published for the update of the phone is without the AXES sections, the phone
computes them again (complete, docs/speedcam/highways.md).
"""
import math
import os

import numpy as np

from openpilot.selfdrive.zeropilot.speedcam.geo import EARTH_RADIUS, distance

HIGHWAYS_PATH = '/data/media/0/zp_maps/highways.bin'
MAGIC = b'ZPHW'
VERSION = 2
# the OSM highway tag of the edge cls; the first HIGHWAY_CLASSES are taken everywhere, the rest only in the street areas
CLASSES = ('motorway', 'trunk', 'motorway_link', 'trunk_link', 'primary', 'primary_link',
           'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street')
HIGHWAY_CLASSES = 6
CELL_E7 = 500_000  # 0.05 degrees
CELL_COLS = 360 * 10**7 // CELL_E7

HEADER = np.dtype([('magic', 'S4'), ('version', '<u4'), ('n_edge', '<u4'), ('n_point', '<u4'), ('n_vert', '<u4'),
                   ('n_vert_edge', '<u4'), ('n_cell', '<u4'), ('n_cell_seg', '<u4'), ('n_string', '<u4'),
                   ('osm_time', 'S20')])  # osm_time: the time of the OSM data, 2026-09-29T20:21:02Z
EDGE = np.dtype([('v_from', '<u4'), ('v_to', '<u4'), ('ref', '<u4'), ('name', '<u4'), ('cls', 'u1'), ('oneway', 'u1'),
                 ('pad', '<u2')])  # ref, name: offsets in strings
AXES = ('along', 'gal', 'bear', 'edge_len', 'edge_off')  # the sections computed from the points


def cell_key(lat_e7, lon_e7):
  return (np.asarray(lat_e7, np.int64) + 90 * 10**7) // CELL_E7 * CELL_COLS + (np.asarray(lon_e7, np.int64) + 180 * 10**7) // CELL_E7


def sections(h):
  """Names, types and lengths of the parts after the header, in file order."""
  return [('edges', EDGE, h['n_edge']), ('edge_first', '<u4', h['n_edge'] + 1), ('points', '<i4', 2 * h['n_point']),
          ('vert_first', '<u4', h['n_vert'] + 1), ('vert_edges', '<u4', h['n_vert_edge']), ('cell_keys', '<u4', h['n_cell']),
          ('cell_first', '<u4', h['n_cell'] + 1), ('cell_segs', '<u4', h['n_cell_seg']), ('strings', 'u1', h['n_string']),
          ('along', '<f8', h['n_point']), ('gal', '<f8', h['n_point']), ('bear', '<f8', h['n_point']),
          ('edge_len', '<f8', h['n_edge']), ('edge_off', '<f8', h['n_edge'])]


def bearing(lat1, lon1, lat2, lon2):
  p1, p2 = np.radians(lat1), np.radians(lat2)
  dl = np.radians(lon2 - lon1)
  return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) % 360


def axes(points, edge_first) -> dict:
  """The AXES sections of the file from its points and edge_first: the axes of position.Graph, m along the edge, every
  edge on its own stretch of one axis, the heading of the segments."""
  lat, lon = points[0::2] * 1e-7, points[1::2] * 1e-7
  first = edge_first
  step = np.zeros(len(lat))
  step[1:] = distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
  step[first[:-1]] = 0
  along = np.cumsum(step)
  along -= np.repeat(along[first[:-1]], np.diff(first))
  edge_len = along[first[1:] - 1]
  edge_off = np.r_[0, np.cumsum(edge_len[:-1] + 1.0)]
  gal = along + np.repeat(edge_off, np.diff(first))
  bear = np.zeros(len(lat))
  bear[:-1] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
  return {'along': along, 'gal': gal, 'bear': bear, 'edge_len': edge_len, 'edge_off': edge_off}


def core_size(h) -> int:
  """Bytes of the file before the AXES sections: a published map."""
  at = HEADER.itemsize
  for name, dtype, n in sections(h):
    if name in AXES:
      break
    at += np.dtype(dtype).itemsize * int(n)
  return at


def complete(core_path: str, path: str) -> None:
  """Writes the map file from a published map without the AXES sections."""
  core = np.memmap(core_path, dtype=np.uint8, mode='r')
  h = np.frombuffer(core, HEADER, 1)[0]
  if h['magic'] != MAGIC or h['version'] != VERSION:
    raise ValueError(f'{core_path}: not a highway map of version {VERSION}')
  if core_size(h) != len(core):
    raise ValueError(f'{core_path}: {len(core)} bytes, the header says {core_size(h)}')
  parts, at = {}, HEADER.itemsize
  for name, dtype, n in sections(h):
    if name in AXES:
      break
    parts[name] = np.frombuffer(core, dtype, int(n), at)
    at += parts[name].nbytes
  computed = axes(parts['points'], parts['edge_first'])
  with open(path + '.tmp', 'wb') as f:
    f.write(core)
    for name in AXES:
      f.write(np.ascontiguousarray(computed[name], '<f8').tobytes())
  os.replace(path + '.tmp', path)


class Degrees:
  """Latitude or longitude of the points in degrees, from the 1e-7 degrees of the file, only for the points asked."""
  def __init__(self, e7):
    self.e7 = e7

  def __getitem__(self, i):
    return self.e7[i] * 1e-7

  def __len__(self):
    return len(self.e7)


class Highways:
  def __init__(self, path: str = HIGHWAYS_PATH):
    data = np.memmap(path, dtype=np.uint8, mode='r')
    h = np.frombuffer(data, HEADER, 1)[0]
    if h['magic'] != MAGIC or h['version'] != VERSION:
      raise ValueError(f'{path}: not a highway file of version {VERSION}')
    self.osm_time = h['osm_time'].decode()
    at = HEADER.itemsize
    for name, dtype, n in sections(h):
      a = np.frombuffer(data, dtype, int(n), at)
      setattr(self, name, a)
      at += a.nbytes
    if at != len(data):
      raise ValueError(f'{path}: {len(data)} bytes, the header says {at}')

    pts = self.points.reshape(-1, 2)
    self.lat, self.lon = Degrees(pts[:, 0]), Degrees(pts[:, 1])

  def edge_of(self, point):
    return np.searchsorted(self.edge_first, point, side='right') - 1

  def edge_length(self, e):
    return self.edge_len[e]

  def string(self, offset: int) -> str:
    end = offset
    while self.strings[end]:
      end += 1
    return self.strings[offset:end].tobytes().decode()

  def nearest(self, lat: float, lon: float, radius: float):
    """Segments within radius m of the point, nearest first.

    Returns arrays seg (the first point of the segment), edge, dist (m to the segment), along (m from the start of the
    edge to the foot of the perpendicular).
    """
    lat_e7, lon_e7 = round(lat * 1e7), round(lon * 1e7)
    rows = math.ceil(radius / (EARTH_RADIUS * math.radians(CELL_E7 * 1e-7)))
    cols = math.ceil(rows / max(math.cos(math.radians(lat)), 0.01))
    keys = (cell_key(lat_e7, lon_e7) + np.arange(-rows, rows + 1)[:, None] * CELL_COLS + np.arange(-cols, cols + 1)).ravel()
    i = np.searchsorted(self.cell_keys, keys)
    i = i[(i < len(self.cell_keys)) & (self.cell_keys[np.minimum(i, len(self.cell_keys) - 1)] == keys)]
    seg = np.unique(np.concatenate([self.cell_segs[self.cell_first[j]:self.cell_first[j + 1]] for j in i] or [[]])).astype(np.int64)

    # flat projection around the point, good enough at a few km
    kx = EARTH_RADIUS * math.radians(1) * math.cos(math.radians(lat))
    ky = EARTH_RADIUS * math.radians(1)
    ax, ay = (self.lon[seg] - lon) * kx, (self.lat[seg] - lat) * ky
    bx, by = (self.lon[seg + 1] - lon) * kx, (self.lat[seg + 1] - lat) * ky
    dx, dy = bx - ax, by - ay
    t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-9), 0, 1)
    dist = np.hypot(ax + t * dx, ay + t * dy)
    along = self.along[seg] + t * (self.along[seg + 1] - self.along[seg])

    keep = dist <= radius
    order = np.argsort(dist[keep], kind='stable')
    seg, dist, along = seg[keep][order], dist[keep][order], along[keep][order]
    return seg, self.edge_of(seg), dist, along

