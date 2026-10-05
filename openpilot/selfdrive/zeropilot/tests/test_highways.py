#!/usr/bin/env python3
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from openpilot.selfdrive.zeropilot import highways
from openpilot.selfdrive.zeropilot.highways import AXES, HEADER, Highways, axes, axes_parts, complete, core_size
from openpilot.selfdrive.zeropilot.speedcam.geo import EARTH_RADIUS, distance
from openpilot.tools.zp.osm_highways import Way, build, combine, oneway, wanted, write

M_LAT = 1 / (EARTH_RADIUS * math.pi / 180)  # degrees of latitude in 1 m


def way(nodes, pts, cls=0, direction=1, ref='', name=''):
  lat, lon = zip(*pts)
  return Way(nodes, np.round(np.array(lat) * 1e7).astype(np.int32), np.round(np.array(lon) * 1e7).astype(np.int32),
             cls, direction, ref, name)


def load(ways):
  with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, 'highways.bin')
    write(path, build(ways), '2026-09-29T20:21:02Z')
    return Highways(path)


class TestOneway(unittest.TestCase):
  def test_tags(self):
    self.assertEqual(oneway({'highway': 'trunk', 'oneway': 'yes'}), 1)
    self.assertEqual(oneway({'highway': 'trunk', 'oneway': '-1'}), -1)
    self.assertEqual(oneway({'highway': 'motorway', 'oneway': 'no'}), 0)

  def test_implied(self):
    self.assertEqual(oneway({'highway': 'motorway'}), 1)
    self.assertEqual(oneway({'highway': 'motorway_link'}), 1)
    self.assertEqual(oneway({'highway': 'trunk', 'junction': 'roundabout'}), 1)
    self.assertEqual(oneway({'highway': 'trunk'}), 0)
    self.assertEqual(oneway({'highway': 'trunk_link'}), 0)


class TestWanted(unittest.TestCase):
  def test_classes(self):
    self.assertTrue(wanted({'highway': 'trunk'}))
    self.assertTrue(wanted({'highway': 'motorway_link'}))
    self.assertFalse(wanted({'highway': 'secondary', 'ref': 'М-4'}))

  def test_primary_needs_a_federal_number(self):
    self.assertTrue(wanted({'highway': 'primary', 'ref': 'М-4'}))
    self.assertTrue(wanted({'highway': 'primary', 'ref': 'A-160'}))  # Latin letter
    self.assertTrue(wanted({'highway': 'primary_link', 'ref': '03К-401;Р-217'}))
    self.assertFalse(wanted({'highway': 'primary', 'ref': '60К-1'}))
    self.assertFalse(wanted({'highway': 'primary'}))

  def test_street_box(self):
    self.assertTrue(wanted({'highway': 'residential'}, streets=True))
    self.assertTrue(wanted({'highway': 'primary'}, streets=True))  # no federal number needed
    self.assertFalse(wanted({'highway': 'residential'}))
    self.assertFalse(wanted({'highway': 'service'}, streets=True))
    self.assertFalse(wanted({'highway': 'footway'}, streets=True))


class TestCombine(unittest.TestCase):
  def test_a_way_of_both_passes_goes_in_once(self):
    hw = way([1, 2], [(45., 39.), (45., 39.01)])._replace(id=10)
    street = way([2, 3], [(45., 39.01), (45.01, 39.01)], cls=11, direction=0)._replace(id=11)
    ways = combine([hw], [hw, street])
    self.assertEqual([w.id for w in ways], [10, 11])
    h = load(ways)
    self.assertEqual(list(h.edges['cls']), [0, 11])
    self.assertEqual(h.edges['v_to'][0], h.edges['v_from'][1])  # the street meets the highway at node 2


class TestHighways(unittest.TestCase):
  # a motorway to the east with a trunk road to the north from its middle node
  EAST = way([1, 2, 3], [(45., 39.), (45., 39.01), (45., 39.02)], ref='М-4', name='Дон')
  NORTH = way([2, 4], [(45., 39.01), (45.01, 39.01)], cls=1, direction=0, ref='А-146')

  def test_junctions_cut_the_ways(self):
    hw = load([self.EAST, self.NORTH])
    self.assertEqual(hw.osm_time, '2026-09-29T20:21:02Z')
    self.assertEqual(len(hw.edges), 3)
    self.assertEqual(len(hw.vert_first) - 1, 4)
    e = hw.edges
    self.assertEqual(e['v_to'][0], e['v_from'][1])  # node 2
    self.assertEqual(e['v_from'][2], e['v_from'][1])
    mid = e['v_from'][1]
    self.assertEqual(sorted(hw.vert_edges[hw.vert_first[mid]:hw.vert_first[mid + 1]]), [0, 1, 2])
    self.assertEqual(list(e['oneway']), [1, 1, 0])
    self.assertEqual(list(e['cls']), [0, 0, 1])
    self.assertEqual([hw.string(e['ref'][i]) for i in range(3)], ['М-4', 'М-4', 'А-146'])
    self.assertEqual(hw.string(e['name'][2]), '')
    self.assertAlmostEqual(hw.edge_length(0), distance(45., 39., 45., 39.01), delta=0.01)

  def test_reversed_way(self):
    hw = load([way([1, 2], [(45., 39.), (45., 39.01)], direction=-1)])
    self.assertEqual(hw.edges['oneway'][0], 1)
    self.assertAlmostEqual(hw.lon[0], 39.01)
    self.assertAlmostEqual(hw.lon[1], 39.)

  def test_nearest(self):
    hw = load([self.EAST, self.NORTH])
    seg, edge, dist, along = hw.nearest(45. + 10 * M_LAT, 39.005, 50)
    self.assertEqual(list(edge), [0])
    self.assertAlmostEqual(dist[0], 10, delta=0.05)
    self.assertAlmostEqual(along[0], distance(45., 39., 45., 39.005), delta=0.5)

    seg, edge, dist, along = hw.nearest(45., 39.01, 5)
    self.assertEqual(sorted(edge), [0, 1, 2])
    self.assertTrue(np.all(dist < 0.01))

    self.assertEqual(len(hw.nearest(45. + 60 * M_LAT, 39.005, 50)[0]), 0)

  def test_nearest_in_the_next_cell(self):
    # the segment is below the cell border at 45.05, the point is 20 m above it
    hw = load([way([1, 2], [(45.049, 39.), (45.049, 39.01)])])
    seg, edge, dist, along = hw.nearest(45.05 + 20 * M_LAT, 39.005, 200)
    self.assertEqual(list(edge), [0])
    self.assertAlmostEqual(dist[0], 0.001 / M_LAT + 20, delta=0.5)

  def test_long_segment(self):
    # one straight segment through many cells, found from its middle
    hw = load([way([1, 2], [(45., 39.), (45.3, 39.4)], direction=0)])
    self.assertGreater(len(hw.cell_keys), 10)
    seg, edge, dist, along = hw.nearest(45.15, 39.2, 100)
    self.assertEqual(list(edge), [0])
    self.assertLess(dist[0], 100)

  def test_axes(self):
    # the builder computes what position.Graph computed at every start before the file version 2
    hw = load([self.EAST, self.NORTH])
    for e in range(3):
      k = np.arange(hw.edge_first[e], hw.edge_first[e + 1])
      self.assertEqual(hw.along[k[0]], 0)
      self.assertEqual(hw.along[k[-1]], hw.edge_len[e])
      np.testing.assert_array_equal(hw.gal[k], hw.along[k] + hw.edge_off[e])
    self.assertAlmostEqual(hw.edge_len[2], distance(45., 39.01, 45.01, 39.01), delta=0.01)
    np.testing.assert_array_equal(hw.edge_off, np.r_[0, np.cumsum(hw.edge_len[:-1] + 1.0)])
    self.assertAlmostEqual(hw.bear[hw.edge_first[0]], 90, delta=0.01)
    self.assertAlmostEqual(hw.bear[hw.edge_first[2]], 0, delta=0.01)

  @staticmethod
  def grid():
    # 4 roads to the east and 4 to the north through the same 16 nodes: 24 edges
    ways = []
    for i in range(4):
      ways.append(way([10 * i + j for j in range(4)], [(45 + 0.01 * i, 39 + 0.01 * j) for j in range(4)], direction=0))
      ways.append(way([10 * j + i for j in range(4)], [(45 + 0.01 * j, 39 + 0.01 * i) for j in range(4)], direction=0))
    return ways

  def test_axes_in_parts(self):
    # the sums run on from one part to the next: parts of any size give the axes of the whole map
    parts = build(self.grid())
    self.assertEqual(len(parts['edges']), 24)
    whole = axes(parts['points'], parts['edge_first'])
    for chunk in (1, 5, 24, 100):
      got = list(axes_parts(parts['points'], parts['edge_first'], chunk))
      self.assertEqual([e0 for e0, _, _ in got], list(range(0, 24, chunk)))
      self.assertEqual([p0 for _, p0, _ in got], list(parts['edge_first'][0:24:chunk]))
      for name in AXES:
        np.testing.assert_array_equal(np.concatenate([p[name] for _, _, p in got]), whole[name], f'{name}, {chunk} edges')

  def test_complete_in_parts(self):
    # the phone completes the published map part by part, the result is the file of the builder
    with tempfile.TemporaryDirectory() as tmp:
      full, done = os.path.join(tmp, 'full.bin'), os.path.join(tmp, 'done.bin')
      write(full, build(self.grid()), '2026-09-29T20:21:02Z')
      with open(full, 'rb') as f:
        data = f.read()
      with open(done, 'wb') as f:
        f.write(data[:core_size(np.frombuffer(data, HEADER, 1)[0])])
      with mock.patch.object(highways, 'AXES_CHUNK', 5):
        complete(done)
      with open(done, 'rb') as f:
        self.assertEqual(f.read(), data)

  def test_sections_map_the_file(self):
    hw = load([self.EAST, self.NORTH])
    for name in ('edges', 'edge_first', 'points', 'vert_first', 'vert_edges', 'cell_keys', 'cell_first', 'cell_segs',
                 'strings', 'along', 'gal', 'bear', 'edge_len', 'edge_off'):
      self.assertIsInstance(getattr(hw, name).base, np.memmap, name)
    self.assertTrue(np.shares_memory(hw.lat.e7, hw.points))
    self.assertEqual(hw.gal.ctypes.data % 8, 0)


if __name__ == '__main__':
  unittest.main()
