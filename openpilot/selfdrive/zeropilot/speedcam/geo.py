"""Distance and direction on the sphere, the same formulas as in the GPS Antiradar app."""
import math

import numpy as np

EARTH_RADIUS = 6372795.  # m


def distance(lat1, lng1, lat2, lng2):
  """Great-circle distance in m. Takes numbers or arrays."""
  p1, p2 = np.radians(lat1), np.radians(lat2)
  dl = np.radians(lng2) - np.radians(lng1)
  y = np.hypot(np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl), np.sin(dl) * np.cos(p2))
  x = np.cos(p1) * np.cos(p2) * np.cos(dl) + np.sin(p1) * np.sin(p2)
  return np.arctan2(y, x) * EARTH_RADIUS


def bearing(lat_from: float, lng_from: float, lat_to: float, lng_to: float) -> float:
  """Direction from one point to another, degrees clockwise from north, 0..360."""
  p_from, p_to = math.radians(lat_from), math.radians(lat_to)
  east = math.cos((p_from + p_to) / 2) * math.radians(lng_to - lng_from)
  return math.degrees(math.atan2(east, p_to - p_from)) % 360


def ang_diff(a: float, b: float) -> float:
  """Smallest angle between two directions, degrees."""
  return abs((a - b + 180) % 360 - 180)
