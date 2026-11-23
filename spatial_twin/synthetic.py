"""A made-up flat of three rooms and a hallway, and two scans of it."""
from __future__ import annotations

import numpy as np

from .geometry import densify
from .register import Transform, rotation

ROOMS = {
    "Room A": [(0, 5000), (5000, 5000), (5000, 9000), (0, 9000)],
    "Room B": [(5000, 5000), (12000, 5000), (12000, 8000), (5000, 8000)],
    "Room C": [(0, 0), (7000, 0), (7000, 3000), (0, 3000)],
    "Hallway": [(0, 3000), (12000, 3000), (12000, 5000), (0, 5000)],
}
ENVELOPE = np.array([(0, 0), (7000, 0), (7000, 3000), (12000, 3000), (12000, 8000),
                     (5000, 8000), (5000, 9000), (0, 9000)], float)
WALLS = np.array([
    (0, 0, 7000, 0), (7000, 0, 7000, 3000), (7000, 3000, 12000, 3000),
    (12000, 3000, 12000, 8000), (12000, 8000, 5000, 8000), (5000, 8000, 5000, 9000),
    (5000, 9000, 0, 9000), (0, 9000, 0, 0),
    (0, 5000, 2000, 5000), (3100, 5000, 5000, 5000), (5000, 5000, 5000, 8000),
    (5000, 5000, 8000, 5000), (9100, 5000, 12000, 5000),
    (0, 3000, 4000, 3000), (5100, 3000, 7000, 3000),
], float)
# the three openings in the walls above, as door segments
DOORS = np.array([(2000, 5000, 3100, 5000), (8000, 5000, 9100, 5000), (4000, 3000, 5100, 3000)], float)
# x0, y0, x1, y1, bottom, top
OBJECTS = [
    (1000, 6500, 2600, 7300, 600, 750), (6500, 6100, 8500, 7000, 0, 900),
    (10450, 6250, 11000, 7250, 0, 1900), (1200, 1200, 2700, 1330, 0, 1500),
    (5600, 1100, 6050, 1550, 0, 1750), (8000, 3700, 8900, 4300, 2300, 2450),
]


def cloud(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = densify(WALLS, 70)
    z = np.arange(0, 2600, 120)
    walls = np.c_[np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))]
    X, Y = np.meshgrid(np.arange(100, 12000, 180), np.arange(100, 9000, 180))
    floor = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    from matplotlib.path import Path
    parts = [walls, floor[Path(ENVELOPE).contains_points(floor[:, :2])]]
    for x0, y0, x1, y1, low, high in OBJECTS:
        parts.append(rng.uniform([x0, y0, low], [x1, y1, high], (1600, 3)))
    p = np.vstack(parts)
    return p + rng.normal(0, 8, p.shape)


def two_scans():
    """Overlapping halves in their own frames, with the true transforms.

    Returns first, second, truth (first to plan), truth (second to first).
    """
    p = cloud()
    a = p[p[:, 0] < 11100]
    b = p[p[:, 0] > 1900]
    first = Transform(rotation(np.radians(24)), np.array([3100.0, -1700.0]), 420)
    between = Transform(rotation(np.radians(-67)), np.array([-2500.0, 5200.0]), -260)

    def undo(points, tf):
        out = points.copy()
        out[:, :2] = (points[:, :2] - tf.t) @ tf.R
        out[:, 2] -= tf.dz
        return out
    return undo(a, first), undo(undo(b, first), between), first, between
