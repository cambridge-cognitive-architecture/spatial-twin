"""Segments, sampling and rasters."""
from __future__ import annotations

import numpy as np


def polyline_segments(points, closed: bool = False) -> np.ndarray:
    """(N, 4) segments from an (M, 2) polyline."""
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    return np.column_stack([p[:-1], p[1:]])


def densify(segments, step: float = 100) -> np.ndarray:
    """Points every step mm along each segment."""
    out = []
    for x0, y0, x1, y1 in np.asarray(segments, float).reshape(-1, 4):
        n = int(np.ceil(np.hypot(x1 - x0, y1 - y0) / step)) + 1
        out.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
    return np.vstack(out) if out else np.empty((0, 2))


def raster(segments, x0, y0, cell, width, height) -> np.ndarray:
    grid = np.zeros((height, width), bool)
    i, j = np.floor((densify(segments, cell * 0.4) - [x0, y0]) / cell).astype(int).T
    ok = (i >= 0) & (i < width) & (j >= 0) & (j < height)
    grid[j[ok], i[ok]] = True
    return grid


def distance_to_segments(xy, segments, step: float = 50) -> np.ndarray:
    from scipy.spatial import cKDTree

    return cKDTree(densify(segments, step)).query(xy)[0]
