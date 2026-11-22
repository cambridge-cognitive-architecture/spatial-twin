"""Where a person can stand."""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt, label

from .geometry import raster


def walkable_mask(envelope, barriers, cell: float = 500, clearance: float = 250,
                  min_patch_m2: float = 2, fine: float = 100):
    """(mask, grid x, grid y): inside, clear of barriers, not a tiny island."""
    ring = np.asarray(envelope, float)
    x0, y0 = ring.min(0)
    x1, y1 = ring.max(0)
    gx, gy = np.arange(x0 + cell / 2, x1, cell), np.arange(y0 + cell / 2, y1, cell)
    X, Y = np.meshgrid(gx, gy)
    inside = Path(ring).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    width, height = int((x1 - x0) / fine) + 1, int((y1 - y0) / fine) + 1
    blocked = raster(barriers, x0, y0, fine, width, height)
    if blocked.any():
        distance = distance_transform_edt(~blocked) * fine
        inside &= distance[((Y - y0) / fine).astype(int), ((X - x0) / fine).astype(int)] > clearance
    patches, n = label(inside)
    areas = np.bincount(patches.ravel()) * (cell / 1000) ** 2
    mask = np.isin(patches, [i for i in range(1, n + 1) if areas[i] >= min_patch_m2])
    return mask, gx, gy
