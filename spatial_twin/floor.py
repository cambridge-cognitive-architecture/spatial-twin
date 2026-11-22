"""The local floor, so that height means height above the floor right there."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def local_floor(points_m, cell: float = 1.0, percentile: float = 5.0):
    """(floor under each point, the floor grid). Metres in and out."""
    p = np.asarray(points_m)
    key = np.floor(p[:, :2] / cell).astype(np.int64)
    idx = key - key.min(0)
    nx, ny = idx[:, 0].max() + 1, idx[:, 1].max() + 1
    flat = idx[:, 1] * nx + idx[:, 0]
    order = np.argsort(flat)
    cells, z = flat[order], p[order, 2]
    bounds = np.searchsorted(cells, np.arange(nx * ny + 1))
    grid = np.full(nx * ny, np.nan)
    for c in range(nx * ny):
        a, b = bounds[c], bounds[c + 1]
        if b - a >= 8:
            grid[c] = np.percentile(z[a:b], percentile)
    g = grid.reshape(ny, nx)
    if np.isnan(g).all():
        raise ValueError("too few points to find a floor")
    missing = np.isnan(g)
    if missing.any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # windows with no floor at all
            filled = ndimage.generic_filter(g, np.nanmedian, size=5, mode="nearest")
        g = np.where(missing, filled, g)
    g = np.where(np.isnan(g), np.nanmedian(g[~np.isnan(g)]), g)
    g = ndimage.median_filter(g, size=3, mode="nearest")
    return g[idx[:, 1], idx[:, 0]], g
