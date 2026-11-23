"""Flow: how the view changes along a track.

Every rate is gated: a derivative is taken only between two consecutive valid
moments no further apart than max_gap_factor times the typical step. The step
(flow_dt_s) is then the denominator for every rate, so nothing is ever computed
across a gap.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def polygon_area_m2(p) -> float:
    x, y = np.asarray(p, float).T
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2)


def new_information(cur, prev, grid_m: float = 0.5) -> float:
    """Share of what is visible now that was hidden at the previous moment.

    area(current minus previous) / area(current), by sampling the current isovist
    on a regular grid and testing each point against the previous one. Sampling
    rather than exact clipping: the polygons have hundreds of vertices, are often
    non-convex and self-touch where rays graze a corner. Heading-free: both
    isovists are the full 360 degrees.
    """
    if cur is None or prev is None or len(cur) < 3 or len(prev) < 3:
        return float("nan")
    cur, prev = np.asarray(cur, float), np.asarray(prev, float)
    (x0, y0), (x1, y1) = cur.min(0), cur.max(0)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return float("nan")
    gx, gy = np.arange(x0, x1 + grid_m, grid_m), np.arange(y0, y1 + grid_m, grid_m)
    if len(gx) * len(gy) > 400_000:
        return float("nan")
    pts = np.c_[np.repeat(gx, len(gy)), np.tile(gy, len(gx))]
    inside = Path(cur).contains_points(pts)
    n = int(inside.sum())
    if n == 0:
        return float("nan")
    return float(1.0 - Path(prev).contains_points(pts[inside]).sum() / n)


def flow_measures(t_s, x_mm, y_mm, polygons, valid=None, max_gap_factor: float = 2.5) -> dict:
    """Per-moment flow_dt_s, flow_new_information, flow_area_rate and flow_speed_mps.

    polygons: one isovist polygon (metres) or None per moment. valid: moments
    whose position is real (not a gap); invalid moments break the chain.
    """
    t, x, y = (np.asarray(a, float) for a in (t_s, x_mm, y_mm))
    n = len(t)
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    out = {c: np.full(n, np.nan) for c in
           ("flow_dt_s", "flow_new_information", "flow_area_rate", "flow_speed_mps")}
    if n < 2:
        return out
    dts = np.diff(np.sort(t[np.isfinite(t)]))
    typical = float(np.median(dts)) if len(dts) else 0.5
    if not 0.01 < typical < 10.0:
        typical = 0.5
    for i in range(1, n):
        if not (valid[i] and valid[i - 1]):
            continue
        dt = t[i] - t[i - 1]
        if not np.isfinite(dt) or dt <= 0 or dt > max_gap_factor * typical:
            continue
        pa, pb = polygons[i - 1], polygons[i]
        out["flow_dt_s"][i] = dt
        out["flow_new_information"][i] = new_information(pb, pa)
        if pa is not None and pb is not None:
            out["flow_area_rate"][i] = (polygon_area_m2(pb) - polygon_area_m2(pa)) / dt
        out["flow_speed_mps"][i] = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / 1000.0 / dt
    return out


def rate(values, flow_dt_s) -> np.ndarray:
    """Rate of change per second of a level column, gated by flow_dt_s.

    Used for flow_occlusivity_rate (from iso_occlusivity) and flow_depth_rate
    (from iso_radial_mean_m).
    """
    v, dt = np.asarray(values, float), np.asarray(flow_dt_s, float)
    d = np.full(len(v), np.nan)
    d[1:] = v[1:] - v[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(dt) & (dt > 0), d / dt, np.nan)
