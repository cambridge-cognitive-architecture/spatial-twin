"""Position grain: where a point stands relative to thresholds and anchors.

Radial statistics and visual depth are Isovist.properties(): radial min, max,
mean (visual depth), variance and skew of the ray lengths, in metres.
"""
from __future__ import annotations

import numpy as np

from .isovist import Isovist, cast


def relational(x_mm: float, y_mm: float, isovist: Isovist, thresholds,
               anchors: dict[str, tuple[float, float]] | None = None,
               max_range_mm: float = 50_000, visible_tol_mm: float = 400.0) -> dict:
    """Distance and bearing to the nearest threshold, visible thresholds, anchor distances.

    thresholds: (N, 2) one point per threshold, mm. A threshold is visible when
    the isovist ray at its bearing reaches at least as far as it is, so this
    agrees with the isovist by construction. Bearing is degrees, 0 = +x,
    counter-clockwise, in plan coordinates.
    """
    out: dict = {}
    doors = np.asarray(thresholds, float).reshape(-1, 2)
    if len(doors):
        d = np.hypot(doors[:, 0] - x_mm, doors[:, 1] - y_mm)
        j = int(np.argmin(d))
        out["threshold_dist_m"] = float(d[j] / 1000.0)
        out["threshold_bearing_deg"] = float(np.degrees(np.arctan2(doors[j, 1] - y_mm, doors[j, 0] - x_mm)) % 360.0)
        ang = np.arctan2(doors[:, 1] - y_mm, doors[:, 0] - x_mm) % (2 * np.pi)
        n = len(isovist.rays_m)
        k = np.round(ang / (2 * np.pi) * n).astype(int) % n
        reach = isovist.rays_m[k] * 1000.0
        out["thresholds_visible_n"] = int(((d <= reach + visible_tol_mm) & (d <= max_range_mm)).sum())
    else:
        out.update(threshold_dist_m=np.nan, threshold_bearing_deg=np.nan, thresholds_visible_n=0)
    for name, (ax, ay) in (anchors or {}).items():
        key = "".join(c if c.isalnum() else "_" for c in name.lower())
        out[f"dist_to_{key}_m"] = float(np.hypot(ax - x_mm, ay - y_mm) / 1000.0)
    return out


def position_measures(x_mm: float, y_mm: float, segments, thresholds, anchors=None,
                      n_rays: int = 720, max_range_mm: float = 50_000) -> tuple[dict, Isovist]:
    """Isovist properties (prefixed iso_) and relational measures for one point."""
    v = cast(x_mm, y_mm, segments, n_rays=n_rays, max_range_mm=max_range_mm)
    out = {f"iso_{k}": val for k, val in v.properties().items()}
    out["visual_depth_m"] = out["iso_radial_mean_m"]
    out.update(relational(x_mm, y_mm, v, thresholds, anchors, max_range_mm))
    return out, v
