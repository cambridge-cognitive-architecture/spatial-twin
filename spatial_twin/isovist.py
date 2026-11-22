"""360-degree isovists by uniform ray casting against wall segments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Isovist:
    origin_m: np.ndarray
    rays_m: np.ndarray
    polygon_m: np.ndarray
    n_open: int  # rays that met no wall

    def occlusivity(self, jump_mm: float = 300) -> float:
        """Share of the perimeter made of radial edges: what bounds hidden space."""
        length = np.linalg.norm(np.roll(self.polygon_m, -1, axis=0) - self.polygon_m, axis=1)
        total = length.sum()
        jump = np.abs(np.roll(self.rays_m, -1) - self.rays_m) > jump_mm / 1000
        return float(length[jump].sum() / total) if total > 1e-12 else 0.0

    def properties(self) -> dict[str, float]:
        p, r = self.polygon_m, self.rays_m
        x, y = p.T
        nx, ny = np.roll(x, -1), np.roll(y, -1)
        cross = x * ny - nx * y
        signed = cross.sum() / 2
        area = float(abs(signed))
        perimeter = float(np.hypot(nx - x, ny - y).sum())
        if area > 1e-12:
            centroid = np.array([((x + nx) * cross).sum(), ((y + ny) * cross).sum()]) / (6 * signed)
        else:
            centroid = p.mean(0)
        eig = np.clip(np.linalg.eigvalsh(np.cov(p.T)), 1e-12, None)
        m, s = r.mean(), r.std()
        return dict(
            area_m2=area, perimeter_m=perimeter,
            compactness=float(4 * np.pi * area / perimeter ** 2) if perimeter > 1e-9 else 0.0,
            occlusivity=self.occlusivity(),
            radial_min_m=float(r.min()), radial_max_m=float(r.max()), radial_mean_m=float(m),
            radial_var_m2=float(r.var()),
            radial_skew=float(((r - m) ** 3).mean() / s ** 3) if s > 1e-9 else 0.0,
            drift_m=float(np.linalg.norm(centroid - self.origin_m)),
            elongation=float(np.sqrt(eig[1] / eig[0])),
            open_ray_frac=self.n_open / len(r),
        )


def cast(x_mm: float, y_mm: float, segments, n_rays: int = 720,
         max_range_mm: float = 50_000) -> Isovist:
    a = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    dx, dy = np.cos(a), np.sin(a)
    depth = np.full(n_rays, float(max_range_mm))
    s = np.asarray(segments, float).reshape(-1, 4)
    if len(s):
        # P + t d = A + u e, with t >= 0 and u in [0, 1]
        px, py, ex, ey = s[:, 0] - x_mm, s[:, 1] - y_mm, s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        den = dx[:, None] * ey - dy[:, None] * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (px * ey - py * ex) / den
            u = (px * dy[:, None] - py * dx[:, None]) / den
        hit = (np.abs(den) > 1e-12) & (t > 1e-9) & (u >= 0) & (u <= 1)
        depth = np.minimum(depth, np.where(hit, t, np.inf).min(axis=1))
    origin = np.array([x_mm, y_mm]) / 1000
    r = depth / 1000
    return Isovist(origin, r, origin + np.column_stack([r * dx, r * dy]),
                   int((depth >= max_range_mm - 1e-6).sum()))
