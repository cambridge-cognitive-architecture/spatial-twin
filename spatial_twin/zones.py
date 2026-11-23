"""Zone grain: envelope, isovist statistics across a zone, and how zones connect.

Zones are drawn polygons in plan millimetres. They are rasterised once into a
label grid; every zone measure reads that grid, so a zone's area, its perimeter
and the points its isovists are cast from all agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt
from scipy.ndimage import label as connected
from scipy.spatial import ConvexHull

from .isovist import cast


@dataclass
class ZoneMap:
    names: list[str]
    grid: np.ndarray          # 0 = no zone, i + 1 = names[i]
    res_mm: float
    origin_mm: tuple[float, float]

    @classmethod
    def from_polygons(cls, polygons: dict[str, np.ndarray], res_mm: float = 100.0,
                      margin_mm: float = 0.0) -> "ZoneMap":
        """Rasterise named polygons. Where two overlap, the first one listed wins."""
        names = list(polygons)
        pts = np.vstack([np.asarray(p, float) for p in polygons.values()])
        (x0, y0), (x1, y1) = pts.min(0) - margin_mm, pts.max(0) + margin_mm
        W, H = int(np.ceil((x1 - x0) / res_mm)), int(np.ceil((y1 - y0) / res_mm))
        X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) * res_mm, y0 + (np.arange(H) + 0.5) * res_mm)
        centres = np.c_[X.ravel(), Y.ravel()]
        grid = np.zeros(H * W, dtype="int32")
        for i, n in enumerate(names):
            inside = Path(np.asarray(polygons[n], float)).contains_points(centres)
            grid[(grid == 0) & inside] = i + 1
        return cls(names, grid.reshape(H, W), float(res_mm), (float(x0), float(y0)))

    def mask(self, zone: str) -> np.ndarray:
        return self.grid == self.names.index(zone) + 1

    def zone_at(self, x_mm, y_mm) -> np.ndarray:
        i = ((np.asarray(x_mm, float) - self.origin_mm[0]) // self.res_mm).astype(int)
        j = ((np.asarray(y_mm, float) - self.origin_mm[1]) // self.res_mm).astype(int)
        H, W = self.grid.shape
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        v = np.zeros(np.shape(i), dtype="int32")
        v[ok] = self.grid[j[ok], i[ok]]
        lut = np.array([""] + self.names, dtype=object)
        return lut[v]

    def points_in(self, zone: str, spacing_mm: float = 1500.0) -> np.ndarray:
        """Cell centres of the zone on a regular grid, in mm."""
        m = self.mask(zone)
        step = max(1, int(round(spacing_mm / self.res_mm)))
        jj, ii = np.nonzero(m[::step, ::step])
        return np.c_[self.origin_mm[0] + (ii * step + 0.5) * self.res_mm,
                     self.origin_mm[1] + (jj * step + 0.5) * self.res_mm]

    def centroid(self, zone: str) -> tuple[float, float]:
        jj, ii = np.nonzero(self.mask(zone))
        return (self.origin_mm[0] + (ii.mean() + 0.5) * self.res_mm,
                self.origin_mm[1] + (jj.mean() + 0.5) * self.res_mm)

    # ---- envelope
    def envelope(self, zone: str) -> dict:
        """Floor area, raster perimeter, jaggedness, convexity and aspect ratio.

        jaggedness = perimeter / perimeter of the circle with the same area
        convexity  = area / area of the convex hull
        """
        m = self.mask(zone)
        if not m.any():
            return {}
        area = float(m.sum()) * (self.res_mm / 1000.0) ** 2
        p = np.pad(m, 1)
        edges = ((p[1:-1, 1:-1] & ~p[:-2, 1:-1]).sum() + (p[1:-1, 1:-1] & ~p[2:, 1:-1]).sum()
                 + (p[1:-1, 1:-1] & ~p[1:-1, :-2]).sum() + (p[1:-1, 1:-1] & ~p[1:-1, 2:]).sum())
        perim = float(edges) * self.res_mm / 1000.0
        jj, ii = np.nonzero(m)
        pts = np.c_[self.origin_mm[0] + (ii + 0.5) * self.res_mm,
                    self.origin_mm[1] + (jj + 0.5) * self.res_mm] / 1000.0
        convexity, aspect = _hull_measures(pts, area)
        equiv = 2.0 * np.sqrt(np.pi * area)
        width = 2.0 * float(np.median(distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1][m])) \
            * self.res_mm / 1000.0
        return {"zone": zone, "floor_area_m2": area, "perimeter_m": perim,
                "jaggedness": perim / equiv if equiv > 0 else 0.0,
                "convexity": convexity, "aspect_ratio": aspect, "median_width_m": width}


def _hull_measures(pts_m: np.ndarray, area_m2: float) -> tuple[float, float]:
    if len(pts_m) > 4000:
        pts_m = pts_m[np.linspace(0, len(pts_m) - 1, 4000).astype(int)]
    try:
        h = ConvexHull(pts_m)
        convexity = float(area_m2 / h.volume) if h.volume > 0 else 0.0
    except Exception:
        convexity = 0.0
    q = pts_m - pts_m.mean(0)
    ev = np.clip(np.linalg.eigvalsh(q.T @ q / max(len(q), 1)), 1e-12, None)
    return convexity, float(np.sqrt(ev[1] / ev[0]))


def _moments(v, name: str) -> dict:
    a = np.asarray(v, float)
    s = a.std()
    return {f"{name}_mean": float(a.mean()), f"{name}_var": float(a.var()),
            f"{name}_skew": float(((a - a.mean()) ** 3).mean() / s ** 3) if s > 1e-9 else 0.0}


def zone_isovist_stats(zones: ZoneMap, zone: str, segments, spacing_mm: float = 1500.0,
                       n_rays: int = 360, max_points: int = 400,
                       max_range_mm: float = 100_000.0) -> dict:
    """Isovists cast on a grid across the zone, summarised.

    Mean, variance and skew of occlusivity, visual depth (each isovist's mean
    radial length), compactness and drift; mean isovist area; the deepest view;
    and the share of rays that met nothing, which is ~0 once the plan is closed.
    """
    pts = zones.points_in(zone, spacing_mm)
    if not len(pts):
        return {"zone": zone, "n_samples": 0}
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    props = [cast(x, y, segments, n_rays=n_rays, max_range_mm=max_range_mm).properties() for x, y in pts]
    col = lambda k: [p[k] for p in props]
    out = {"zone": zone, "n_samples": int(len(pts))}
    out.update(_moments(col("occlusivity"), "occlusivity"))
    out.update(_moments(col("radial_mean_m"), "visual_depth"))
    out.update(_moments(col("compactness"), "compactness"))
    out.update(_moments(col("drift_m"), "drift"))
    out["isovist_area_mean_m2"] = float(np.mean(col("area_m2")))
    out["visual_depth_max_m"] = float(np.max(col("radial_max_m")))
    out["open_ray_frac"] = float(np.mean(col("open_ray_frac")))
    return out


# ---------------------------------------------------------------- connectivity
def zone_graph(zones: ZoneMap, gap_mm: float = 600.0) -> dict[str, list[str]]:
    """Adjacency: zones whose rasters come within gap_mm of each other.

    This is adjacency, not proven permeability: two zones drawn flush either side
    of a solid wall are linked. Use permeable_graph where walls are known.
    """
    k = max(1, int(round(gap_mm / zones.res_mm)))
    masks = {n: zones.mask(n) for n in zones.names}
    grown = {n: binary_dilation(m, iterations=k) for n, m in masks.items()}
    adj = {n: [] for n in zones.names}
    for a in zones.names:
        for b in zones.names:
            if a < b and (grown[a] & masks[b]).any():
                adj[a].append(b); adj[b].append(a)
    return adj


def wall_raster(zones: ZoneMap, segments, seal_mm: float = 200.0) -> np.ndarray:
    """Walls drawn into the zone grid, with hairline cracks under seal_mm closed."""
    res, (x0, y0) = zones.res_mm, zones.origin_mm
    H, W = zones.grid.shape
    walls = np.zeros((H, W), dtype=bool)
    for a, b, c, d in np.asarray(segments, float).reshape(-1, 4):
        n = max(2, int(np.hypot(c - a, d - b) / (res * 0.4)))
        i = ((np.linspace(a, c, n) - x0) / res).astype(int)
        j = ((np.linspace(b, d, n) - y0) / res).astype(int)
        ok = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        walls[j[ok], i[ok]] = True
    k = max(1, int(round(seal_mm / res)))
    pad = k + 1
    return binary_closing(np.pad(walls, pad), iterations=k)[pad:-pad, pad:-pad]


def door_graph(zones: ZoneMap, doors, walls: np.ndarray | None = None,
               max_probe_mm: float = 2000.0, labels: np.ndarray | None = None,
               names: list[str] | None = None) -> dict[tuple[str, str], int]:
    """{(zone a, zone b): number of doors} from door segments.

    From each door's midpoint, march perpendicular to it on both sides, up to
    max_probe_mm, and take the first zone met. Two different zones mean the door
    sits between them. The march stops at a wall when `walls` is given, so a
    probe cannot tunnel through a partition into the room beyond. A door drawn
    open lies inside one room, finds the same zone twice and adds nothing.
    """
    grid = zones.grid if labels is None else labels
    lut = list(zones.names) if names is None else list(names)
    res, (ox, oy) = zones.res_mm, zones.origin_mm
    H, W = grid.shape
    step = res * 0.7
    n_steps = int(max_probe_mm / step)

    def march(mx, my, ux, uy):
        for k in range(1, n_steps + 1):
            i, j = int((mx + ux * step * k - ox) / res), int((my + uy * step * k - oy) / res)
            if not (0 <= i < W and 0 <= j < H):
                return None
            if walls is not None and walls[j, i]:
                return None
            if grid[j, i] > 0:
                return lut[int(grid[j, i]) - 1]
        return None

    counts: dict[tuple[str, str], int] = {}
    for x0, y0, x1, y1 in np.asarray(doors, float).reshape(-1, 4):
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L < 200.0:
            continue
        nx, ny, mx, my = -dy / L, dx / L, (x0 + x1) / 2, (y0 + y1) / 2
        a, b = march(mx, my, nx, ny), march(mx, my, -nx, -ny)
        if a and b and a != b:
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def threshold_counts(pair_counts: dict[tuple[str, str], int], zones: list[str]) -> dict[str, int]:
    """Doors per zone: the per-pair counts of door_graph, summed for each zone."""
    out = {z: 0 for z in zones}
    for (a, b), n in pair_counts.items():
        out[a] = out.get(a, 0) + n
        out[b] = out.get(b, 0) + n
    return out


def permeable_graph(zones: ZoneMap, walls_segments, doors, bridge_mm: float = 700.0,
                    seal_mm: float = 200.0, split_disconnected: bool = True,
                    min_piece_m2: float = 2.0, min_gate_mm: float = 350.0) -> dict:
    """Connectivity from the drawing: door evidence plus wall-free reachability.

    bridge_mm   how far a zone may reach across unassigned floor to meet its
                neighbour. Hand-drawn polygons rarely abut exactly. The reach
                grows only through free pixels, so it can go round the end of a
                wall but never over one.
    seal_mm     closes hairline cracks in the wall raster before the flood, kept
                far below a doorway's width.
    min_gate_mm the clear width a link with no door behind it must have. A flood
                cannot tell a doorway from a crack where two wall segments fail
                to meet; a door-backed link is kept whatever its gate.
    split_disconnected
                gives each separately drawn piece of a zone its own node,
                "name #1", "name #2". One node for pieces at opposite ends of a
                building is a teleport: shortest paths run in one piece and out
                of the other, and the zone looks far more integrated than it is.

    Returns adjacency, per-link evidence (doors, walkable, gate_mm, basis),
    nodes, the zone of each node, pieces per zone, the node grid and isolated nodes.
    """
    res = zones.res_mm
    H, W = zones.grid.shape
    walls = wall_raster(zones, walls_segments, seal_mm)
    free = ~walls
    px_m2 = (res / 1000.0) ** 2
    nodes, zone_of, pieces = [], {}, {}
    node_grid = np.zeros((H, W), dtype="int32")
    for zi, zn in enumerate(zones.names):
        m = zones.grid == zi + 1          # the drawn outline, not outline minus walls
        lab, k = connected(m)
        keep = [c for c in range(1, k + 1) if (lab == c).sum() * px_m2 >= min_piece_m2]
        if not keep:
            continue
        pieces[zn] = len(keep)
        multi = split_disconnected and len(keep) > 1
        for r, c in enumerate(sorted(keep, key=lambda c: -(lab == c).sum())):
            name = f"{zn} #{r + 1}" if multi else zn
            nodes.append(name); zone_of[name] = zn
            node_grid[lab == c] = len(nodes)
            if not multi:
                node_grid[m] = len(nodes)
                break
    node_free = node_grid * free
    masks = {n: node_free == i + 1 for i, n in enumerate(nodes)}
    k_bridge = max(1, int(round(bridge_mm / res)))
    reach = {}
    for n, m in masks.items():
        g = m.copy()
        for _ in range(k_bridge):
            g = binary_dilation(g) & free
        reach[n] = g
    door_counts = door_graph(zones, doors, walls=walls, labels=node_grid, names=nodes)

    adj = {n: [] for n in nodes}
    evidence = {}
    for ia, a in enumerate(nodes):
        for b in nodes[ia + 1:]:
            key = (a, b) if a < b else (b, a)
            n_doors = door_counts.get(key, 0)
            linked = False
            if (reach[a] & reach[b]).any():
                lab, _ = connected(reach[a] | reach[b])
                la = set(np.unique(lab[masks[a]]).tolist()) - {0}
                lb = set(np.unique(lab[masks[b]]).tolist()) - {0}
                linked = bool(la & lb)
            if not (linked or n_doors):
                continue
            gate_mm = 0.0
            if linked:
                # the overlap band is ~2 * bridge_mm deep: area / depth ~ clear width
                gate_mm = int((reach[a] & reach[b] & free).sum()) * res * res / (2.0 * max(bridge_mm, res))
                if not n_doors and gate_mm < min_gate_mm:
                    continue
            adj[a].append(b); adj[b].append(a)
            evidence[f"{key[0]}|{key[1]}"] = {
                "doors": int(n_doors), "walkable": bool(linked), "gate_mm": round(gate_mm),
                "basis": "door" if n_doors and linked else "door-only" if n_doors else "opening"}
    return {"adjacency": adj, "evidence": evidence, "nodes": nodes, "zone_of": zone_of,
            "pieces": pieces, "node_grid": node_grid, "res_mm": res, "origin_mm": zones.origin_mm,
            "door_counts": door_counts, "isolated": sorted(n for n, v in adj.items() if not v)}
