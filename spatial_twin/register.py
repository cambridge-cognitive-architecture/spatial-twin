"""Registration at fixed scale.

scan_to_scan        3-D: FPFH + RANSAC for the global pose, then point-to-point
                    ICP at two voxel sizes.
scan_to_plan        2-D: an eye-height slice, a global FFT sweep over rotation
                    and translation, ICP on the best basins, and the residual
                    measured on wall-like runs only.
scan_to_scan_walls  2-D: the same search against another scan's walls, to check
                    a scan without trusting the drawing.

The public functions take millimetres. The algorithms work in metres, the
unit their thresholds are in.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.fft import irfft2, rfft2
from scipy.spatial import cKDTree

from .floor import local_floor
from .geometry import densify

PIX_COARSE, PIX = 0.10, 0.05
SLICE_LOW, SLICE_HIGH = 1.0, 1.7
MIN_RUN, NEAR_RUN = 2.0, 0.075
ICP_STEPS = (2.0, 1.0, 0.5, 0.3)
MM = 1000.0


@dataclass
class Transform:
    R: np.ndarray
    t: np.ndarray
    dz: float = 0.0
    residual: float = float("nan")  # median on wall-like points, mm
    inliers: float = float("nan")  # share of wall-like points within 150 mm
    scale: float = 1.0

    def __call__(self, points) -> np.ndarray:
        p = np.asarray(points, float)
        out = p.copy()
        out[:, :2] = self.scale * p[:, :2] @ self.R.T + self.t
        if p.shape[1] > 2:
            out[:, 2] = p[:, 2] + self.dz
        return out

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0])))

    def passes(self, gate_mm: float) -> bool:
        return bool(self.residual <= gate_mm)


@dataclass
class Transform3D:
    matrix: np.ndarray  # 4 x 4, mm
    fitness: float  # share of source points with a partner
    rmse: float  # over those points, mm

    def __call__(self, points) -> np.ndarray:
        return np.asarray(points, float) @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


# 2-D, metres

def _raster(xy, origin, shape, pix):
    g = np.zeros(shape, np.float32)
    ix = ((xy[:, 0] - origin[0]) / pix).astype(np.int64)
    iy = ((xy[:, 1] - origin[1]) / pix).astype(np.int64)
    k = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    g[iy[k], ix[k]] = 1.0
    return g


def icp_2d(source, tree: cKDTree, threshold: float, iterations: int = 80):
    current = source.copy()
    R, t = np.eye(2), np.zeros(2)
    for _ in range(iterations):
        d, i = tree.query(current, workers=-1)
        keep = d < threshold
        if keep.sum() < 100:
            break
        a, b = current[keep], tree.data[i[keep]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        r = Vt.T @ U.T
        if np.linalg.det(r) < 0:
            Vt[-1] *= -1
            r = Vt.T @ U.T
        step = cb - r @ ca
        current = current @ r.T + step
        R, t = r @ R, r @ t + step
        if np.linalg.norm(step) < 1e-5:
            break
    return current, R, t


def register_2d(xy, target, n_basins: int = 6):
    """(R, t, moved). Global FFT sweep in 1 degree steps, then ICP on the best basins."""
    tree = cKDTree(target)
    low, high = target.min(0) - 12.0, target.max(0) + 12.0
    shape = (int((high[1] - low[1]) / PIX_COARSE) + 1, int((high[0] - low[0]) / PIX_COARSE) + 1)
    F = rfft2(cv2.dilate(_raster(target, low, shape, PIX_COARSE), np.ones((5, 5), np.uint8)))
    centre = xy.mean(0)
    sub = (xy - centre)[::7]
    middle = np.array(shape[::-1]) * PIX_COARSE / 2 + low
    candidates = []
    for deg in np.arange(0, 360, 1.0):
        turned = sub @ rotation(np.radians(deg)).T
        corr = irfft2(rfft2(_raster(turned + middle, low, shape, PIX_COARSE)) * np.conj(F), s=shape)
        j = int(np.argmax(corr))
        dy, dx = np.unravel_index(j, shape)
        dy -= shape[0] if dy > shape[0] // 2 else 0
        dx -= shape[1] if dx > shape[1] // 2 else 0
        candidates.append((float(corr.flat[j]), deg, np.array([-dx * PIX_COARSE, -dy * PIX_COARSE])))
    candidates.sort(key=lambda c: -c[0])

    best = None
    for _, deg, shift in candidates[:n_basins]:
        R = rotation(np.radians(deg))
        t = middle + shift - R @ centre
        current = xy @ R.T + t
        for threshold in ICP_STEPS:
            current, dR, dt = icp_2d(current, tree, threshold)
            R, t = dR @ R, dR @ t + dt
        median = float(np.median(tree.query(current, workers=-1)[0]))
        if best is None or median < best[0]:
            best = (median, R, t, current)
    return best[1], best[2], best[3]


def wall_candidates(xy) -> np.ndarray:
    """Points on straight runs of at least 2 m: walls, not furniture."""
    low = xy.min(0) - 1.0
    image = np.zeros((int((xy[:, 1].max() - low[1]) / PIX) + 3,
                      int((xy[:, 0].max() - low[0]) / PIX) + 3), np.uint8)
    image[((xy[:, 1] - low[1]) / PIX).astype(int), ((xy[:, 0] - low[0]) / PIX).astype(int)] = 255
    lines = cv2.HoughLinesP(image, 1, np.pi / 360, int(MIN_RUN / PIX * 0.6),
                            minLineLength=int(MIN_RUN / PIX), maxLineGap=int(0.25 / PIX))
    if lines is None:
        return np.zeros(len(xy), bool)
    samples = []
    # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
    for x0, y0, x1, y1 in lines.reshape(-1, 4).astype(float) * PIX + np.tile(low, 2):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / PIX))
        samples.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
    d, _ = cKDTree(np.vstack(samples)).query(xy, distance_upper_bound=NEAR_RUN, workers=-1)
    return np.isfinite(d)


def eye_slice(points_m, voxel: float = 0.05) -> tuple[np.ndarray, float]:
    """(XY of the 1.0-1.7 m band above the local floor, the median floor height)."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_m, float))
    cloud = cloud.voxel_down_sample(voxel)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    p = np.asarray(cloud.points)
    floor, grid = local_floor(p)
    height = p[:, 2] - floor
    return p[(height >= SLICE_LOW) & (height <= SLICE_HIGH)][:, :2], float(np.median(grid))


def _fit_walls(xy, target, n_basins) -> Transform:
    if min(len(xy), len(target)) < 100:
        raise ValueError("under 100 points at eye height: too few to register")
    R, t, moved = register_2d(xy, target, n_basins)
    walls = wall_candidates(moved)
    d = cKDTree(target).query(moved[walls])[0] if walls.any() else np.array([])
    return Transform(R, t * MM,
                     residual=float(np.median(d) * MM) if d.size else float("nan"),
                     inliers=float(np.mean(d < 0.15)) if d.size else float("nan"))


def scan_to_plan(scan, segments, n_basins: int = 6) -> Transform:
    xy, floor = eye_slice(np.asarray(scan, float) / MM)
    fit = _fit_walls(xy, densify(segments, PIX * MM) / MM, n_basins)
    fit.dz = -floor * MM
    return fit


def scan_to_scan_walls(source, target, n_basins: int = 4) -> Transform:
    xy, floor = eye_slice(np.asarray(source, float) / MM)
    target_xy, target_floor = eye_slice(np.asarray(target, float) / MM)
    fit = _fit_walls(xy, target_xy[wall_candidates(target_xy)], n_basins)
    fit.dz = (target_floor - floor) * MM
    return fit


# 3-D

def trim(points, percent: float = 0.2) -> np.ndarray:
    """Drop the outer percent on every axis: stray returns stretch the extents."""
    low = np.percentile(points, percent, axis=0)
    high = np.percentile(points, 100 - percent, axis=0)
    return points[np.all((points >= low) & (points <= high), axis=1)]


def scan_to_scan(source, target, voxel_m: float = 0.40, refine_m=(0.20, 0.10),
                 seed: int = 0) -> Transform3D:
    """3-D rigid fit. A low fitness means do not trust it, not that the place changed."""
    import open3d as o3d

    registration = o3d.pipelines.registration
    o3d.utility.random.seed(seed)

    def cloud(points):
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(np.asarray(points, float))
        return c

    def features(full):
        q = cloud(trim(np.asarray(full.voxel_down_sample(voxel_m).points)))
        q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_m * 2, 30))
        return q, registration.compute_fpfh_feature(
            q, o3d.geometry.KDTreeSearchParamHybrid(voxel_m * 5, 100))

    src, dst = cloud(np.asarray(source, float) / MM), cloud(np.asarray(target, float) / MM)
    small_src, f_src = features(src)
    small_dst, f_dst = features(dst)
    reach = voxel_m * 1.5
    result = registration.registration_ransac_based_on_feature_matching(
        small_src, small_dst, f_src, f_dst, True, reach,
        registration.TransformationEstimationPointToPoint(False), 3,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         registration.CorrespondenceCheckerBasedOnDistance(reach)],
        registration.RANSACConvergenceCriteria(400000, 0.999))
    for v in refine_m:
        result = registration.registration_icp(
            src.voxel_down_sample(v), dst.voxel_down_sample(v), v * 3, result.transformation,
            registration.TransformationEstimationPointToPoint(),
            registration.ICPConvergenceCriteria(max_iteration=50))
    matrix = np.array(result.transformation)
    matrix[:3, 3] *= MM
    return Transform3D(matrix, float(result.fitness), float(result.inlier_rmse * MM))


def fit_similarity(source, target, reflect: bool = False) -> Transform:
    """Least-squares similarity (Umeyama) from paired landmarks, e.g. pixels to mm."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    x, y = a - a.mean(0), b - b.mean(0)
    U, s, Vt = np.linalg.svd(x.T @ y / len(a))
    d = np.sign(np.linalg.det(U @ Vt))
    S = np.diag([1.0, -d if reflect else d])
    R = (U @ S @ Vt).T
    scale = float((s * np.diag(S)).sum() * len(a) / (x ** 2).sum())
    t = b.mean(0) - scale * R @ a.mean(0)
    residual = np.median(np.linalg.norm(scale * a @ R.T + t - b, axis=1))
    return Transform(R, t, residual=float(residual), scale=scale)
