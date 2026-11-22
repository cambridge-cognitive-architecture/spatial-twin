"""Objects by shape.

  1. the band 0.1-2.0 m above the local floor, on a 5 cm voxel cloud without
     statistical outliers;
  2. walls out: a straight run of at least 2 m is removed where it also spans
     at least 1.2 m of height, so low furniture against a wall stays;
  3. 3-D connected components on 10 cm cells, of at least 25 points;
  4. each component classed from its own extents, along the plan's axes:

     person-sized     height 1.1-2.1 m, longer side 0.25-0.90 m
     bed-sized        height 0.5-1.2 m, footprint 1.5-2.5 x 0.7-1.1 m
     equipment-sized  height 1.2-2.0 m, footprint under 1 m2
     other            everything else

The names are sizes, not recognition: a coat stand is person-sized.
Millimetres in and out. The thresholds are in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .floor import local_floor

BAND_LOW, BAND_HIGH = 0.10, 2.00
VOXEL, CELL = 0.05, 0.10
MIN_POINTS = 25
RUN, RUN_NEAR, RUN_HEIGHT = 2.0, 0.12, 1.20
KINDS = ("person-sized", "bed-sized", "equipment-sized", "other")
MM = 1000.0


@dataclass
class ShapeObject:
    kind: str
    centre: np.ndarray
    length: float
    width: float
    height: float
    points: np.ndarray  # (N, 3): plan x, plan y, height above the local floor


def classify(height: float, side_a: float, side_b: float) -> str:
    short, long = sorted([side_a, side_b])
    if 1.1 <= height <= 2.1 and 0.25 <= long <= 0.90:
        return "person-sized"
    if 0.5 <= height <= 1.2 and 1.5 <= long <= 2.5 and 0.7 <= short <= 1.1:
        return "bed-sized"
    if 1.2 <= height <= 2.0 and short * long < 1.0:
        return "equipment-sized"
    return "other"


def object_band(points_m, transform=None) -> tuple[np.ndarray, np.ndarray]:
    """(plan XY, height above the local floor). The floor is found in the scan's own frame."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points_m, float))
    cloud = cloud.voxel_down_sample(VOXEL)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    p = np.asarray(cloud.points)
    height = p[:, 2] - local_floor(p)[0]
    keep = (height >= BAND_LOW) & (height <= BAND_HIGH)
    xy = p[keep][:, :2]
    if transform is not None:
        xy = transform(xy)
    return xy, height[keep]


def planar_mask(xy, height, pixel: float = 0.05) -> np.ndarray:
    """True where a point belongs to a long, straight, tall run: a wall."""
    out = np.zeros(len(xy), bool)
    if not len(xy):
        return out
    origin = xy.min(0) - 1.0
    image = np.zeros((int((xy[:, 1].max() - origin[1]) / pixel) + 3,
                      int((xy[:, 0].max() - origin[0]) / pixel) + 3), np.uint8)
    image[((xy[:, 1] - origin[1]) / pixel).astype(int),
          ((xy[:, 0] - origin[0]) / pixel).astype(int)] = 255
    lines = cv2.HoughLinesP(image, 1, np.pi / 360, int(RUN / pixel * 0.6),
                            minLineLength=int(RUN / pixel), maxLineGap=int(0.25 / pixel))
    if lines is None:
        return out
    tree = cKDTree(xy)
    # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4)
    for x0, y0, x1, y1 in lines.reshape(-1, 4).astype(float) * pixel + np.tile(origin, 2):
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 0.05))
        samples = np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
        near = set()
        for found in tree.query_ball_point(samples, RUN_NEAR):
            near.update(found)
        if not near:
            continue
        idx = np.fromiter(near, int)
        if height[idx].max() - height[idx].min() >= RUN_HEIGHT:
            out[idx] = True
    return out


def components(xy, height) -> list[tuple[str, tuple[float, float, float], np.ndarray]]:
    """(kind, (height, x extent, y extent), point indices) per component."""
    if not len(xy):
        return []
    k = np.floor(np.column_stack([xy, height]) / CELL).astype(np.int64)
    k -= k.min(0)
    occupied = np.zeros((k[:, 2].max() + 1, k[:, 1].max() + 1, k[:, 0].max() + 1), bool)
    occupied[k[:, 2], k[:, 1], k[:, 0]] = True
    labels, n = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    owner = labels[k[:, 2], k[:, 1], k[:, 0]]
    sizes = np.bincount(owner, minlength=n + 1)
    order = np.argsort(owner, kind="stable")
    start = np.searchsorted(owner[order], np.arange(n + 2))
    out = []
    for i, where in enumerate(ndimage.find_objects(labels), start=1):
        if where is None or sizes[i] < MIN_POINTS:
            continue
        dz, dy, dx = ((s.stop - s.start) * CELL for s in where)
        out.append((classify(dz, dx, dy), (dz, dx, dy), order[start[i]:start[i + 1]]))
    return out


def find_objects(points, transform=None) -> list[ShapeObject]:
    """points: the scan in mm. transform: scan to plan, as scan_to_plan returns it."""
    move = None if transform is None else (lambda xy: transform(xy * MM) / MM)
    xy, height = object_band(np.asarray(points, float) / MM, move)
    free = ~planar_mask(xy, height)
    xy, height = xy[free], height[free]
    out = []
    for kind, (dz, dx, dy), idx in components(xy, height):
        width, length = sorted([dx, dy])
        out.append(ShapeObject(kind, xy[idx].mean(0) * MM, length * MM, width * MM, dz * MM,
                               np.column_stack([xy[idx], height[idx]]) * MM))
    return out
