"""The exploded axonometric: plan, isovist field and scan in one frame."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from .isovist import cast

ROT_DEG, SQUASH = 12.0, 0.52
KIND_COLOURS = {"person-sized": "#7b4fa8", "bed-sized": "#e0903c",
                "equipment-sized": "#2f6db5", "other": "#3d9970"}
INK = "#4a4d55"


def project(x, y, layer=0.0, z=0.0):
    """One projection for every layer, so the layers cannot drift apart."""
    a = np.radians(ROT_DEG)
    u = np.asarray(x) * np.cos(a) - np.asarray(y) * np.sin(a)
    v = (np.asarray(x) * np.sin(a) + np.asarray(y) * np.cos(a)) * SQUASH + np.asarray(z) + layer
    return u, v


def twin_figure(walls, rooms, walkable, points, objects, standpoint,
                include_person_sized: bool = False):
    """Person-sized objects are left out of the drawn scan unless asked for."""
    import matplotlib.pyplot as plt

    mask, gx, gy = walkable
    ends = np.vstack([walls[:, :2], walls[:, 2:]])
    lo, hi = ends.min(0) - 400, ends.max(0) + 400
    plate = np.array([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])])
    gap = np.ptp(project(*plate.T)[1]) * 1.15
    layers = [0, gap, 2 * gap]
    fig, ax = plt.subplots(figsize=(8, 9.5), layout="constrained")

    def flat(polygon, layer, **kw):
        ax.fill(*project(*np.asarray(polygon).T, layer), **kw)

    def wall_lines(layer, order):
        start = np.column_stack(project(walls[:, 0], walls[:, 1], layer))
        end = np.column_stack(project(walls[:, 2], walls[:, 3], layer))
        ax.add_collection(LineCollection(np.stack([start, end], 1), colors=INK,
                                         linewidths=1.1, zorder=order))

    for layer, name in zip(layers, ("plan + walkable space", "isovist area", "scan + objects by shape")):
        order = layer / gap * 10
        flat(plate, layer, color="#f6f6f8", ec="#c9c9d0", zorder=order)
        u, v = project(*plate[0], layer)
        ax.text(u, v - gap * 0.07, name, ha="left", va="top", fontsize=10, color=INK)

    for name, polygon in rooms.items():
        flat(polygon, 0, color="#dbe6f3", zorder=1)
        ax.text(*project(*np.mean(polygon, axis=0), 0), name, ha="center", fontsize=9, zorder=4)
    X, Y = np.meshgrid(gx, gy)
    ax.scatter(*project(X[mask], Y[mask]), s=2, color="#6f93bd", lw=0, zorder=2)
    wall_lines(0, 3)

    half = (gx[1] - gx[0]) / 2
    cells, areas = [], []
    for x, y in zip(X[mask], Y[mask]):
        corners = np.array([(x - half, y - half), (x + half, y - half),
                            (x + half, y + half), (x - half, y + half)])
        cells.append(np.column_stack(project(*corners.T, gap)))
        areas.append(cast(x, y, walls, n_rays=180).properties()["area_m2"])
    field = PolyCollection(cells, array=np.array(areas), cmap="YlGnBu", lw=0, zorder=11)
    ax.add_collection(field)
    seen = cast(*standpoint, walls).polygon_m * 1000
    flat(seen, gap, fc="#c0392b22", ec="#c0392b", lw=1.5, zorder=12)
    wall_lines(gap, 13)
    fig.colorbar(field, ax=ax, shrink=0.25, label="visible area (m²)")

    show = points[:, 2] <= 2600
    drawn = []
    for o in objects:
        if o.kind != "person-sized" or include_person_sized:
            drawn.append(o)
        else:
            # the whole column, feet included
            show[np.hypot(*(points[:, :2] - o.centre).T) <= o.length / 2 + 200] = False
    u, v = project(points[show, 0], points[show, 1], layers[2], points[show, 2] * 0.55)
    ax.scatter(u, v, c="#b9bec7", s=0.8, lw=0, rasterized=True, zorder=21)
    for o in drawn:
        u, v = project(o.points[:, 0], o.points[:, 1], layers[2], o.points[:, 2] * 0.55)
        ax.scatter(u, v, c=KIND_COLOURS[o.kind], s=1.6, lw=0, rasterized=True, zorder=22)
    for kind in sorted({o.kind for o in drawn}):
        ax.scatter([], [], c=KIND_COLOURS[kind], s=25, label=kind)
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    return fig
