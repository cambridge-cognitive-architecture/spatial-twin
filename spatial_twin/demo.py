"""Register two synthetic scans, then draw the twin."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import synthetic as S
from .figure import twin_figure
from .register import scan_to_plan, scan_to_scan
from .shapes import find_objects
from .walkable import walkable_mask


def run():
    a, b, _, _ = S.two_scans()
    second = scan_to_scan(b, a)
    merged = np.vstack([a, second(b)])
    first = scan_to_plan(merged, S.WALLS)
    points = first(merged)
    walkable = walkable_mask(S.ENVELOPE, S.WALLS, cell=250)
    return second, first, points, find_objects(merged, first), walkable


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("docs/demo.png"))
    a = ap.parse_args()
    second, first, points, objects, walkable = run()
    fig = twin_figure(S.WALLS, S.ROOMS, walkable, points, objects, (4500, 4000))
    fig.suptitle("Synthetic layout and point cloud", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110, facecolor="white", metadata={"Software": None})
    print(f"scan to scan {second.rmse:.0f} mm rmse, scan to plan {first.residual:.0f} mm "
          f"median on walls; {len(objects)} objects; wrote {a.out}")


if __name__ == "__main__":
    main()
