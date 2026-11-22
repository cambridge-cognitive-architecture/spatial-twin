# spatial-twin

Turns a floor plan and point clouds into a spatial twin.

- **Registration:** scan to scan in 3-D (FPFH + RANSAC, then ICP), and scan to plan or to another scan's walls (FFT sweep over every pose, then ICP, scored on wall runs). Scale is fixed.
- **Isovists:** area, perimeter, compactness, occlusivity, radial statistics and drift.
- **Walkable space:** where a person can stand, clear of barriers.
- **Objects by shape:** long, straight, tall runs are removed as walls, and what remains 0.1–2.0 m above the local floor is clustered in 3-D and labelled person-sized, bed-sized, equipment-sized or other.
- **Twin figure:** plan, isovist field and scan in one projection.

![Twin of a synthetic layout and point cloud](docs/demo.png)

## Install

Tested with Python 3.12.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick example

```sh
python -m spatial_twin.demo
```

```python
from spatial_twin import cast, scan_to_plan, synthetic

scan = synthetic.cloud()
fit = scan_to_plan(scan, synthetic.WALLS)
print(fit.residual, fit.inliers, fit.passes(gate_mm=200))
print(cast(4500, 4000, synthetic.WALLS).properties()["area_m2"])
```

## Limits

Units are millimetres and the 2-D fits need levelled scans. **A good residual is not proof of a correct pose.** In our synthetic tests at 74% overlap, the wall-only fit was turned or shifted in every trial and still passed a 150 mm gate. Below about 90% overlap, or in symmetrical spaces, check each fit against a known landmark. A low `fitness` means the 3-D fit is not to be trusted. Shape labels are sizes, not object recognition, and are measured along the plan's axes, so a rotated object looks bigger. Short wall pieces can survive as objects, and tall objects in a straight run can be removed as wall. Leaving out person-sized objects is **not anonymisation**: people who are seated, lying, grouped or against a wall will remain.

## Licence and credit

MIT, see [LICENSE](LICENSE). Developed by the Cambridge Cognitive Architecture Lab.
