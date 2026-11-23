# spatial-twin

Turns a floor plan and point clouds into a spatial twin.

Author: Salome Egli

- **Registration:** scan to scan in 3-D (FPFH + RANSAC, then ICP), and scan to plan or to another scan's walls (FFT sweep over every pose, then ICP, scored on wall runs). Scale is fixed.
- **Isovists:** area, perimeter, compactness, occlusivity, radial statistics and drift.
- **Walkable space:** where a person can stand, clear of barriers.
- **Objects by shape:** long, straight, tall runs are removed as walls, and what remains 0.1–2.0 m above the local floor is clustered in 3-D and labelled person-sized, bed-sized, equipment-sized or other.
- **Spatial metrics at three grains:** position, zone and field of view, plus space syntax and flow. See below.
- **Twin figure:** plan, isovist field and scan in one projection.

![Twin of a synthetic layout and point cloud](docs/demo.png)

## Install

Tested with Python 3.12.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or as a package: `pip install "spatial-twin @ git+https://github.com/cambridge-cognitive-architecture/spatial-twin"`. Scan registration and objects by shape also need `open3d` (`pip install "spatial-twin[scans] @ git+..."`).

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

## Spatial metrics at three grains

<!-- Three-grains diagram goes here: docs/three_grains.png -->

Lengths go in as millimetres in plan coordinates; isovist measures come out in metres and square metres.

**Position** (`position.py`, `isovist.py`): what can be seen from one point.

| Measure | Function |
|---|---|
| isovist area, perimeter, compactness, occlusivity, drift, elongation, open-ray share | `cast(x, y, segments).properties()` |
| radial statistics: min, max, mean, variance and skew of the ray lengths | `properties()`: `radial_*` |
| visual depth: the mean ray length | `position_measures`: `visual_depth_m` |
| distance and bearing to the nearest threshold; thresholds visible from the point | `relational` |
| distance to each anchor | `relational`: `dist_to_<anchor>_m` |

A threshold counts as visible when the isovist ray at its bearing reaches it (within 400 mm), so visibility agrees with the isovist by construction.

**Zone** (`zones.py`): a drawn zone as a whole. Zones are rasterised once (`ZoneMap.from_polygons`), so area, perimeter and the points isovists are cast from all agree.

| Measure | Function |
|---|---|
| envelope: floor area, perimeter, jaggedness (perimeter / perimeter of the circle with the same area), convexity (area / convex hull area), aspect ratio, median width | `ZoneMap.envelope` |
| isovist statistics across the zone: mean, variance and skew of occlusivity, visual depth, compactness and drift; mean isovist area; deepest view; open-ray share | `zone_isovist_stats` |
| boundary configuration: doors between each pair of zones, and doors per zone | `door_graph`, `threshold_counts` |
| permeability: which zones connect, from door evidence plus wall-free reachability, with the gate width and basis of each link | `permeable_graph` |

`zone_graph` gives plain adjacency (zones that touch). It links two zones drawn flush either side of a solid wall, and `permeable_graph` does not. `permeable_graph` gives a zone drawn in separate pieces one node per piece, so no shortest path can teleport between them.

**Space syntax** (`syntax.py`) on the zone graph: connectivity, mean depth, RA and RRA (normalised by the diamond value), integration (1/RRA; undefined, not infinite, for a zone one step from every other, where RRA = 0), local integration at radius 3, choice (betweenness), control, and the system's intelligibility (correlation of connectivity with integration).

**Flow** (`flow.py`): how the view changes along a track.

| Measure | Function |
|---|---|
| new-information rate: the share of the current isovist that was hidden at the previous moment | `new_information`, `flow_measures` |
| speed, rate of change of isovist area | `flow_measures` |
| rate of change of occlusivity, or of any other isovist measure | `rate` |

A rate is taken only between consecutive valid moments no more than 2.5 times the typical step apart, so nothing is computed across a gap.

Semantic measures of the scene video, such as class shares and clutter, are not here. They belong to the compute chain of field-workbench (in preparation).

### Planned

These are specified but not yet implemented:

- **Space syntax on a visibility graph (VGA):** integration, connectivity, choice, mean depth and intelligibility over a grid of visible points. Today's measures use the zone graph.
- **Directional isovist and directional-to-full ratio:** a cone of about 120° along the heading. This needs head pose in plan coordinates.
- **Bearing to anchors:** today only distance is given.
- **A single permeability index per zone:** today permeability is a graph, with its doors, gates and evidence.

## Limits

Units are millimetres and the 2-D fits need levelled scans. **A good residual is not proof of a correct pose.** In our synthetic tests at 74% overlap, the wall-only fit was turned or shifted in every trial and still passed a 150 mm gate. Below about 90% overlap, or in symmetrical spaces, check each fit against a known landmark. A low `fitness` means the 3-D fit is not to be trusted. Shape labels are sizes, not object recognition, and are measured along the plan's axes, so a rotated object looks bigger. Short wall pieces can survive as objects, and tall objects in a straight run can be removed as wall. Leaving out person-sized objects is **not anonymisation**: people who are seated, lying, grouped or against a wall will remain.

## Part of the toolkit

Five tools for field studies with wearable sensors:

[field-logger](https://github.com/cambridge-cognitive-architecture/field-logger) → [wearable-pipeline](https://github.com/cambridge-cognitive-architecture/wearable-pipeline) → **spatial-twin** → field-workbench (in preparation) → [egoanon](https://github.com/cambridge-cognitive-architecture/egoanon)

<!-- Overview diagram goes here: docs/toolkit.png -->

| Tool | What it does |
|---|---|
| [field-logger](https://github.com/cambridge-cognitive-architecture/field-logger) | tablet app: location by threshold crossings, tasks, interruptions and ratings, on one clock |
| [wearable-pipeline](https://github.com/cambridge-cognitive-architecture/wearable-pipeline) | eye tracking, ECG and the field log on one clock; pupil and HRV preprocessing; quality control |
| **spatial-twin** (this repository) | LiDAR scans against the floor plan: registration, isovists, walkable space, objects |
| field-workbench (in preparation) | one local app from project set-up through coding and compute to the master table |
| [egoanon](https://github.com/cambridge-cognitive-architecture/egoanon) | anonymisation of the head-mounted scene video |

## Licence and credit

MIT, see [LICENSE](LICENSE). Developed by the Cambridge Cognitive Architecture Lab.
