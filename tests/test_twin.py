import numpy as np
import pytest

from spatial_twin import cast, fit_similarity, scan_to_scan_walls, synthetic
from spatial_twin.demo import run
from spatial_twin.geometry import polyline_segments
from spatial_twin.ply import load
from spatial_twin.register import rotation
from spatial_twin.shapes import planar_mask


@pytest.fixture(scope="module")
def result():
    return run()


def test_scan_to_scan(result):
    truth = synthetic.two_scans()[3]
    fit = result[0]
    assert abs(fit.rotation_deg - truth.rotation_deg) < 0.5
    shift = fit.matrix[:3, 3]
    assert np.linalg.norm(shift[:2] - truth.t) < 100 and abs(shift[2] - truth.dz) < 30
    assert fit.fitness > 0.6


def test_scan_to_scan_walls():
    a, b, _, truth = synthetic.two_scans()
    fit = scan_to_scan_walls(b, a)
    assert abs(fit.rotation_deg - truth.rotation_deg) < 0.5
    assert np.linalg.norm(fit.t - truth.t) < 100 and fit.passes(150)


def test_scan_to_plan(result):
    truth = synthetic.two_scans()[2]
    fit = result[1]
    assert abs(fit.rotation_deg - truth.rotation_deg) < 0.5
    assert np.linalg.norm(fit.t - truth.t) < 100 and abs(fit.dz - truth.dz) < 50


def test_objects_by_shape(result):
    expected = {(1800, 6900): "other", (7500, 6550): "bed-sized", (10725, 6750): "equipment-sized",
                (1950, 1265): "equipment-sized", (5825, 1325): "person-sized"}
    for where, kind in expected.items():
        near = min(result[3], key=lambda o: np.linalg.norm(o.centre - where))
        assert np.linalg.norm(near.centre - where) < 400 and near.kind == kind, where


def test_band_excludes_what_hangs_above_two_metres(result):
    assert not any(np.linalg.norm(o.centre - (8450, 4000)) < 400 for o in result[3])


def test_walls_are_removed_but_low_furniture_against_one_stays():
    x, z = [a.ravel() for a in np.meshgrid(np.arange(0, 4, 0.05), np.arange(0.1, 2.0, 0.05))]
    wall = planar_mask(np.c_[x, np.zeros_like(x)], z)
    assert wall.mean() > 0.95
    low = z < 0.9
    assert not planar_mask(np.c_[x[low], np.zeros(low.sum())], z[low]).any()


def test_walkable(result):
    mask, gx, gy = result[4]
    assert mask[np.abs(gy - 4000).argmin(), np.abs(gx - 4000).argmin()]
    assert not mask[np.abs(gy - 1000).argmin(), np.abs(gx - 10000).argmin()]


def test_isovist_in_a_rectangle():
    walls = polyline_segments([(0, 0), (6000, 0), (6000, 4000), (0, 4000)], closed=True)
    p = cast(3000, 2000, walls).properties()
    assert abs(p["area_m2"] - 24) < 0.05
    assert p["drift_m"] < 0.01 and p["open_ray_frac"] == 0 and p["occlusivity"] == 0


def test_similarity():
    a = np.array([(0, 0), (5, 0), (5, 3), (1, 7)], float)
    b = 2.5 * a @ rotation(0.4).T + [10, -3]
    np.testing.assert_allclose(fit_similarity(a, b)(a), b, atol=1e-9)


def test_point_cloud_reader(tmp_path):
    rec = np.array([(1, 2, 3), (4, 5, 6)], dtype=[(c, "<f4") for c in "xyz"])
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n")
    path = tmp_path / "cloud"
    path.write_bytes(head.encode() + rec.tobytes())
    data, n = load(path)
    assert n == 2 and data["z"][1] == 6
