"""Position, zone, space syntax and flow measures on the made-up flat."""
import numpy as np
import pytest

from spatial_twin import cast
from spatial_twin.flow import flow_measures, new_information, rate
from spatial_twin.position import position_measures, relational
from spatial_twin.synthetic import DOORS, ROOMS, WALLS
from spatial_twin.syntax import diamond, space_syntax
from spatial_twin.zones import (ZoneMap, door_graph, permeable_graph, threshold_counts,
                                zone_graph, zone_isovist_stats)

MID = (DOORS[:, :2] + DOORS[:, 2:]) / 2


@pytest.fixture(scope="module")
def zm():
    return ZoneMap.from_polygons(ROOMS, res_mm=100)


# ---------------------------------------------------------------- zone grain
def test_envelope_of_a_rectangle(zm):
    e = zm.envelope("Room C")                      # 7 m x 3 m
    assert e["floor_area_m2"] == pytest.approx(21, rel=0.01)
    assert e["perimeter_m"] == pytest.approx(20, rel=0.01)
    # the hull runs through cell centres, half a cell inside the drawn edge, so a
    # raster rectangle reads a little above 1 (the source computes it the same way)
    assert 1 <= e["convexity"] < 1.06
    assert e["jaggedness"] == pytest.approx(20 / (2 * np.sqrt(np.pi * 21)), rel=0.02)
    assert e["aspect_ratio"] == pytest.approx(7 / 3, rel=0.05)


def test_an_l_shape_is_less_convex_and_more_jagged():
    l_shape = {"L": [(0, 0), (6000, 0), (6000, 2000), (2000, 2000), (2000, 6000), (0, 6000)],
               "square": [(10000, 0), (14000, 0), (14000, 4000), (10000, 4000)]}
    z = ZoneMap.from_polygons(l_shape, res_mm=100)
    a, b = z.envelope("L"), z.envelope("square")
    assert a["convexity"] < 0.8 < b["convexity"]
    assert a["jaggedness"] > b["jaggedness"]


def test_zone_isovists_in_a_closed_plan(zm):
    s = zone_isovist_stats(zm, "Hallway", WALLS, spacing_mm=1000)
    assert s["n_samples"] >= 10
    assert s["open_ray_frac"] == 0
    assert s["visual_depth_max_m"] > 10              # sees down the 12 m hallway
    assert 0 < s["occlusivity_mean"] < 1


def test_adjacency_links_rooms_through_a_wall_but_permeability_does_not(zm):
    adj = zone_graph(zm)
    assert "Room B" in adj["Room A"]                # drawn flush, solid wall between
    g = permeable_graph(zm, WALLS, DOORS)
    assert "Room B" not in g["adjacency"]["Room A"]
    assert sorted(g["adjacency"]["Hallway"]) == ["Room A", "Room B", "Room C"]
    assert all(e["basis"] == "door" for e in g["evidence"].values())
    assert g["isolated"] == []


def test_door_counts(zm):
    pairs = door_graph(zm, DOORS)
    assert pairs == {("Hallway", "Room A"): 1, ("Hallway", "Room B"): 1, ("Hallway", "Room C"): 1}
    assert threshold_counts(pairs, zm.names) == {"Room A": 1, "Room B": 1, "Room C": 1, "Hallway": 3}


def test_a_door_probe_does_not_tunnel_through_a_wall(zm):
    from spatial_twin.zones import wall_raster
    fake = np.array([(5000, 6000, 5000, 7000)], float)   # a "door" drawn along the A|B wall
    assert door_graph(zm, fake, walls=wall_raster(zm, WALLS)) == {}


def test_split_pieces_become_separate_nodes():
    rooms = {"Store": [(0, 0), (2000, 0), (2000, 2000), (0, 2000)] ,
             "Hall": [(2000, 0), (10000, 0), (10000, 2000), (2000, 2000)]}
    rooms["Store"] = rooms["Store"]
    z = ZoneMap.from_polygons({**rooms, "Store2": [(10000, 0), (12000, 0), (12000, 2000), (10000, 2000)]})
    z.names[z.names.index("Store2")] = "Store"     # one name, two separate outlines
    z.grid[z.grid == 3] = 1
    walls = np.array([(2000, 0, 2000, 800), (2000, 1800, 2000, 2000), (10000, 0, 10000, 800),
                      (10000, 1800, 10000, 2000)], float)
    g = permeable_graph(z, walls, np.array([(2000, 800, 2000, 1800), (10000, 800, 10000, 1800)], float))
    assert g["pieces"]["Store"] == 2
    assert {"Store #1", "Store #2"} <= set(g["nodes"])
    assert "Store #2" not in g["adjacency"]["Store #1"]   # no teleport between the pieces


# ---------------------------------------------------------------- space syntax
def test_diamond_value():
    assert diamond(3) == pytest.approx(2 * (3 * (np.log2(5 / 3) - 1) + 1) / 2)
    assert np.isnan(diamond(2))


def test_star_graph():
    adj = {"hub": ["a", "b", "c", "d"], "a": ["hub"], "b": ["hub"], "c": ["hub"], "d": ["hub"]}
    s = space_syntax(adj)
    hub, leaf = s["zones"]["hub"], s["zones"]["a"]
    assert hub["connectivity"] == 4 and hub["mean_depth"] == 1
    assert hub["integration"] == np.inf             # one step from everything
    assert leaf["mean_depth"] == pytest.approx((1 + 2 * 3) / 4)
    assert hub["choice"] == 6 and leaf["choice"] == 0   # all 6 leaf pairs pass the hub
    assert hub["control"] == 4 and leaf["control"] == pytest.approx(0.25)


def test_line_graph_and_intelligibility():
    adj = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c", "e"], "e": ["d"]}
    s = space_syntax(adj)
    z = s["zones"]
    assert z["c"]["integration"] > z["b"]["integration"] > z["a"]["integration"]
    assert z["c"]["choice"] == 4
    assert s["intelligibility"] > 0.8               # degree and integration agree on a line


def test_syntax_on_the_flat(zm):
    s = space_syntax(permeable_graph(zm, WALLS, DOORS)["adjacency"])
    assert max(s["zones"], key=lambda k: s["zones"][k]["connectivity"]) == "Hallway"
    assert s["zones"]["Hallway"]["integration"] == np.inf


# ---------------------------------------------------------------- position grain
def test_relational_in_the_hallway():
    x, y = 2550, 4000                               # right under the Room A door
    v = cast(x, y, WALLS)
    r = relational(x, y, v, MID, {"Room C": (3500, 1500)})
    assert r["threshold_dist_m"] == pytest.approx(1.0, abs=0.01)
    assert r["threshold_bearing_deg"] == pytest.approx(90, abs=0.5)
    assert r["thresholds_visible_n"] == 3
    assert r["dist_to_room_c_m"] == pytest.approx(np.hypot(950, 2500) / 1000)


def test_a_closed_room_sees_only_its_own_door():
    x, y = 500, 2500                                # a corner of Room C
    r = relational(x, y, cast(x, y, WALLS), MID)
    assert r["thresholds_visible_n"] == 1


def test_position_measures():
    m, v = position_measures(6000, 4000, WALLS, MID)
    assert m["visual_depth_m"] == m["iso_radial_mean_m"]
    assert m["iso_radial_min_m"] <= m["iso_radial_mean_m"] <= m["iso_radial_max_m"]
    assert {"iso_radial_var_m2", "iso_radial_skew", "threshold_dist_m"} <= set(m)


# ---------------------------------------------------------------- flow
def test_new_information_bounds():
    sq = np.array([(0, 0), (2, 0), (2, 2), (0, 2)], float)
    assert new_information(sq, sq) == 0
    assert new_information(sq + 10, sq) == 1
    assert new_information(np.array([(0, 0), (4, 0), (4, 2), (0, 2)], float), sq) == pytest.approx(0.5, abs=0.06)


def test_stepping_through_a_door_opens_new_space():
    inside = cast(2550, 3500, WALLS).polygon_m
    through = cast(2550, 5600, WALLS).polygon_m
    assert new_information(through, inside) > 0.3


def test_flow_is_gated_at_gaps():
    t = np.array([0, 1, 2, 10, 11], float)
    x = np.array([1000, 2000, 3000, 4000, 5000], float)
    y = np.full(5, 4000.0)
    polys = [cast(xx, yy, WALLS).polygon_m for xx, yy in zip(x, y)]
    f = flow_measures(t, x, y, polys)
    assert np.isnan(f["flow_dt_s"][0]) and np.isnan(f["flow_dt_s"][3])   # first row, and the 8 s gap
    assert f["flow_speed_mps"][1] == pytest.approx(1.0)
    valid = np.array([1, 1, 0, 1, 1], bool)
    g = flow_measures(t, x, y, polys, valid=valid)
    assert np.isnan(g["flow_dt_s"][2]) and np.isfinite(g["flow_dt_s"][4])


def test_rate_uses_the_gated_step():
    r = rate([0.1, 0.3, 0.2, 0.9], [np.nan, 2.0, np.nan, 1.0])
    assert np.isnan(r[0]) and np.isnan(r[2])
    assert r[1] == pytest.approx(0.1) and r[3] == pytest.approx(0.7)
