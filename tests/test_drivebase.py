"""The drivebase plugin (scrimmage/drivebase.py): a robot's motion from MuJoCo's truth, and placing it."""

import math

import pytest

from scrimmage.drivebase import TYPE
from scrimmage.multi import MultiWorld
from scrimmage.protocol import HalState, frame
from test_multi import KITBOT, drive, kitbot_cfg, two_kitbots

WHEEL_RADIUS = 0.0762


def world() -> MultiWorld:
    def cfg(n: str) -> dict:
        return {**kitbot_cfg(f"{n}_"), "plugin": [{
            "class": "scrimmage.drivebase:Drivebase", "device": f"drive_{n}",
            "wheels": [f"{n}_left_center", f"{n}_right_center"], "wheel_radius": WHEEL_RADIUS}]}
    return MultiWorld(two_kitbots(), {n: (cfg(n), KITBOT.parent) for n in ("a", "b")}, timestep=0.001)


def reading(w: MultiWorld, name: str) -> dict:
    (typ, dev, data), = w.robots[name].plugins.sensor_updates(HalState())
    assert (typ, dev) == (TYPE, f"drive_{name}")
    return data


def test_it_reports_each_robots_own_pose_travel_and_wheels():
    w = world()
    start = {n: w.robots[n].robot_pose() for n in w.robots}
    a = HalState()
    drive(a, 0.5)
    for _ in range(1000):
        w.step({"a": a})
    r = reading(w, "a")
    x, y, h = w.robots["a"].robot_pose()
    assert (r[">x"], r[">y"]) == pytest.approx((x, y), abs=1e-9)
    assert r[">heading"] == pytest.approx(h, abs=1e-9)
    moved = x - start["a"][0]
    assert moved > 0.3
    assert r[">travel_x"] == pytest.approx(moved, rel=0.02)
    assert abs(r[">travel_y"]) < 0.01 and abs(r[">turned"]) < 0.02
    assert r[">rolled0"] == pytest.approx(moved, rel=0.05) and r[">rolled1"] == pytest.approx(moved, rel=0.05)
    assert r[">vx"] > 0.3
    still = reading(w, "b")
    assert abs(still[">travel_x"]) < 0.01 and abs(still[">rolled0"]) < 0.01


def test_the_robot_program_places_its_robot_once_for_each_placing():
    w = world()
    hal = HalState()
    hal.apply(frame(TYPE, "drive_a", {"<place_x": 3.0, "<place_y": 2.0, "<place_heading": math.pi / 2,
                                      "<place_seq": 1}))
    w.step({"a": hal})
    r = reading(w, "a")
    assert (r[">x"], r[">y"], r[">heading"]) == pytest.approx((3.0, 2.0, math.pi / 2), abs=1e-3)
    assert r[">place_seq"] == 1
    drive(hal, 0.5)                       # the same placing again is not taken: the robot drives off it
    for _ in range(500):
        w.step({"a": hal})
    assert reading(w, "a")[">y"] > 2.1


def test_a_wheel_that_is_not_in_the_model_is_named():
    cfg = {**kitbot_cfg("a_"), "plugin": [{"class": "scrimmage.drivebase:Drivebase", "device": "d",
                                           "wheels": ["a_nothing"], "wheel_radius": 0.05}]}
    with pytest.raises(ValueError, match="a_nothing"):
        MultiWorld(two_kitbots({"a": 4.0}), {"a": (cfg, KITBOT.parent)}, timestep=0.001)
