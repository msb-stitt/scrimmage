"""Plugins a robot's config names (scrimmage/plugins.py), in one robot's World and in a MultiWorld."""

import tomllib
from pathlib import Path

import pytest

from scrimmage.multi import MultiWorld
from scrimmage.plugins import PluginError
from scrimmage.protocol import HalState
from scrimmage.world import World
from test_multi import KITBOT, kitbot_cfg, two_kitbots

TESTS = Path(__file__).resolve().parent


def fake_robot(**extra) -> World:
    cfg = tomllib.loads((TESTS / "fake_robot.toml").read_text())
    return World({**cfg, **extra}, TESTS)


def test_a_plugin_acts_on_the_physics_before_each_step():
    pushed = fake_robot(plugin=[{"class": "plugin_probe:Push", "body": "chassis", "force": 200.0}])
    still = fake_robot()
    hal = HalState()
    for _ in range(500):
        pushed.step(hal)
        still.step(hal)
    assert pushed.robot_pose()[0] - still.robot_pose()[0] > 0.05


def test_each_robots_plugins_run_in_order_with_their_hooks_and_readings():
    cfgs = {n: ({**kitbot_cfg(f"{n}_"), "plugin": [{"class": "plugin_probe:Recorder", "device": f"rec_{n}"}]},
                KITBOT.parent) for n in ("a", "b")}
    w = MultiWorld(two_kitbots(), cfgs, timestep=0.001)
    recs = {n: w.robots[n].plugins.plugins[0] for n in cfgs}
    assert all(r.calls == ["reset"] for r in recs.values())      # MultiWorld's reset at its start
    for _ in range(3):
        w.step({})
    for n, r in recs.items():
        assert r.calls == ["reset"] + ["before", "after"] * 3
        assert ("SimDevice", f"rec_{n}", {">steps": 3}) in w.robots[n].sensor_updates(HalState())
        assert w.robots[n].plugins.seconds[0] > 0
    w.reset()
    assert recs["a"].steps == 0 and recs["a"].calls[-1] == "reset"


@pytest.mark.parametrize("entry, message", [
    ({"class": "plugin_probe"}, "package.module:Class"),
    ({"class": "plugin_probe:Nothing"}, "can't be loaded"),
    ({"class": "plugin_probe:Push", "body": "chassis"}, "force"),
    ({"body": "chassis"}, "needs class"),
])
def test_a_plugin_entry_that_cannot_be_made_says_why(entry, message):
    with pytest.raises(PluginError, match=message):
        fake_robot(plugin=[entry])
